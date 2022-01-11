import lmdb
import os
import cv2
import argparse
import pickle
from tqdm import tqdm
import numpy as np
from glob import glob
from joblib import delayed, Parallel
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset.process_data.process_video import get_video_frames_cv, compute_TVL1
from sts.motion_sts import compute_motion_boudary, motion_mag_downsample, zero_boundary


def create_lmdb_video_dataset_rgb(dataset, root_path, dst_path, workers=-1, quality=100, video_type='mp4'):

    if dataset == 'kinetics': video_type = 'mp4' 
    else: video_type = 'avi'
    
    videos = glob(os.path.join(root_path,'*/*.{}'.format(video_type)))
    print('begin')
    
    def make_video(video_path, dst_path):
        vid_names = '/'.join(video_path.split('/')[-2:])
        dst_file = os.path.join(dst_path, vid_names[:-4])
        os.makedirs(dst_file, exist_ok=True)
        try:
            frames = get_video_frames_cv(video_path, dataset)
        except Exception as e:
            return
        else:
            if frames == None: return 
        
        _, frame_byte = cv2.imencode('.jpg', frames[0],  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        env = lmdb.open(dst_file, frame_byte.nbytes * len(frames) * 50)
        frames_num = len(frames)
        for i in range(frames_num):
            txn = env.begin(write=True)
            key = 'image_{:05d}.jpg'.format(i+1)
            frame = frames[i]
            _, frame_byte = cv2.imencode('.jpg', frame,  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            txn.put(key.encode(), frame_byte)
            txn.commit()
        with open(os.path.join(dst_file, 'split.txt'),'w') as f:
            f.write(str(frames_num))

    Parallel(n_jobs=workers)(delayed(make_video)(vp, dst_path) for vp in tqdm(videos, total=len(videos)))


def create_lmdb_video_dataset_optical_flow(dataset, root_path, dst_path, workers=-1, quality=100):

    videos = glob(os.path.join(root_path,'*/*'))
    print('begin')
    
    def make_video_optical_flow(video_path, dst_path):
        vid_names = '/'.join(video_path.split('/')[-2:])
        dst_file = os.path.join(dst_path, vid_names)
        os.makedirs(dst_file, exist_ok=True)
        
        # load rgb frames from lmdb. You can change the code to load it in another way
        frames = []
        env = lmdb.open(video_path, readonly=True)
        txn = env.begin(write=False)
        for k,v in txn.cursor():
            frame_decode = cv2.imdecode(np.frombuffer(v, np.uint8), cv2.IMREAD_COLOR) 
            frames.append(frame_decode)
        env.close()

        height, width, _ = frames[0].shape
        empty_img = 128 * np.ones((int(height),int(width),3)).astype(np.uint8)
        # extract flows
        flows = []
        for idx, frame in enumerate(frames):
            if idx == 0: 
                pre_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                continue
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = compute_TVL1(pre_frame, frame_gray)
            # create flow frame with 3 channel
            flow_img = empty_img.copy()
            flow_img[:,:,0:2] = flow
            flows.append(flow_img)
            pre_frame = frame_gray

        # save flows
        _, frame_byte = cv2.imencode('.jpg', flows[0],  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        env = lmdb.open(dst_file, frame_byte.nbytes * len(flows) * 50)
        frames_num = len(flows)

        for i in range(frames_num):
            txn = env.begin(write=True)
            key = 'image_{:05d}.jpg'.format(i+1)
            flow_img = flows[i]
            _, frame_byte = cv2.imencode('.jpg', flow_img,  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            txn.put(key.encode(), frame_byte)
            txn.commit()
        with open(os.path.join(dst_file, 'split.txt'),'w') as f:
            f.write(str(frames_num))

    Parallel(n_jobs=workers)(delayed(make_video_optical_flow)(vp, dst_path) for vp in tqdm(videos, total=len(videos)))


def create_lmdb_video_dataset_flow_mag(dataset, root_path, dst_path, workers=-1, ws=8, quality=100):
    videos = glob(os.path.join(root_path,'*/*'))
    print('begin')

    def make_video_flow_mag(video_path, dst_path):
        vid_names = '/'.join(video_path.split('/')[-2:])
        dst_file = os.path.join(dst_path, vid_names)
        os.makedirs(dst_file, exist_ok=True)

        if dataset == 'kinetics': ws = 4
        else: ws = 8

        # load flows from lmdb. You can change the code to load it in another way
        if not os.path.exists(os.path.join(video_path, 'data.mdb')): return
        flows = []
        env = lmdb.open(video_path, readonly=True)
        txn = env.begin(write=False)
        for k,v in txn.cursor():
            flow_decode = cv2.imdecode(np.frombuffer(v, np.uint8), cv2.IMREAD_COLOR) 
            flows.append(flow_decode)
        env.close()
        duration = len(flows)

        # compute frame mag offline with a sliding window
        env_flow_mag = lmdb.open(dst_file, readonly=False, map_size=int(2e12))
        for idx in range(1, duration+1):
            txn_flow_mag = env_flow_mag.begin(write=True)
            if ws == 1:
                flow_clip = [flows[idx-1]]
            else:
                if idx - ws//2 >= 0 and idx + ws//2 <= duration:
                    flow_clip = flows[idx - ws//2 : idx + ws//2]
                elif idx - ws//2 >= 0 and idx + ws//2 > duration:
                    flow_clip = flows[-ws:]
                elif idx + ws//2 <= duration and idx - ws//2 < 0:
                    flow_clip = flows[:ws]
                else:
                    flow_clip = flows[:]

            flows_u = list([cv2.resize(flow[:,:,0], (224, 224), interpolation=cv2.INTER_CUBIC).astype(np.float32) for flow in flow_clip])
            flows_v = list([cv2.resize(flow[:,:,1], (224, 224), interpolation=cv2.INTER_CUBIC).astype(np.float32) for flow in flow_clip])

            _, _, mb_x_u, mb_y_u = compute_motion_boudary(flows_u)
            _, _, mb_x_v, mb_y_v = compute_motion_boudary(flows_v)

            frame_mag_u, _ = cv2.cartToPolar(mb_x_u, mb_y_u, angleInDegrees=True)
            frame_mag_v, _ = cv2.cartToPolar(mb_x_v, mb_y_v, angleInDegrees=True)
            frame_mag = (frame_mag_u + frame_mag_v) / 2
            
            # zero boundary
            frame_mag = zero_boundary(frame_mag)

            # downsample to match the fearture size of backbone output
            if ws == 1:
                frame_mag_down = (motion_mag_downsample(frame_mag, 7, 224) * 5).astype(np.uint8)
            else:
                frame_mag_down = motion_mag_downsample(frame_mag, 7, 224).astype(np.uint8)

            # save frame mag
            key = 'image_{:05d}.jpg'.format(idx)
            _, frame_byte = cv2.imencode('.jpg', frame_mag_down,  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            txn_flow_mag.put(key.encode(), frame_byte)
            txn_flow_mag.commit()
        
        with open(os.path.join(dst_file, 'split.txt'),'w') as f:
            f.write(str(duration))

    Parallel(n_jobs=workers)(delayed(make_video_flow_mag)(vp, dst_path) for vp in tqdm(videos, total=len(videos)))

def create_lmdb_video_dataset_clip_mag(dataset, root_path, dst_path, clip_length=16, steps=2, workers=-1):
    videos = glob(os.path.join(root_path,'*/*'))

    print('begin')

    def make_video_clip_mag(video_path):
        vid_names = '/'.join(video_path.split('/')[-2:])

        # load flows from lmdb. You can change the code to load it in another way
        flows = []
        if not os.path.exists(os.path.join(video_path, 'data.mdb')): return

        env = lmdb.open(video_path, readonly=True, lock=False)
        txn = env.begin(write=False)
        for k,v in txn.cursor():
            flow_decode = cv2.imdecode(np.frombuffer(v, np.uint8), cv2.IMREAD_COLOR) 
            flows.append(flow_decode)
        env.close()
        duration = len(flows)

        clip_mag_list = []
        for idx in range(1, duration+1):
            if idx + clip_length * steps <= duration + 1:
                flow_clip = flows[idx-1 : idx + clip_length * steps - 1]
            else:
                break

            flows_u = list([flow[:,:,0].astype(np.float32) for flow in flow_clip])
            flows_v = list([flow[:,:,1].astype(np.float32) for flow in flow_clip])

            _, _, mb_x_u, mb_y_u = compute_motion_boudary(flows_u)
            _, _, mb_x_v, mb_y_v = compute_motion_boudary(flows_v)

            clip_mag_u, _ = cv2.cartToPolar(mb_x_u, mb_y_u, angleInDegrees=True)
            clip_mag_v, _ = cv2.cartToPolar(mb_x_v, mb_y_v, angleInDegrees=True)
            clip_mag = (clip_mag_u + clip_mag_v) / 2
            
            # zero boundary
            clip_mag = zero_boundary(clip_mag).mean()
            clip_mag_list.append(clip_mag)

        return vid_names, clip_mag_list

    video_clip_mag_dic = {}
    data = Parallel(n_jobs=workers)(delayed(make_video_clip_mag)(vp) for vp in tqdm(videos, total=len(videos)))

    for vid_names, l in data:
        video_clip_mag_dic[vid_names] = l
    
    with open(os.path.join(dst_path, 'video_clip_mag_{}.pickle'.format(dataset)), 'wb') as f:
        pickle.dump(video_clip_mag_dic, f)

def parse_option():
    parser = argparse.ArgumentParser('training')

    # dataset
    parser.add_argument('--root-path', type=str, default='/lr/export2/home/zhangyiheng8/dataset/UCF-101/UCF-101-flow-lmdb240', help='path of original data')
    parser.add_argument('--dst-path', type=str, default='/dev/shm/UCF-101-flow-lmdb240', help='path to store generated data')
    parser.add_argument('--dataset', type=str, default='ucf101', choices=['kinetics','ucf101'], help='dataset to training')
    parser.add_argument('--data-type', type=str, default='mag', choices=['rgb','flow','mag','clip-mag'], help='which data')
    parser.add_argument('--video-type', type=str, default='mp4', choices=['mp4', 'avi'], help='which data')
    parser.add_argument('--num-workers', type=int, default=1, help='num of workers to use')
    parser.add_argument('--clip-length', type=int, default=16, help='num of clip length')
    parser.add_argument('--num-steps', type=int, default=2, help='num of sampling steps')


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args =  parse_option()

    if args.data_type == 'rgb':
        create_lmdb_video_dataset_rgb(args.dataset, args.root_path, args.dst_path, workers=args.num_workers, video_type=args.video_type)
    elif args.data_type == 'flow':
        create_lmdb_video_dataset_optical_flow(args.dataset, args.root_path, args.dst_path, workers=args.num_workers)
    elif args.data_type == 'mag':
        create_lmdb_video_dataset_flow_mag(args.dataset, args.root_path, args.dst_path, workers=args.num_workers)
    elif args.data_type == 'clip-mag':
        create_lmdb_video_dataset_clip_mag(args.dataset, args.root_path, args.dst_path, clip_length=args.clip_length, steps=args.num_steps, workers=args.num_workers)
