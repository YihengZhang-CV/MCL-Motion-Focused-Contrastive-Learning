import argparse
import numpy as np
import os

def parse_option():
    parser = argparse.ArgumentParser('training')
    
    parser.add_argument('--num-gpu', type=int, default=4, help='num of gpu')
    parser.add_argument('--num-crop', type=int, default=3, help='num of crop')
    parser.add_argument('--num-classes', type=int, required=True, help='num of predict classes')
    parser.add_argument('--num-clips', type=int, default=20, help='num of sampled clips')
    parser.add_argument('--output-dir', type=str, default='../../output/eval_output_finetune', help='output director')
    parser.add_argument('--list-file', type=str, required=True, help='list of dataset')
    
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    opt = parse_option()
    num_gpu = opt.num_gpu
    num_crop = opt.num_crop
    num_cls = opt.num_classes
    num_clip = opt.num_clips
    score_dir = opt.output_dir
    list_dir = opt.list_file
    
    for crop_id in range(num_crop):
        all_num = 0
        all_data = []
        for gpu_id in range(num_gpu):
            all_data.append(np.load(os.path.join(score_dir, 'all_scores_' + str(crop_id * num_gpu + gpu_id) +'.npy')))
            all_num += all_data[-1].shape[0]
    
        merge_data = np.empty((all_num, num_cls))
        for gpu_id in range(num_gpu):
            merge_data[gpu_id::num_gpu, :] = all_data[gpu_id]

        # make ave score
        num_video = all_num // num_clip
        merge_data = merge_data[0:num_video * num_clip, :]
        if crop_id == 0:
            reshape_data = np.zeros((num_video, num_clip, num_cls))
        reshape_data += np.reshape(merge_data, (num_video, num_clip, num_cls))  / num_crop

    # make gt
    gt = np.zeros((num_video,))
    lines = open(list_dir, 'r').readlines()
    for idx, line in enumerate(lines):
        ss = line.split(' ')
        label = ss[-1]
        gt[idx] = int(label)

    pred = (reshape_data.mean(axis=1)).argmax(axis=1)
    acc = (pred == gt).mean()

    print(acc)
