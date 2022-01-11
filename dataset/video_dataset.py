import torch.utils.data
import os
import random
import torch
import numpy as np
import lmdb
import math
import io
import sys
sys.path.insert(0, '../sts')
sys.path.insert(0, '../utils')
from sts.motion_sts import *
from dataset.augmentations import clip_transforms
from PIL import Image
from torchvision import transforms
import json
import csv
import pickle


class VideoDataset(torch.utils.data.Dataset):

    def __init__(self, list_root,
                    transform,
                    root_path,
                    root_path_flow=None, root_path_mag=None,
                    clip_length=1, num_steps=1, num_segments=1, num_channels=3,
                    dataset='ucf101',
                    data_form='lmdb',
                    split='train',
                    with_motion=False
                    ):

        super(VideoDataset, self).__init__()
        self.transform = transform
        self.root_path = root_path
        self.root_path_flow = root_path_flow
        self.root_path_mag = root_path_mag
        self.list_root = list_root
        self.clip_length = clip_length
        self.num_steps = num_steps
        self.num_segments = num_segments
        self.num_channels = num_channels
        self.dataset = dataset
        self.data_form = data_form
        self.split = split
        self.with_motion = with_motion
        self.length_ext =  self.clip_length * self.num_steps
        self.samples = self._load_list_csv()

        # for motion-focus temporal sampling
        with open(os.path.join(os.path.dirname(self.root_path), 'video_clip_mag_{}.pickle'.format(self.dataset)), 'rb') as f:
            self.video_clip_mags = pickle.load(f)

    def _load_list_csv(self):
        """ load annotation from csv """
        samples = []
        # csv format
        cols_name = ['duration_flow', 'duration_rgb', 'label', 'video_class',  'video_path',  'vname'] 
        with open(self.list_root) as f:
            f_csv = csv.DictReader(f, fieldnames=cols_name, delimiter=' ')
            rows = list(f_csv)
            for idx, sample in enumerate(rows):
                if idx == 0: continue

                # filter out missing data especially for kinetics dataset
                sample['video_path'] = os.path.join(self.root_path, sample['video_path'])
                if self.with_motion:
                    if not os.path.exists(os.path.join(self.root_path_mag, sample['video_path'])): continue
                else:
                    if not os.path.exists(sample['video_path']): continue

                duration = int(sample["duration_rgb"])
                if self.split == 'train':
                    # filter out short video for training
                    if self.dataset == 'ucf101':
                        if duration >= 2 * self.length_ext:
                            samples.append(sample)
                    elif duration > self.length_ext:
                        samples.append(sample)
                else:
                    # for short video, we duplicate it when test
                    dup = math.ceil(self.length_ext / int(sample['duration_rgb']))
                    sample['dup'] = dup
                    samples.append(sample)

        return samples


    def _parse_rgb_lmdb(self, video_path, offsets, frame_num, clip_length, num_steps, dup_time=1):
        """Return the clip buffer sample from video lmdb."""
        lmdb_env = lmdb.open(video_path, readonly=True, lock=False)
        with lmdb_env.begin() as lmdb_txn:
            frame_list_all = []
            for idx, offset in enumerate(offsets):
                frame_list = []
                for frame_id in range(offset, offset + num_steps * clip_length, num_steps):
                    if frame_id > frame_num:
                        frame_id = frame_num
                    frame_id = math.ceil(frame_id / dup_time)
                    bio = io.BytesIO(lmdb_txn.get('image_{:05d}.jpg'.format(frame_id).encode()))
                    image = Image.open(bio).convert('RGB')
                    frame_list.append(image)
                frame_list_all.append(frame_list)
        lmdb_env.close()
        return frame_list_all

    def _parse_rgb_jpg(self, video_path, offsets, frame_num, clip_length, num_steps, dup_time=1):
        """Return the clip buffer sample from video jpgs."""
        frame_list_all = []
        for idx, offset in enumerate(offsets):
            frame_list = []
            for frame_id in range(offset, offset + num_steps * clip_length, num_steps):
                if frame_id > frame_num:
                    frame_id = frame_num
                frame_id = math.ceil(frame_id / dup_time)
                frame_path = os.path.join(video_path, 'image_{:05d}.jpg'.format(frame_id))
                image = Image.open(frame_path).convert('RGB')
                frame_list.append(image)
            frame_list_all.append(frame_list)

        return frame_list_all

    def _parse_flow_two_channel_lmdb(self, video_path, offsets, frame_num, clip_length, num_steps):
        """ Return the clip buffer sample with seprate channel from video flow lmdb  """
        flow_list_all = []
        for offset in offsets:
            flows_u = []
            flows_v = []
            lmdb_env = lmdb.open(video_path, readonly=True, lock=False)
            with lmdb_env.begin() as lmdb_txn:
                for frame_id in range(offset, offset + num_steps * clip_length, num_steps):
                    if frame_id > frame_num:
                        frame_id = frame_num
                    bio = io.BytesIO(lmdb_txn.get('image_{:05d}.jpg'.format(frame_id).encode()))
                    frame_flow = Image.open(bio).convert('RGB')
                    _, v, u = frame_flow.split()
                    flows_u.append(u)
                    flows_v.append(v)
            flow_list_all.append([flows_u, flows_v])
        return flow_list_all

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        raise NotImplementedError


class VideoTrainDataset(VideoDataset):

    def temporal_sampling(self, duration, length_ext, num_segment=1):
        offsets = []
        for i in range(num_segment):
            offsets.append(random.randint(1, duration - length_ext))
        return offsets

    def temporal_sampling_triplet(self, order, duration, length_ext):
        offsets = []
        if order:
            if duration // length_ext >= 2:
                offsets.append(random.randint(1, duration // 2 - length_ext + 1))
                offsets.extend(sorted([random.randint(duration // 2, duration - length_ext),
                                       random.randint(duration // 2, duration - length_ext)]))
            else:
                offsets = sorted(self.temporal_sampling(duration, length_ext, 3))
        else:
            if duration // length_ext >= 2:
                offsets.append(random.randint(duration // 2, duration - length_ext))
                offsets.extend(sorted([random.randint(1, duration // 2 - length_ext + 1),
                                       random.randint(1, duration // 2 - length_ext + 1)]))
            else:
                offsets = sorted(self.temporal_sampling(duration, length_ext, 3), reverse=True)
                offsets[2], offsets[1] = offsets[1], offsets[2]
            
        return offsets

    def motion_focus_temporal_sampling(self, order, duration, length_ext, video_path=None, threshold=0.5):
        video_name = '/'.join(video_path.split('/')[-2:])
        clip_mags = np.array(self.video_clip_mags[video_name])
        max_clip_idxs = (np.argsort(clip_mags)[::-1] + 1).tolist()

        if duration // length_ext >= 2:
            idxs_1 = list(range(1, duration // 2 - length_ext +1))
            idxs_2 = list(range(duration // 2, duration - length_ext))
            max_idxs_1 = list([ i for i in max_clip_idxs if i in idxs_1 ])
            max_idxs_2 = list([ i for i in max_clip_idxs if i in idxs_2 ])

            num_max = int(len(idxs_1) * threshold) + 1
            max_idxs_1 = max_idxs_1[:num_max]
            max_idxs_2 = max_idxs_2[:num_max]
            if len(max_idxs_1) == 0 or len(max_idxs_2) == 0 :
                max_idxs_1 = [1]
                max_idxs_2 = [length_ext]

        offsets = []
        if order:
            if duration // length_ext >= 2:
                offsets.append(random.choice(max_idxs_1))
                offsets.extend(sorted([random.choice(max_idxs_2), random.choice(max_idxs_2)]))
            else:
                offsets = sorted(self.temporal_sampling(duration, length_ext, 3))
        else:
            if duration // length_ext >= 2:
                offsets.append(random.choice(max_idxs_2))
                offsets.extend(sorted([random.choice(max_idxs_1), random.choice(max_idxs_1)]))
            else:
                offsets = sorted(self.temporal_sampling(duration, length_ext, 3), reverse=True)
                offsets[2], offsets[1] = offsets[1], offsets[2]

        return offsets


    def _parse_sample_str(self, sample, order=None, idx=0):

        # read basic data for video
        duration = int(sample['duration_rgb'])

        label = int(sample['label'])
        vname =  1
        video_path =  sample['video_path']

        # sample frames offsets
        length_ext = self.clip_length * self.num_steps
        video_name = video_path.split('/')[-1]

        org_duration = duration
        if duration < length_ext:
            duration = length_ext

        if self.with_motion:
            if random.random() <= 0.7:
                offsets = self.motion_focus_temporal_sampling(order, duration, length_ext, video_path, threshold=0.5)
            else:
                offsets = self.temporal_sampling_triplet(order, duration, length_ext)
        else:
            offsets = self.temporal_sampling(duration, length_ext, num_segment=self.num_segments)

        return video_path, offsets, label, org_duration, vname


class VideoTestDataset(VideoDataset):
    def __init__(self, num_clips=1, **kwargs):
        super(VideoTestDataset, self).__init__(**kwargs)
        self.num_clips = num_clips

    def __len__(self):
        return len(self.samples) * self.num_clips

    def _parse_sample_str(self, sample, clip_idx, idx):

        # read basic data for video
        duration = int(sample['duration_rgb'])
        dup = sample['dup']
        duration = duration * dup
        label = int(sample['label'])
        vname = sample['vname']
        video_path = sample['video_path']

        # sample frames offsets
        offsets = []
        length_ext = self.clip_length * self.num_steps
        ave_duration = duration // self.num_segments

        org_duration = duration
        if duration < length_ext:
            duration = length_ext

        if ave_duration >= length_ext:
            for i in range(self.num_segments):
                offsets.append(int(i * ave_duration) + 1)
        else:
            if duration >= length_ext:
                float_ave_duration = float(duration - length_ext) / float(self.num_segments - 1) # left length_ext frames for last clip
                for i in range(self.num_segments):
                    offsets.append(int(i * float_ave_duration) + 1)
            else:
                raise NotImplementedError
        return video_path, offsets, label, org_duration, vname, dup

class VideoRGBTrainDataset(VideoTrainDataset):
    def __getitem__(self, item):

        video_path, offsets, label, org_duration, vname = self._parse_sample_str(self.samples[item])

        if self.data_form == 'lmdb':
            frame_list_all = self._parse_rgb_lmdb(video_path, offsets, org_duration, self.clip_length, self.num_steps)
        elif self.data_form == 'rgb':
            frame_list_all = self._parse_rgb_jpg(video_path, offsets, org_duration, self.clip_length, self.num_steps)

        input_list = []
        for idx, frame_list in enumerate(frame_list_all):
            frame_list = self.transform(frame_list)[0]
            input_list.append(frame_list)

        if len(input_list) == 1:
            return input_list[0], label, video_path
        else:
            return torch.stack(input_list,0), label, video_path

class VideoRGBTestDataset(VideoTestDataset):

    def __getitem__(self, item):
        item_in = item % len(self.samples)
        item_out = item / len(self.samples)

        video_path, offsets, label, org_duration, vname, dup = self._parse_sample_str(self.samples[item_in], item_out, item)

        if self.data_form == 'lmdb':
            frame_list_all = self._parse_rgb_lmdb(video_path, offsets, org_duration, self.clip_length, self.num_steps, dup_time=dup)
        elif self.data_form == 'rgb':
            frame_list_all = self._parse_rgb_jpg(video_path, offsets, org_duration, self.clip_length, self.num_steps, dup_time=dup)

        input_list = []
        for idx, frame_list in enumerate(frame_list_all):
            frame_list = self.transform(frame_list)[0]
            input_list.append(frame_list)

        return torch.stack(input_list,0), label, video_path

class VideoRGBTrainDataset_Motion(VideoTrainDataset):
    def __init__(self, motion_focus_spatial_crop, input_size, mag_size, **kwargs):
        super(VideoRGBTrainDataset_Motion, self).__init__(with_motion=True, **kwargs)
        self.motion_focus_spatial_crop = motion_focus_spatial_crop
        self.input_size = input_size
        self.mag_size = mag_size

        self.flip_trans = clip_transforms.ClipRandomHorizontalFlip()
        self.mag_trans = clip_transforms.Compose([
                                clip_transforms.ToClipTensor(),
                                clip_transforms.Lambda(lambda clip: torch.stack(clip, dim=1))
                                ])
    
    def __getitem__(self, item):
        if random.randint(0, 1) == 0:
            order_cls = 1
        else:
            order_cls = 0

        video_path, offsets, label, org_duration, vname = self._parse_sample_str(self.samples[item], order_cls, item)
        if self.data_form == 'lmdb':
            frame_list_all = self._parse_rgb_lmdb(video_path, offsets, org_duration, self.clip_length, self.num_steps)
        elif self.data_form == 'rgb':
            frame_list_all = self._parse_rgb_jpg(video_path, offsets, org_duration, self.clip_length, self.num_steps)

        mags = []
        flows = []
        path_flow_mag = os.path.join(self.root_path_mag, '/'.join(video_path.split('/')[-2:]))
        path_flow = os.path.join(self.root_path_flow, '/'.join(video_path.split('/')[-2:]))

        for idx, offset in enumerate(offsets):
            mags.append(self._parse_rgb_lmdb(path_flow_mag, [offset], org_duration, self.clip_length, self.num_steps)[0])
            if idx == 0:
                flows.append(self._parse_flow_two_channel_lmdb(path_flow, [offset], org_duration, self.clip_length, self.num_steps)[0])

        k_list = []
        for idx,frame_list in enumerate(frame_list_all):
            if idx == 0:
                q_aug1, flows_q1_u, flows_q1_v = self.motion_focus_spatial_trans(frame_list, mags[idx], flows[idx])
                q_aug2, _, _ = self.motion_focus_spatial_trans(frame_list, mags[idx])

            else:
                aug, _, _, = self.motion_focus_spatial_trans(frame_list, mags[idx])
                k_list.append(aug)

        # calc ST-motoin maps base on croppped flows
        st_motion_mags = self.get_st_motion_mag(flows_q1_u, flows_q1_v) 
        st_motion_mags = self.mag_trans(st_motion_mags)[0]

        return q_aug1, st_motion_mags.squeeze(0), q_aug2, \
                k_list[0], k_list[1],  order_cls


    def motion_focus_spatial_trans(self, image_list,  mag, flows=None):
        
        motion_aug, flows_u, flows_v = self.motion_focus_spatial_crop(image_list, mag, flows=flows)
        aug, is_flip = self.transform(motion_aug)
        if flows != None:
            flows_u, _ = self.flip_trans(flows_u, is_flip)
            flows_v, _ = self.flip_trans(flows_v, is_flip)
        return aug, flows_u, flows_v

    def get_st_motion_mag(self, flows_u, flows_v):

        flows_u = np.stack([np.array(flow) for flow in flows_u], 0).astype(np.float32)
        flows_v = np.stack([np.array(flow) for flow in flows_v], 0).astype(np.float32)

        flows = [flows_u, flows_v]

        clip_per_frames = flows_v.shape[0] // 4
        st_mags = [[] for i in range(4)]

        for i in range(2):
            flows_ = flows[i]
            for idx, j in enumerate(range(0, clip_per_frames * 4, clip_per_frames)):
                flow_clip = flows_[j:j+clip_per_frames]
                clip_mag = motion_sts(flow_clip, self.mag_size, self.input_size)
                st_mags[idx].append(clip_mag)

        st_mags = list(map(lambda x: sum(x).astype(np.float32), st_mags))

        return st_mags
