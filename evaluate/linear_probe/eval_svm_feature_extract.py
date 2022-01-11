import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as transforms
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset.video_dataset import  VideoRGBTestDataset
from utils.logger import setup_logger
from torch.utils.data.distributed import DistributedSampler
from model.backbone import p3da_resnet50 as backbone
from dataset.augmentations import clip_transforms
import argparse
import json
import numpy as np
from tqdm import tqdm


def parse_option():
    parser = argparse.ArgumentParser('svm eval')
    parser.add_argument('--root-path', type=str, default='/lr/exportssd/home/lirui295/dataset/Kinetics/Kinetics_frame256', help='root director of dataset')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--list-path', type=str,default='../data/K400/val_list.csv', help='path of list file')
    parser.add_argument('--pretrained-model', type=str, default='output/checkpoints/current.pth', help="pretrained model path")
    parser.add_argument('--output-dir', type=str, default='output/eval_output_linear', help='output director')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel', default=0)
    parser.add_argument('--time-dim', type=str, default='T', choices=["T", "C"], help="dimension for time")
    parser.add_argument('--clip-length', type=int, default=16, help='num of clip length')
    parser.add_argument('--num-steps', type=int, default=2, help='num of sampling steps')
    parser.add_argument('--num-segments', type=int, default=20, help='num of segments')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--inflate-weights', type=str, default='')
    parser.add_argument('--dataset', type=str, default='kinetics', choices=['kinetics', 'ucf101'])
    args = parser.parse_args()
    return args

def load_pretrained(args, model):
    ckpt = torch.load(args.pretrained_model, map_location='cpu')

    if 'model' in ckpt:
        state_dict = {k.replace("module.backbone.", ""): v for k, v in ckpt['model'].items()}
    else:
        state_dict = {k.replace("module.backbone.", ""): v for k, v in ckpt.items()}

    [misskeys, unexpkeys] = model.load_state_dict(state_dict, strict=False)

    logger.info(misskeys)
    logger.info(unexpkeys)
    logger.info("==> loaded checkpoint '{}' (epoch {})".format(args.pretrained_model, ckpt['epoch']))


def get_loader(args, mode):

    crop = clip_transforms.ClipCenterCrop((224, 224))
    test_transform = clip_transforms.Compose([
        crop,
        clip_transforms.ToClipTensor(),
        clip_transforms.ClipNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        clip_transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if opt.time_dim == "T" else transforms.Lambda(
            lambda clip: torch.cat(clip, dim=0))
    ])


    dataset = VideoRGBTestDataset(
            num_clips=1,
            list_root=args.list_path, root_path=args.root_path,
            transform=test_transform,
            clip_length=args.clip_length, num_steps=args.num_steps,
            num_segments=args.num_segments,dataset=args.dataset,
            split=mode
        )

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=False)
    
    return loader, len(dataset)


def main(opt):
    data_loader, total_num = get_loader(opt, opt.mode)

    if opt.mode == 'trainsvm':
        opt.mode = 'train'

    model = backbone().cuda()

    load_pretrained(opt, model)
    model.eval()
    global_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))

    logger.info('model init done')
    all_feat = []
    feat_cls = []
    paths = []
    with torch.no_grad():
        for idx, (data, cls, vpath) in tqdm(enumerate(data_loader), total=len(data_loader)):
            is_normal = cls.view(-1).item()
            if is_normal == -1: continue
            data_size = data.size()

            data = data.view((-1, 3, data_size[-3], data_size[-2], data_size[-1]))
            data = data.cuda()
            feat = global_pooling(model(data, layer=5)).squeeze()
            feat_avg = torch.mean(feat, dim=0).view(-1)
            all_feat.append(feat_avg.data.cpu().numpy())
            feat_cls.append(cls.item())
            paths.append(vpath[0])
    
    all_feat = np.stack(all_feat, axis=0)
    all_feat_cls = np.array(feat_cls)
    np.save(os.path.join(opt.output_dir, 'feature_{}_{}.npy'.format(opt.mode, opt.local_rank)), all_feat)
    np.save(os.path.join(opt.output_dir, 'feature_{}_cls_{}.npy'.format(opt.mode, opt.local_rank)), all_feat_cls)
    with open(os.path.join(opt.output_dir,'paths_{}_{}.json'.format(opt.mode, opt.local_rank)),'w') as f:
        json.dump(paths, f)

    if dist.get_rank() == 0:
        np.save(os.path.join(opt.output_dir, 'vid_num_{}.npy'.format(opt.mode)), np.array([total_num]))


if __name__ == '__main__':
    opt = parse_option()

    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    os.makedirs(opt.output_dir, exist_ok=True)
    logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name="extract-feature")
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "config.json")
        with open(path, "w") as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))
    
    main(opt)