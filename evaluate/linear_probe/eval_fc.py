import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataset.video_dataset import VideoRGBTrainDataset, VideoRGBTestDataset
from utils.logger import setup_logger
from utils.util import *
from torch.utils.data.distributed import DistributedSampler
from model.encoder import Encoder
from dataset.augmentations import clip_transforms
from utils.lr_scheduler import get_scheduler

import json
import os
import argparse
from tqdm import tqdm


def parse_option():
    parser = argparse.ArgumentParser('fc test')
    parser.add_argument('--root-path', type=str, default='/lr/export/home/lirui295/dataset/Kinetics/Kinetics_frame256', help='root director of dataset')
    parser.add_argument('--list-path', type=str,default='data/kinetics/train_list_no_dup.csv', help='path of list file')
    parser.add_argument('--list-path-test', type=str,default='data/kinetics/val_list_no_dup.csv', help='path of list file')
    parser.add_argument('--pretrained-model', type=str, default='output/checkpoints/current.pth', help="pretrained model path")
    parser.add_argument('--output-dir', type=str, default='output/eval_output_linear', help='output director')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel', default=0)
    parser.add_argument('--time-dim', type=str, default='T',
                        choices=["T", "C"], help="dimension for time")
    parser.add_argument('--clip-length', type=int, default=32, help='num of clip length')
    parser.add_argument('--lr', type=float, default=16, help='num of sampling steps')
    parser.add_argument('--batch-size', type=int, default=32, help='num of batch size')
    parser.add_argument('--batch-size-test', type=int, default=1, help='num of batch size for evaluation')
    parser.add_argument('--trans', type=str, default='0-5', help='select augmentations')

    parser.add_argument('--epochs', type=int, default=100, help='num of training epoch')
    parser.add_argument('--warmup-epoch', type=int, default=0, help='num of warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--weight-decay', type=float, default=0, help='weight decay')
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--lr-decay-epochs', type=int, default=[60,80], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--num-steps', type=int, default=2, help='num of sampling steps')
    parser.add_argument('--num-segments', type=int, default=1, help='num of segments')
    parser.add_argument('--num-segments-test', type=int, default=20, help='num of segments for evaluation')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--save-freq', type=int, default=50)

    parser.add_argument('--resume', type=str2bool, default='false')
    parser.add_argument('--inflate-weights', type=str2bool, default='false', help='initialize from 2d backbone')

    parser.add_argument('--dataset', type=str, default='kinetics', choices=['kinetics', 'ucf101'],
                        help='dataset to training')
    parser.add_argument('--num-classes', type=int, default=400, help='num of classes')
    parser.add_argument('--seed', type=int, default=2560, help='num of random seed')
    parser.add_argument('--gpu-id', type=int, default=0)

    parser.add_argument('--trainval', type=str, default='train', help='train or test')
    args = parser.parse_args()
    return args

def load_pretrained(args, model):
    ckpt = torch.load(args.pretrained_model, map_location='cpu')

    if 'model' in cpkt:
        state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}
    else:
        state_dict = {k.replace("module.", ""): v for k, v in ckpt.items()}

    [misskeys, unexpkeys] = model.load_state_dict(state_dict, strict=False)
    logger.info(misskeys)
    logger.info(unexpkeys)
    logger.info("==> loaded checkpoint '{}' (epoch {})".format(args.pretrained_model, ckpt['epoch']))


def get_loader(args, trainval):

    select_transform = [
        clip_transforms.ClipRandomResizedCrop(224, scale=(0.2, 1.), ratio=(0.75, 1.3333333333333333)),
        clip_transforms.ClipResize((224, 224)),
        transforms.RandomApply([clip_transforms.ClipColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        clip_transforms.ClipRandomGrayscale(p=0.2),
        transforms.RandomApply([clip_transforms.ClipGaussianBlur([.1, 2.])], p=0.5),
        clip_transforms.ClipRandomHorizontalFlip(),
    ]
    basic_transoform = [
        clip_transforms.ToClipTensor(),
        clip_transforms.ClipNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        clip_transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if opt.time_dim == "T" else transforms.Lambda(
            lambda clip: torch.cat(clip, dim=0))
    ]

    T = []
    for id in args.trans.split('-'):
        T.append(select_transform[int(id)])
    T.extend(basic_transoform)
    data_transforms = clip_transforms.Compose(T)

    dataset = VideoRGBTrainDataset(
                                    list_root=args.list_path,
                                    root_path=args.root_path,
                                    transform=data_transforms,
                                    clip_length=args.clip_length,
                                    num_steps=args.num_steps,
                                    num_segments=args.num_segments,
                                    dataset=args.dataset,
                                    split=trainval,
                                    data_form='lmdb'
                                    )

    dataset_test = VideoRGBTestDataset(
                                        list_root=args.list_path_test,
                                        root_path=args.root_path,
                                        transform=data_transforms,
                                        clip_length=args.clip_length,
                                        num_steps=args.num_steps,
                                        num_segments=args.num_segments_test,
                                        dataset=args.dataset,
                                        split='test',
                                        data_form='lmdb'
                                        )


    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler,
        num_workers=args.num_workers, pin_memory=False,
        drop_last=False)
    
    loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size_test, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
        drop_last=False)

    return loader, loader_test



def save_checkpoint(args, epoch, model, scheduler, optimizer):
    logger.info('==> Saving...')
    state = {
        'opt': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    save_dir = os.path.join(args.output_dir,'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, 'current.pth'))
    if epoch % args.save_freq == 0:
        torch.save(state, os.path.join(save_dir, 'ckpt_epoch_{}.pth'.format(epoch)))

def main(args, logger):

    data_loader, data_loader_test = get_loader(opt, opt.trainval)

    model = Encoder(classfier=True, num_classes=args.num_classes)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['classfier.fc.weight', 'classfier.fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.classfier.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.classfier.fc.bias.data.zero_()

    load_pretrained(opt, model)

    if args.trainval == 'test':
        torch.cuda.set_device(args.gpu_id)
        seed_torch(args.seed)

        model = model.cuda()

        if dist.get_rank() == 0:

            top1, top5 = validate(args, model, data_loader_test)
            logger.info('epoch {}, top1 acc:{} top5 acc: {}'.format(100, top1, top5))

    else:
        model = model.cuda()

        model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

        # for loss
        criterion = nn.CrossEntropyLoss().cuda()

        # optimize only the linear classifier
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias
        optimizer = torch.optim.SGD(parameters, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        # for sch
        scheduler = get_scheduler(optimizer, len(data_loader), args)

        if args.resume:
            ckpt = torch.load(args.pretrained_model, map_location='cpu')
            scheduler.load_state_dict(ckpt['scheduler'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epochs = ckpt['epoch'] + 1
        else:
            start_epochs = 1

        for epoch in range(start_epochs, args.epochs + 1):
            start = time.time()
            loss =  train(model, data_loader, criterion, optimizer, epoch, scheduler, logger, args)

            logger.info('epoch {}, total time {:.2f}'.format(epoch, time.time() - start))
            logger.info('epoch {}, average loss is {:>0.3f}'.format(epoch, loss))

            if dist.get_rank() == 0:
                # save model
                save_checkpoint(args, epoch, model, scheduler, optimizer)



def train(model, data_loader, criterion, optimizer, epoch, scheduler, logger, args):

    model.eval()
    loss_meter = AverageMeter()

    for idx, (data, label, vname) in enumerate(data_loader):
        bsz = data.size(0)
        data = data.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # compute output
        output = model(data, mode='classfier')
        loss = criterion(output, label)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


        if idx % args.print_freq == 0:
            logger.info('Train: [{:>3d}]/[{:>4d}/{:>4d}] Loss={:>0.3f} / {:>0.3f}'.format(
                    epoch, idx, len(data_loader),
                    loss.item(), loss_meter.avg)
                    )

        # update meters
        loss_meter.update(loss.item(), bsz)
    
    return loss_meter.avg


def validate(args, model, data_loader):

    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    model.eval()

    with torch.no_grad():
        for idx, (data, label, vname) in enumerate(tqdm(data_loader, total=len(data_loader))):

            data_size = data.size()
            data = data.view((-1, 3, data_size[-3], data_size[-2], data_size[-1]))
            data = data.cuda()


            prediction = model(data, mode='classfier')
            prediction = prediction.view(args.batch_size_test, args.num_segments_test, args.num_classes).mean(1).cpu()


            top1, top5 = calc_topk_accuracy(prediction, label, (1, 5))
            top1_meter.update(top1, args.batch_size_test)
            top5_meter.update(top5, args.batch_size_test)

    return top1_meter.avg, top5_meter.avg

if __name__ == '__main__':
    opt = parse_option()

    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    os.makedirs(opt.output_dir, exist_ok=True)
    logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name="linear-probe")
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "config.json")
        with open(path, "w") as f:
            json.dump(vars(opt), f, indent=2)

        if opt.trainval == 'train':
            logger.info("Start to train a fc")
        else:
            logger.info("Start to test the fc")

        logger.info("Full config saved to {}".format(path))

    main(opt, logger)

