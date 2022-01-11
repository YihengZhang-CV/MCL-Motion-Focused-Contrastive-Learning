import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import time
import os
import json
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms

from dataset.video_dataset import VideoRGBTrainDataset_Motion
from dataset.augmentations import clip_transforms
from contrast.NCEContrast import NCESoftmaxLoss, MemoryMCL
from utils.logger import setup_logger
from utils.util import *
from utils.lr_scheduler import get_scheduler
from model.encoder import Encoder
from model.mal import MAL


def parse_option():
    parser = argparse.ArgumentParser('training')

    # dataset
    # for video_dataset
    parser.add_argument('--list-path', type=str, default='data/ucf101/train_list_01.csv', help='path of list file')
    parser.add_argument('--root-path', type=str, default='/lr/export2/home/zhangyiheng8/dataset/UCF-101/UCF-101-lmdb240', help='path of rgb root folder')
    parser.add_argument('--root-path-flow', type=str, default='/dev/shm/UCF-101-flow-lmdb240', help='path of flow root folder')
    parser.add_argument('--root-path-mag', type=str, default='/dev/shm/UCF-101-flow-mag-lmdb7', help='path of flow mag root folder')
    parser.add_argument('--data-form', type=str, default='lmdb', choices=['jpg','lmdb','video'], help='data type of input')
    parser.add_argument('--dataset', type=str, default='ucf101', choices=['kinetics','ucf101'], help='dataset to training')
    parser.add_argument('--crop', type=float, default=0.2, help='minimum crop')
    parser.add_argument('--batch-size', type=int, default=8, help='batch_size')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--clip-length', type=int, default=16, help='num of clip length')
    parser.add_argument('--num-steps', type=int, default=2, help='num of sampling steps')
    parser.add_argument('--num-segments', type=int, default=3, help='num of segments')
    parser.add_argument('--input-size', type=int, default=224, help='size pf input rgb')
    parser.add_argument('--mag-size', type=int, default=7, help='size pf input mag')

    # model and loss function
    parser.add_argument('--alpha', type=float, default=0.999, help='exponential moving average weight')
    parser.add_argument('--nce-k', type=int, default=131072, help='num negative sampler')
    parser.add_argument('--nce-t', type=float, default=0.10, help='NCE temperature')
    parser.add_argument('--nce-t-intra', type=float, default=0.10, help='NCE temperature')

    # optimization
    parser.add_argument('--base-learning-rate', '--base-lr', type=float, default=0.2)
    parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr-decay-epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--time-dim', type=str, default='T',
                        choices=["T", "C"], help="dimension for time")
    parser.add_argument('--resume', type=str2bool, default='false', help='warmup epoch')

    # io
    parser.add_argument('--pretrained-model', default='/lr/export/home/lirui295/models/moco_v2_200ep_pretrain.pth.tar', type=str, metavar='PATH',
                        help='path to pretrained weights like moco imagenet (default: none)')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save-freq', type=int, default=100, help='save frequency')
    parser.add_argument('--output-dir', type=str, default='output', help='output director')

    # misc
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel', default=0)

    # model
    parser.add_argument("--inflate-weights", type=str2bool, default='true')
    parser.add_argument("--target-module", type=str, default='module.backbone.layer4.2.conv3', help="the target module name")


    args = parser.parse_args()
    return args


def get_loader(args, mode='train'):

    transoform_list = [
        transforms.RandomApply([clip_transforms.ClipColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        clip_transforms.ClipRandomGrayscale(p=0.2),
        transforms.RandomApply([clip_transforms.ClipGaussianBlur([.1, 2.])], p=0.5),
        clip_transforms.ClipRandomHorizontalFlip(),
        clip_transforms.ToClipTensor(),
        clip_transforms.ClipNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        clip_transforms.Lambda(lambda clip: torch.stack(clip, dim=1)) if opt.time_dim == "T" else transforms.Lambda(
            lambda clip: torch.cat(clip, dim=0))
    ]

    if mode == 'train':
        train_transform = clip_transforms.Compose(transoform_list)
        motion_focus_spatial_crop = clip_transforms.ClipRandomResizedCropMotion(opt.input_size, scale=(args.crop, 1.), ratio=(0.75, 1.3333333333333333))

        train_dataset = VideoRGBTrainDataset_Motion(
                                                    root_path_flow=args.root_path_flow,
                                                    root_path_mag=args.root_path_mag,
                                                    input_size=args.input_size, mag_size=args.mag_size,
                                                    list_root=args.list_path, root_path=args.root_path,
                                                    transform=train_transform, motion_focus_spatial_crop=motion_focus_spatial_crop,
                                                    clip_length=args.clip_length, num_steps=args.num_steps,
                                                    dataset=args.dataset,
                                                    data_form=args.data_form
                                                    )


        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,sampler=train_sampler,
            drop_last=True)
       
        return train_loader
    else:
        pass

def build_model(args):
    model = Encoder(num_channels=128, mlp_layers=2, order=True).cuda()
    model_ema = Encoder(num_channels=128, mlp_layers=2, order=True).cuda()
    mal = MAL().cuda()

    if args.pretrained_model:
        load_pretrained(args, model)

    # copy weights from `model' to `model_ema'
    moment_update(model, model_ema, 0)

    return model, model_ema, mal

def load_pretrained(args, model):
    ckpt = torch.load(args.pretrained_model, map_location='cpu')
    state_dict = {k.replace("module.encoder_q.", "backbone."): v for k, v in ckpt['state_dict'].items()}

    # convert initial weights
    if args.inflate_weights:
        state_dict = convert_pretrained_weights(state_dict, args)

    [misskeys, unexpkeys] = model.load_state_dict(state_dict, strict=False)
    logger.info(misskeys)
    logger.info(unexpkeys)
    logger.info("==> loaded checkpoint '{}' (epoch {})".format(args.pretrained_model, ckpt['epoch']))


def save_checkpoint(args, epoch, model, model_ema, contrast, scheduler, optimizer):
    logger.info('==> Saving...')
    state = {
        'opt': args,
        'model': model.state_dict(),
        'model_ema': model_ema.state_dict(),
        'contrast': contrast.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    save_dir = os.path.join(args.output_dir,'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, 'current.pth'))
    if epoch % args.save_freq == 0:
        torch.save(state, os.path.join(save_dir, 'ckpt_epoch_{}.pth'.format(epoch)))


def main(args):
    train_loader = get_loader(args)
    n_data = len(train_loader.dataset)
    logger.info("length of training dataset: {}".format(n_data))

    model, model_ema, mal = build_model(args)

    contrast = MemoryMCL(128, args.nce_k, args.nce_t, args.nce_t_intra).cuda()
    criterion = NCESoftmaxLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.base_learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = get_scheduler(optimizer, len(train_loader), args)

    start_epochs = 1

    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)
    model_ema = DistributedDataParallel(model_ema, device_ids=[args.local_rank], broadcast_buffers=False)

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=os.path.join(args.output_dir,'logs'))
    else:
        summary_writer = None
   

    # routine
    for epoch in range(start_epochs, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        tic = time.time()
        loss, prob = train_one_epoch(epoch, train_loader, model, mal, model_ema, contrast, criterion, optimizer, scheduler, args, summary_writer)
        logger.info('epoch {}, total time {:.2f}'.format(epoch, time.time() - tic))
        logger.info('epoch {}, average loss is {:>0.3f}'.format(epoch, loss))
        if summary_writer is not None:
            # tensorboard logger
            summary_writer.add_scalar('Epoch/ins_loss', loss, epoch)
            summary_writer.add_scalar('Epoch/ins_prob', prob, epoch)
            summary_writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if dist.get_rank() == 0:
            # save model
            save_checkpoint(args, epoch, model, model_ema, contrast, scheduler, optimizer)

def train_one_epoch(epoch, train_loader, model, mal, model_ema, contrast, criterion, optimizer, scheduler, args, summary_writer):

    model.train()
    set_bn_train(model_ema)

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    prob_inter_meter = AverageMeter()
    prob_intra_meter = AverageMeter()
    end = time.time()

    for name, module in model.named_modules() :
        if name == args.target_module:
            target_module = module

    for idx, (xq, st_mag, x1, x2, x3, order_target) in enumerate(train_loader):
        bsz = xq.size(0)
        # forward
        xq = xq.cuda(non_blocking=True).requires_grad_()  # quary
        x1 = x1.cuda(non_blocking=True)  # same clip diff aug
        x2 = x2.cuda(non_blocking=True)  # diff clip 1
        x3 = x3.cuda(non_blocking=True)  # diff clip 2
        order_target = order_target.cuda(non_blocking=True)  # order: binary labels
        st_mag = st_mag.cuda(non_blocking=True) # st-motion

        total_feat_out = []
        hook = forward_hook(model, total_feat_out, target_module)

        # forward and get the 128d encoded representations of query
        feat_inter_q, feat_intra_q, feat_order_q = model(xq)

        xconv = total_feat_out[-1]
        hook.remove()
        
        # forward and get the 128d encoded representations of keys
        Shuffle = DistributedShuffle

        # shuffled bn is applied
        with torch.no_grad():
            # 1st key
            x1_shuffled, backward_inds = Shuffle.forward_shuffle(x1, epoch)
            feat_inter_x1, feat_intra_x1, _ = model_ema(x1_shuffled)
            # get x_all for update memory
            feat_inter_x1_all, feat_inter_x1 = Shuffle.backward_shuffle(feat_inter_x1, backward_inds, return_local=True)
            feat_intra_x1_all, feat_intra_x1 = Shuffle.backward_shuffle(feat_intra_x1, backward_inds, return_local=True)
            # 2nd key
            x2_shuffled, backward_inds = Shuffle.forward_shuffle(x2, epoch)
            feat_inter_x2, feat_intra_x2, feat_order_x2 = model_ema(x2_shuffled)
            feat_inter_x2_all, feat_inter_x2 = Shuffle.backward_shuffle(feat_inter_x2, backward_inds, return_local=True)
            feat_intra_x2_all, feat_intra_x2 = Shuffle.backward_shuffle(feat_intra_x2, backward_inds, return_local=True)
            if feat_order_x2 is not None:
                _, feat_order_x2 = Shuffle.backward_shuffle(feat_order_x2, backward_inds, return_local=True)
            # 3nd key
            x3_shuffled, backward_inds = Shuffle.forward_shuffle(x3, epoch)
            feat_inter_x3, feat_intra_x3, feat_order_x3 = model_ema(x3_shuffled)
            feat_inter_x3_all, feat_inter_x3 = Shuffle.backward_shuffle(feat_inter_x3, backward_inds, return_local=True)
            feat_intra_x3_all, feat_intra_x3 = Shuffle.backward_shuffle(feat_intra_x3, backward_inds, return_local=True)
            if feat_order_x3 is not None:
                _, feat_order_x3 = Shuffle.backward_shuffle(feat_order_x3, backward_inds, return_local=True)

        # calc nce loss
        out_inter, l_pos = contrast(feat_inter_q, feat_inter_x1, feat_inter_x2, feat_inter_x3, feat_inter_x1_all, feat_inter_x2_all, feat_inter_x3_all, inter=True)
        out_intra = contrast(feat_intra_q, feat_intra_x1, feat_intra_x2, feat_intra_x3, inter=False)
        loss_nce = criterion(out_inter) + criterion(out_intra)

        # calc gradient with regard to target module
        gradient = torch.autograd.grad(outputs=l_pos, inputs=xconv,
                                                  grad_outputs=torch.ones(l_pos.size()).cuda(), retain_graph=True,
                                                  create_graph=True)[0]

        # calc mal loss
        loss_mal = mal(gradient, xconv, st_mag)

        # calc order loss
        feat_order = torch.cat((feat_order_q, feat_order_x2.detach(), feat_order_x3.detach()), dim=-1)
        loss_order = F.cross_entropy(model(feat_order, mode='classfier_order'), order_target)
        loss_order_w = float(epoch) / float(args.epochs)
        if args.dataset == 'kinetics': loss_order_w  = 0
        loss = loss_nce + loss_order_w * loss_order + loss_mal if epoch > args.warmup_epoch else loss_nce + loss_order_w * loss_order + 2 * loss_mal

        prob_inter = F.softmax(out_inter, dim=1)[:, 0].mean()
        prob_intra = F.softmax(out_intra, dim=1)[:, 0].mean()

        # backward
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        moment_update(model, model_ema, args.alpha)

        # update meters
        loss_meter.update(loss.item(), bsz)
        prob_inter_meter.update(prob_inter.item(), bsz)
        prob_intra_meter.update(prob_intra.item(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % args.print_freq == 0:
            logger.info('Train: [{:>3d}]/[{:>4d}/{:>4d}] BT={:>0.3f}/{:>0.3f} Loss={:>0.3f} {:>0.3f} {:>0.3f} / {:>0.3f} Pinter={:>0.3f}/{:>0.3f} Pintra={:>0.3f}/{:>0.3f}'.format(
                epoch, idx, len(train_loader),
                batch_time.val, batch_time.avg,
                loss.item(), loss_nce.item(), loss_mal.item(), loss_meter.avg,
                prob_inter_meter.val, prob_inter_meter.avg,
                prob_intra_meter.val, prob_intra_meter.avg,
            ))

    return loss_meter.avg, prob_inter_meter.avg

if __name__ == '__main__':

    opt = parse_option()
    file_path = os.path.dirname(os.path.abspath(__file__))

    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True
    logs_dir = os.path.join(opt.output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    distributed_rank = dist.get_rank() 
    logger = setup_logger(output=logs_dir, distributed_rank=distributed_rank, name="MCL")

    if dist.get_rank() == 0:
        path = os.path.join(logs_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    main(opt)