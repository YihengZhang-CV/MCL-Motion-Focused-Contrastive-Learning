import argparse
import random
from PIL import ImageFilter,Image
import torch
import torch.distributed as dist
import numpy as np
import cv2
import io
import torch.nn as nn
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0)


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


class DistributedShuffle:
    @staticmethod
    def forward_shuffle(x, epoch):
        """ forward shuffle, return shuffled batch of x from all processes.
        epoch is used as manual seed to make sure the shuffle id in all process is same.
        """
        x_all = dist_collect(x)
        forward_inds, backward_inds = DistributedShuffle.get_shuffle_ids(x_all.shape[0], epoch)

        forward_inds_local = DistributedShuffle.get_local_id(forward_inds)
        return x_all[forward_inds_local], backward_inds

    @staticmethod
    def backward_shuffle(x, backward_inds, return_local=True):
        """ backward shuffle, return data which have been shuffled back
        x is the shared data, should be local data
        if return_local, only return the local batch data of x.
            otherwise, return collected all data on all process.
        """
        x_all = dist_collect(x)
        if return_local:
            backward_inds_local = DistributedShuffle.get_local_id(backward_inds)
            return x_all[backward_inds], x_all[backward_inds_local]
        else:
            return x_all[backward_inds]

    @staticmethod
    def get_local_id(ids):
        return ids.chunk(dist.get_world_size())[dist.get_rank()]

    @staticmethod
    def get_shuffle_ids(bsz, epoch):
        """generate shuffle ids for ShuffleBN"""
        torch.manual_seed(epoch) # only update shuffle idx each epoch
        # global forward shuffle id  for all process
        forward_inds = torch.randperm(bsz).long().cuda()

        # global backward shuffle id
        backward_inds = torch.zeros(forward_inds.shape[0]).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)

        return forward_inds, backward_inds


def set_bn_train(model):
    def set_bn_train_helper(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()

    model.eval()
    model.apply(set_bn_train_helper)


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


class SingeShuffle(DistributedShuffle):

    def forward_shuffle(x, epoch):
        forward_inds, backward_inds = DistributedShuffle.get_shuffle_ids(x.shape[0], epoch)

        #forward_inds_local = DistributedShuffle.get_local_id(forward_inds)

        return x[forward_inds], backward_inds


    def backward_shuffle(x, backward_inds, return_local=True):
        """ backward shuffle, return data which have been shuffled back
        x is the shared data, should be local data
        if return_local, only return the local batch data of x.
            otherwise, return collected all data on all process.
        """
        x_all = x

        return x_all[backward_inds], x_all[backward_inds]

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def forward_hook(model, total_feat_out, module):

    def hook_fn_forward(module, input, output):
        total_feat_out.append(output) 

    hook = module.register_forward_hook(hook_fn_forward)
    return hook

def convert_pretrained_weights(state_dict, args):
    new_state_dict = {}
    for k, v in state_dict.items():
        v = v.detach().numpy()
        if ('conv'in k) or ('downsample.0' in k):
            shape = v.shape
            v = np.reshape(v, newshape=[shape[0], shape[1], 1, shape[2], shape[3]])
            if (shape[2] == 3) and (shape[3] == 3): # basic conv3x3 layer
                kernel = np.zeros(shape=(shape[0], shape[0], 3, 1, 1))
                for n in range(0, shape[0]):
                    kernel[n, n, 0, 0, 0] = 0.0
                    kernel[n, n, 1, 0, 0] = 1.0
                    kernel[n, n, 2, 0, 0] = 0.0
                ss = k.split('.')
                new_state_dict[k[:-len(ss[-1])-1] + '_t.' + ss[-1]] = torch.from_numpy(kernel)
            else:
                if (shape[2] == 7) and (shape[3] == 7): # first conv7x7 layer
                    v = np.concatenate((np.zeros(shape=(shape[0], shape[1], 1, shape[2], shape[3])), v, np.zeros(shape=(shape[0], shape[1], 2, shape[2], shape[3]))), axis=2)
        if args.dataset == 'ucf101':
            if not 'fc' in k: # remove final fc layer
                new_state_dict[k] = torch.from_numpy(v)
        else:
            if 'fc' in k:
                strs = k.split('.')
                new_key = 'nce_inter_head.head.' + '.'.join(strs[-2:])
                new_state_dict[new_key] = torch.from_numpy(v)
            else:
                new_state_dict[k] = torch.from_numpy(v)

    return new_state_dict

def calc_topk_accuracy(output, target, topk=(1,)):
    """
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels,
    calculate top-k accuracies.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
