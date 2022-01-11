from PIL import Image
import numpy as np
import torch
import cv2
import torchvision.transforms.functional as F
import random
import io
import math

def pil_from_raw_rgb(raw):
    return Image.open(io.BytesIO(raw)).convert('RGB')

def pil_from_raw_rgba(raw):
    return Image.open(io.BytesIO(raw)).convert('L')

def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


def calc_over_lab(z1, z2):
    z_min = max(z1[0], z2[0])
    z_max = min(z1[1], z2[1])
    if z_min >= z_max:
        return 0
    else:
        return (z_max - z_min) / (z1[1] - z1[0])

def get_cor(clip_mag, height, width, base_ratio=0.8, t=0.1):
    if isinstance(clip_mag, list):
        clip_mag = sum(clip_mag) / len(clip_mag)
    if len(clip_mag.shape) == 3:
        clip_mag = clip_mag[:,:,0]

    h, w = clip_mag.shape
    flat_mag = clip_mag.reshape(-1)
    idx = np.argsort(flat_mag)[::-1][:math.floor(h*w*t)]
    row = (idx // w) + 1
    col = (idx % w) + 1

    row_s = [row.min() - 1, row.max()]
    col_s = [col.min() - 1, col.max()]

    # c represent the area of the region with high motion
    c = (row_s[1] - row_s[0]) * (col_s[1] - col_s[0])

    # for different c, we would make minor change on base_ratio
    if   c <=  4: crop_ratio = base_ratio + 0.1
    elif c <=  8: crop_ratio = base_ratio 
    elif c <= 12: crop_ratio = base_ratio - 0.1
    else:         crop_ratio = 0

    row_s = list(map(lambda x: x * height / 7, row_s))
    col_s = list(map(lambda x: x * width / 7, col_s))

    return row_s, col_s, crop_ratio


def crop_from_corners(row_s,col_s, height, width):

    if random.randint(0, 1) == 0:
        i_h = random.randint(0, int(row_s[0]) + 1)
        i_w = random.randint(0, int(col_s[0]) + 1)
        w = col_s[1] - i_w
        h = row_s[1] - i_h
        return i_h, i_w, h, w
    else:
        i_h = random.randint(int(row_s[1] - 1), height)
        i_w = random.randint(int(col_s[1] - 1), width)
        w = i_w - col_s[0]
        h = i_h - row_s[0]
        return row_s[0], col_s[0], h, w