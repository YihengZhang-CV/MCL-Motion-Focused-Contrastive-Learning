import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import Tensor
import cv2
import os

try:
    import accimage
except ImportError:
    accimage = None

import torchvision.transforms.functional as F
from torchvision import transforms
from dataset.augmentations.augmentation_util import _get_image_size, calc_over_lab, get_cor, crop_from_corners

__all__ = ["ToClipTensor", "ClipRandomResizedCrop", "ClipRandomResizedCropMotion", "ClipColorJitter", "ClipRandomGrayscale", "ClipRandomHorizontalFlip", "ClipResize", "ClipCenterCrop", "ClipNormalize", "ClipGaussianBlur"
]

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

class Compose(transforms.Compose):

    def __call__(self, input):
        for t in self.transforms:
            if isinstance(input, tuple):
                input = t(*input)
            else:
                input = t(input)
        return input


class ToClipTensor(object):
    """Convert a List of ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, clip, is_flip=False):
        """
        Args:
            clip (List of PIL Image or numpy.ndarray): Clip to be converted to tensor.

        Returns:
            Tensor: Converted clip.
        """

        return ([F.to_tensor(img) for img in clip], is_flip)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ClipRandomResizedCrop(transforms.RandomResizedCrop):
    def __call__(self, clip):
        """
        Args:
            clip (List of PIL Image or Tensor): Clip to be cropped and resized.

        Returns:
            List of PIL Image or Tensor: Randomly cropped and resized clip.
        """
        i, j, h, w = self.get_params(clip[0], self.scale, self.ratio)
        return [F.resized_crop(img, i, j, h, w, self.size, self.interpolation) for img in clip]


class ClipRandomResizedCropMotion(object):
    """Crop the given PIL Image to random size and aspect ratio base on motion.
    """

    def __init__(self, size,  scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), iou = 0.9, interpolation=Image.BILINEAR):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.iou = iou

    @staticmethod
    def get_params(img, scale, ratio,  mags):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        mags = list(map(lambda x: np.array(x).astype(np.float32), mags))
        row_s, col_s, crop_ratio = get_cor(mags, height, width)

        for _ in range(20):
            sc = random.uniform(*scale)
            target_area = sc * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                if mags == [] or crop_ratio == 0:
                    i = random.randint(0, height - h)
                    j = random.randint(0, width - w)
                    return i, j, h, w
                else:
                    crop_list = []
                    ratio_out_list = []
                    for i in range(40):
                        i = random.randint(0, height - h)
                        j = random.randint(0, width - w)

                        row_ = (i, i + h)
                        col_ = (j, j + w)

                        ratio_row = calc_over_lab(row_s, row_)
                        ratio_col = calc_over_lab(col_s, col_)
                        ratio_out = ratio_col * ratio_row

                        crop_list.append((i, j, h, w))
                        ratio_out_list.append(ratio_out)

                        if ratio_out >= crop_ratio:
                            return i, j, h, w
                        else:
                            continue

                    # max ratio_out
                    ratio_outs = np.array(ratio_out_list)
                    idx = np.argsort(ratio_outs)[-1]
                    i, j, h, w = crop_list[idx]

                    return i, j, h, w

        # crop from corners
        if crop_ratio != 0:
            return crop_from_corners(row_s, col_s, height, width)

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, clip, mags, flows=None):
        """
        Args:
            clip (List of PIL Image or Tensor): Input clip.
            flows (List of PIL Image or Tensor): Input clip's optical flow.
            mags (List of PIL Image or Tensor): Input clip's motion magnitude.
            return_flow : if crop and resize flows.
        Returns:
            List of PIL Image or Tensor: croped and resized clip and flows.
        """
        i, j, h, w = self.get_params(clip[0], self.scale, self.ratio, mags)
        if flows == None:
            return [F.resized_crop(img, i, j, h, w, self.size, self.interpolation) for img in clip], \
                   None, \
                    None
        else:
            return [F.resized_crop(img, i, j, h, w, self.size, self.interpolation) for img in clip], \
                [F.resized_crop(flow, i, j, h, w, self.size, self.interpolation) for flow in flows[0]], \
                [F.resized_crop(flow, i, j, h, w, self.size, self.interpolation) for flow in flows[1]]


    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string



class ClipColorJitter(transforms.ColorJitter):
     def __call__(self, clip):
        """
        Args:
            clip (List of PIL Image or Tensor): Input clip.

        Returns:
            List of PIL Image or Tensor: Color jittered clip.
        """
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                clip = [F.adjust_brightness(img, brightness_factor) for img in clip]

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                clip = [F.adjust_contrast(img, contrast_factor) for img in clip]

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                clip = [F.adjust_saturation(img, saturation_factor) for img in clip]

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                clip = [F.adjust_hue(img, hue_factor) for img in clip]

        return clip


class ClipRandomGrayscale(transforms.RandomGrayscale):
    def __call__(self, clip):
        """
        Args:
            clip (List of PIL Image or Tensor): Clip to be converted to grayscale.

        Returns:
            List of PIL Image or Tensor: Randomly grayscaled clip.
        """
        num_output_channels = 1 if clip[0].mode == 'L' else 3
        if random.random() < self.p:
            return [F.to_grayscale(img, num_output_channels=num_output_channels) for img in clip]
        return clip


class ClipRandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __call__(self, clip, is_flip=True):
        """
        Args:
            clip (List of PIL Image or Tensor): Clip to be flipped.

        Returns:
            List of PIL Image or Tensor: Randomly flipped clip.
        """

        if torch.rand(1) < self.p and is_flip:      
            clip =  [F.hflip(img) for img in clip]
            is_flip = True
        else:
            is_flip = False
        
        return (clip, is_flip)


class ClipNormalize(object):
    """Normalize a list of tensor images with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip, is_flip=False):
        """
        Args:
            clip (List of Tensor): List of tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor list.
        """

        return ([F.normalize(img, self.mean, self.std, self.inplace) for img in clip], is_flip)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ClipResize(transforms.Resize):
    """Resize the list of PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __call__(self, clip):
        """
        Args:
            clip (List of Tensor):: Clip to be scaled.

        Returns:
            List of PIL Image: Rescaled clip.
        """
        return [F.resize(img, self.size, self.interpolation) for img in clip]


class ClipCenterCrop(transforms.CenterCrop):
    """Crops the given list of PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, clip):
        """
        Args:
            clip (List of Tensor): Clip to be cropped.

        Returns:
            List of PIL Image: Cropped clip.
        """
        return [F.center_crop(img, self.size) for img in clip]


class ClipFirstCrop(transforms.CenterCrop):
    """Crops the given list of PIL Image at the 1/3.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, clip):
        """
        Args:
            clip (List of Tensor): Clip to be cropped.

        Returns:
            List of PIL Image: Cropped clip.
        """
        if isinstance(self.size, numbers.Number):
            self.size = (int(self.size), int(self.size))
        else:
            assert len(self.size) == 2, "Please provide only two dimensions (h, w) for size."
        image_width, image_height = clip[0].size
        crop_height, crop_width = self.size
        if crop_width > image_width or crop_height > image_height:
            msg = "Requested crop size {} is bigger than input size {}"
            raise ValueError(msg.format(self.size, (image_height, image_width)))

        return [img.crop((0, 0, crop_width, crop_height)) for img in clip]
        
        
class ClipThirdCrop(transforms.CenterCrop):
    """Crops the given list of PIL Image at the 3/3.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, clip):
        """
        Args:
            clip (List of Tensor): Clip to be cropped.

        Returns:
            List of PIL Image: Cropped clip.
        """
        if isinstance(self.size, numbers.Number):
            self.size = (int(self.size), int(self.size))
        else:
            assert len(self.size) == 2, "Please provide only two dimensions (h, w) for size."
        image_width, image_height = clip[0].size
        crop_height, crop_width = self.size
        if crop_width > image_width or crop_height > image_height:
            msg = "Requested crop size {} is bigger than input size {}"
            raise ValueError(msg.format(self.size, (image_height, image_width)))

        return [img.crop((image_width - crop_width, image_height - crop_height, image_width, image_height)) for img in clip]

class ClipGaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, clip):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return [img.filter(ImageFilter.GaussianBlur(radius=sigma)) for img in clip]

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, clip, is_flip=False):

        return (self.lambd(clip), is_flip)

    def __repr__(self):
        return self.__class__.__name__ + '()'


if __name__ == '__main__':
    pass