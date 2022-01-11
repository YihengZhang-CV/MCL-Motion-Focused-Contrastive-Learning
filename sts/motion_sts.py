import cv2
import numpy as np
from scipy import ndimage

def compute_motion_boudary(flow_clip):

    mx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    my = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    dx_all = []
    dy_all = []
    mb_x = 0
    mb_y = 0

    for flow_img in flow_clip:
        d_x = ndimage.convolve(flow_img, mx)
        d_y = ndimage.convolve(flow_img, my)

        dx_all.append(d_x)
        dy_all.append(d_y)

        mb_x += d_x
        mb_y += d_y

    dx_all = np.array(dx_all)
    dy_all = np.array(dy_all)

    return dx_all, dy_all, mb_x, mb_y

def zero_boundary(frame_mag):

    frame_mag[:8, :] = 0
    frame_mag[:, :8] = 0
    frame_mag[-8:, :] = 0
    frame_mag[:, -8:] = 0
    
    return frame_mag

def motion_mag_downsample(mag, size, input_size):
    block_size = input_size // size
    mask = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            x_start = i * block_size
            x_end = x_start + block_size
            y_start = j * block_size
            y_end = y_start + block_size

            tmp_block = mag[x_start:x_end, y_start:y_end]

            block_mean = np.mean(tmp_block)
            mask[i, j] = block_mean
    return mask

def motion_sts(flow_clip, size, input_size):

    dx_all, dy_all, dx_sum, dy_sum = compute_motion_boudary(flow_clip)
    mag, ang = cv2.cartToPolar(dx_sum, dy_sum, angleInDegrees=True)
    mag_down = motion_mag_downsample(mag, size, input_size)

    return mag_down

