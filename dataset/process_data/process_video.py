import numpy as np
import cv2
from PIL import Image
import io

def get_video_frames_cv(v_path, dataset='ucf101'):

    target = 256 # for kinetics
    vidcap = cv2.VideoCapture(v_path)
    nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    if nb_frames == 0: return None
    w = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    h = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float


    short_size = min(w, h)
    success, image = vidcap.read()
    count = 1

    if w >= h:
        size = (int(target * w / h), int(target))
    else:
        size = (int(target), int(target * h / w))

    frames = []
    while success:
        if dataset == 'kinetics':
            if short_size <= 256:
                image = cv2.resize(image, size, cv2.INTER_CUBIC)
            else:
                image = cv2.resize(image, size, cv2.INTER_AREA)

        frames.append(image)

        success, image = vidcap.read()
        count += 1

    vidcap.release()
    return frames

def compute_TVL1(prev, curr, bound=20):
    """Compute the TV-L1 optical flow."""

    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()

    flow = TVL1.calc(prev, curr, None)
    flow = np.clip(flow, -bound, bound)

    flow = (flow + bound) * (255.0 / (2*bound))
    flow = np.round(flow).astype('uint8')

    return flow