import torch.utils.data as data
#from imageio import imread
from pathlib import Path
import random

from PIL import Image
from .mono_seq_dataset import MonoSeqDataset


import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset

def get_K():
    intrinsics = np.eye(3, dtype=np.float32)
    homo_k = np.eye(4, dtype=np.float32)

    fx = 3.379191448899105e+02
    fy = 3.386957068549526e+02
    cx = 3.417366010946575e+02
    cy = 2.007359735313929e+02

    in_h, in_w = 376, 672
    intrinsics[0, 0] = fx/in_w
    intrinsics[0, 2] = cx/in_w
    intrinsics[1, 1] = fy/in_h
    intrinsics[1, 2] = cy/in_h
    
    homo_k[:3, :3] = intrinsics

    return homo_k

class RadiateMonoDataset(MonoSeqDataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, *args, **kwargs):
        super(RadiateMonoDataset, self).__init__(*args, **kwargs)
        self.K = get_K()
        self.full_res_shape = (672, 372)

    def get_color(self, sample, key, shift, do_flip):
        color = self.loader(sample[key + "_{}".format(shift)])
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
    
    def get_depth(self, sample, key, do_flip):
        depth_path = sample[key]
        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
