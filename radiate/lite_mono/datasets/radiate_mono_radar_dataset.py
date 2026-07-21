import torch.utils.data as data
#from imageio import imread
from pathlib import Path
import random

from PIL import Image
from .mono_radar_seq_dataset import MonoRadarSeqDataset 


import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import scipy.ndimage


from typing import List, Tuple


import torch
from torchvision import transforms
import torchvision.transforms.functional as F


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

class RadiateMonoRdarDataset(MonoRadarSeqDataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """
    
    # TODO overwrite the getitem function to get the radar input

    def __init__(self, *args, **kwargs):
        super(RadiateMonoRdarDataset, self).__init__(*args, **kwargs)
        self.K = get_K()
        self.full_res_shape = (672, 372)

    def get_color(self, sample, key, shift, do_flip):
        color = self.loader(sample[key + "_{}".format(shift)])
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color, Path(sample[key + "_{}".format(shift)]).stem
    
    def get_radar_bev(self, sample, key, shift, do_flip):
        radar_bev = self.load_cart_as_float(sample[key + "_{}".format(shift)])
        if do_flip:
            radar_bev = radar_bev.transpose(pil.FLIP_LEFT_RIGHT)
        return radar_bev
    
    def get_radar_pov(self, sample, key, shift, do_flip):
        radar_pov = Image.open(sample[key + "_{}".format(shift)])
        radar_pov = radar_pov.resize((640, 320), Image.Resampling.NEAREST)
        if do_flip:
            radar_pov = radar_pov.transpose(pil.FLIP_LEFT_RIGHT)
        return radar_pov
    
    def get_unc(self, sample, key, do_flip):
        # color = self.loader(sample[key + "_{}".format(shift)])
        unc_path = sample[key]
        unc_map = pil.open(unc_path)  
        # unc_map = np.array(pil.open(unc_path)).astype(np.float32)   
        if do_flip:
            unc_map = unc_map.transpose(pil.FLIP_LEFT_RIGHT)
        return unc_map
    
    def get_depth(self, sample, key, do_flip):
        depth_path = sample[key]
        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        # depth_gt = np.array(depth_gt).astype(np.float32) / 256
        # if do_flip:
        #     depth_gt = np.fliplr(depth_gt)

        return depth_gt
    
    @staticmethod
    def load_cart_as_float(path):
        raw_data = Image.open(path)
        raw_data = np.array(raw_data)
        cart_img = raw_data.astype(np.float32)[np.newaxis, :, :] / 255.
        # cart_img[cart_img < 0.3] = 0

        # Calculate new dimensions based on max_range and resolution
        new_width = 640
        new_height = 640

        # Rescale the image using scipy.ndimage.zoom
        height_scale = cart_img.shape[1] / new_height
        width_scale = cart_img.shape[2] / new_width
        zoom_factors = (1, 1/height_scale, 1/width_scale)
        resized_img = scipy.ndimage.zoom(cart_img, zoom_factors, order=0)
        resized_img[resized_img < 0.1] = 0
        # Convert back to uint8 range [0, 255] for PIL compatibility
        resized_img = (resized_img * 255).astype(np.uint8).squeeze()  # Remove channel dimension

        return Image.fromarray(resized_img)


    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        for i in self.frame_idxs:
            # inputs[("color", i, -1)] = self.get_color(self.samples[index], 'mono', i, do_flip)    
            inputs[("color", i, -1)], inputs[("color_id", i)] = self.get_color(self.samples[index], 'mono', i, do_flip)
            if self.load_radar_bev:
                inputs[("radar_bev", i)] = self.get_radar_bev(self.samples[index], 'radar_bev', i, do_flip)
            if self.load_radar_pov:
                inputs[("radar_pov", i)] = self.get_radar_pov(self.samples[index], 'radar_pov', i, do_flip)
        if self.load_unc:
            inputs[("unc", 0, -1)] = self.get_unc(self.samples[index], 'unc',  do_flip)
            
        
            

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            # color_aug = transforms.ColorJitter.get_params(
            #     self.brightness, self.contrast, self.saturation, self.hue)
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
        if self.load_unc:
            del inputs[("unc", 0, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(self.samples[index],'depth_gt',do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs
# inputs[("color", i, -1)], inputs[("color_id", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
