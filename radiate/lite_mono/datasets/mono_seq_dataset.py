from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
from pathlib import Path
import torchvision.transforms.functional as F


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoSeqDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        mode: train/val/test/...
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 mode,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 sequence=None,
                 load_depth=False,
                 load_unc=False, 
                 mono_folder='stereo_undistorted/left',
                 depth_gt_dir='depth_ac',
                 stereo_timestamps='zed_left.txt',
                 radar_timestamps='Navtech_Polar.txt',
                 radar_folder='Navtech_Cartesian',
                 unc_folder='unc',
                 load_radar_bev=False,
                 load_radar_pov=False, 
                 radar_pov_folder='radar_pov'):
        super(MonoSeqDataset, self).__init__()
        
        
        self.root = Path(data_path)
        if sequence is not None:
            self.scenes = [self.root/sequence]
        else:
            scene_list_path = self.root / f"{mode}.txt"
            self.scenes = [self.root/folder.strip()
                           for folder in open(scene_list_path) if not folder.strip().startswith("#")]

        self.data_path = data_path
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = F.InterpolationMode.BILINEAR

        self.frame_idxs = frame_idxs
        self.mode = mode
        self.is_train = self.mode.startswith('train')
        # self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.skip_frames = 1
        self.mono_folder = mono_folder
        self.depth_gt_dir = depth_gt_dir
        self.stereo_timestamps = stereo_timestamps
        self.radar_timestamps = radar_timestamps
        self.radar_folder = radar_folder
        self.unc_folder = unc_folder
        self.load_unc = load_unc
        self.load_radar_bev = load_radar_bev
        self.load_radar_pov = load_radar_pov
        self.radar_pov_folder  = radar_pov_folder
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = load_depth
        
        self.crawl_folders(len(frame_idxs))
        
        '''
            adapt to radiate dataset structure
        '''

        
        
    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.skip_frames,
                            demi_length * self.skip_frames + 1, self.skip_frames))
        shifts.pop(demi_length)
        for scene in self.scenes:
            
            
            imgs = sorted(list((scene/self.mono_folder).glob('*.png')))
            
            depth_gt_path = scene/self.depth_gt_dir
            depth_paths = sorted(list(depth_gt_path.glob('*.tiff')))
            depth_timestamps = [float(filename.name.split('_depth')[0]) for filename in depth_paths]
            
            f_mt = scene/self.stereo_timestamps
            mts = [float(folder.strip().split(':')[-1].strip())
                for folder in open(f_mt)]
            mts = mts[:len(imgs)]
          
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length * self.skip_frames, len(imgs)-demi_length * self.skip_frames):
                sample = {
                          'mono_0': imgs[i]
                          }
                for j in shifts:
                    sample['mono_{}'.format(j)] = (imgs[i+j])

                if self.load_depth and depth_gt_path.exists():
                    cam_timestamp = mts[i]
                    depth_timestamp = min(depth_timestamps, key=lambda x: abs(cam_timestamp - x))
                    depth_file = depth_gt_path/(str(depth_timestamp) + "_depth.tiff")
                    sample['depth_gt'] = depth_file
                sequence_set.append(sample)
        # if self.train:
        #     random.shuffle(sequence_set)
        self.samples = sequence_set


    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k or "unc" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            if "unc" in k:
                # no aug for uncertainty
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
            if "radar_bev" in k:
                n, im = k
                inputs[(n, im)] = self.to_tensor(f)
            if "radar_pov" in k:
                n, im = k 
                radar_pov_tensor = self.to_tensor(f)
                inputs[(n, im)] = self.radar_vertical_aug(radar_pov_tensor)
                
                
    @staticmethod
    def radar_vertical_aug(radar_pov):

        radar_pov_squeezed = radar_pov.squeeze(0)  # shape becomes [h, w]
        radar_pov_squeezed_nonzero = radar_pov_squeezed.clone()
        radar_pov_squeezed_nonzero[radar_pov_squeezed_nonzero == 0] = float('inf')
        
        # Find the minimum non-zero value in each column (along height dimension)
        min_vals, _ = radar_pov_squeezed_nonzero.min(dim=0, keepdim=True)  # shape becomes [1, w]
        min_vals[min_vals == float('inf')] = 0
        radar_pov_augmented = min_vals.repeat(radar_pov_squeezed.size(0), 1)  # shape becomes [h, w]
        radar_pov_augmented = radar_pov_augmented.unsqueeze(0)  # shape becomes [1, h, w]
        
        return radar_pov_augmented
                
                

    def __len__(self):
        return len(self.samples)

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
            inputs[("color", i, -1)] = self.get_color(self.samples[index], 'mono', i, do_flip)

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

        if self.load_depth:
            depth_gt = self.get_depth(self.samples[index],'depth_gt',do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs

    def get_color(self, sample, key, shift, do_flip):
        raise NotImplementedError

    def get_depth(self, sample, key, do_flip):
        raise NotImplementedError
