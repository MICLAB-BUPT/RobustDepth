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
from typing import List, Tuple

class MonoRadarSeqDataset(MonoSeqDataset):
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
        super(MonoRadarSeqDataset, self).__init__(*args, **kwargs)
        

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.skip_frames,
                            demi_length * self.skip_frames + 1, self.skip_frames))
        shifts.pop(demi_length)
        self.shifts = shifts
        for scene in self.scenes:
            mono_imgs = sorted(list((scene/self.mono_folder).glob('*.png')))
            radar_imgs = sorted(list((scene/self.radar_folder).glob('*.png')))
            radar_pov_imgs = sorted(list((scene/self.radar_pov_folder).glob('*.tiff')))
            
            
            if len(mono_imgs) < sequence_length or len(radar_imgs) < sequence_length:
                continue
            
            depth_gt_path = scene/self.depth_gt_dir
            depth_paths = sorted(list(depth_gt_path.glob('*.tiff')))
            depth_timestamps = [float(filename.name.split('_depth')[0]) for filename in depth_paths]
            
            f_mt = scene/self.stereo_timestamps
            mts = [float(folder.strip().split(':')[-1].strip())
                for folder in open(f_mt)]
            mts = mts[:len(mono_imgs)]
          
            f_rt = scene/self.radar_timestamps
            rts = [float(folder.strip().split(':')[-1].strip())
                    for folder in open(f_rt)]
            radar_idxs = list(
                range(demi_length * self.skip_frames, len(radar_imgs)-demi_length * self.skip_frames))
            cam_matches_all = self.find_cam_samples(
                radar_idxs, rts, mts) 
                          
            for cnt, i in enumerate(range(demi_length * self.skip_frames, len(radar_imgs)-demi_length * self.skip_frames)):
                cam_matches = cam_matches_all[cnt]
                if not cam_matches:
                    continue
                
                # add the radar sample
                sample = {}
                sample['radar_bev_0'] = radar_imgs[i]
                sample['radar_pov_0'] = (radar_pov_imgs[i]) 
                

                for j in shifts:
                    sample['radar_bev_{}'.format(j)] = (radar_imgs[i+j])
                    sample['radar_pov_{}'.format(j)] = (radar_pov_imgs[i+j]) 
                    
                # add the mono sample, cam_matches contains [tgt, tgt-1, tgt+1]
                sample['mono_0'] = mono_imgs[cam_matches[0]]
                for j in range(len(shifts)):
                    sample['mono_{}'.format(shifts[j])] = mono_imgs[cam_matches[j+1]]
                    
                if self.load_unc:
                    sample['unc'] = scene/self.unc_folder/(Path(mono_imgs[cam_matches[0]]).stem + '.tiff')

                if self.load_depth and depth_gt_path.exists():
                    cam_timestamp = mts[cam_matches[0]]
                    depth_timestamp = min(depth_timestamps, key=lambda x: abs(cam_timestamp - x))
                    depth_file = depth_gt_path/(str(depth_timestamp) + "_depth.tiff")
                    sample['depth_gt'] = depth_file
                sequence_set.append(sample)
        # if self.train:
        #     random.shuffle(sequence_set)
        self.samples = sequence_set

    def find_cam_samples(self, t_idxs: List[int], rts: List[List[float]], mts: List[List[float]]) -> List[List[int]]:
        """
        taken from gramme (nature machine intelligence)'s github code
        Returns indexes of monocular frames in the form of
        [[tgt, [src-1,...,tgt], [tgt,...,src+1]],
            [tgt, [src-1,...,tgt], [tgt,...,src+1]],...]

        Args:
            t_idx (List[int]): Indices of the target radar frames
            rts (List[float]): List of radar timestamps
            mts (List[float]): List of monocular timestamps

        Returns:
            List[List[int]]: Indexes of the matched monocular frames
        """
        t_matches = []
        last_search_idx = 0
        for t_idx in t_idxs:
            idxs = [self.find_nearest_mono_idx(rts[t_idx], mts, last_search_idx)]
            for s in self.shifts:
                idxs.append(self.find_nearest_mono_idx(
                    rts[t_idx+s], mts, last_search_idx))

            if any([i < 0 for i in idxs]) or len(set(idxs)) < len(self.shifts)+1:
                t_matches.append([])
                continue
            last_search_idx = idxs[1]
            t_matches.append(idxs)
            
        return t_matches
    
    @staticmethod
    def find_nearest_mono_idx(t: int, mts: List[List[float]], last_search_idx: int) -> int:
        """
        taken from gramme (nature machine intelligence)'s github code
        Finds the nearest monocular timestamp for the given radar timestamp

        Args:
            t (int): Timestamp of the target frame
            mts (List[List[float]]): List of monocular timestamps

        Returns:
            int: Index of the matched monocular frame
        """

        del_t = 0.050  # the match must be within 50ms of t
        # First check if t is outside monocular frames but still within thr close
        # Check if t comes before monocular frames
        if t < mts[last_search_idx]:
            return last_search_idx if mts[last_search_idx]-t < del_t else -1
        # Check if t comes after monocular frames
        if t > mts[-1]:
            return len(mts)-1 if t-mts[-1] < del_t else -1
        # Otherwise search within monocular frames
        for i in range(last_search_idx, len(mts)-1):
            if t > mts[i] and t < mts[i+1]:
                idx = i if (t-mts[i]) < (mts[i+1]-t) else i+1
                idx = idx if abs(mts[idx]-t) < del_t else -1
                return idx
        return -1