# Adapted from https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/losses/supervised_loss.py

import torch
import torch.nn as nn

from config.config import get_cfg
from models.depth_net import DepthNet
import torch.nn.functional as F
from models.uncertainty_miner import UncertaintyMiner


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


class BerHuLoss(nn.Module):
    """Class implementing the BerHu loss."""
    def __init__(self, threshold=0.2):
        """
        Initializes the BerHuLoss class.
        Parameters
        ----------
        threshold : float
            Mask parameter
        """
        super().__init__()
        self.threshold = threshold

    def forward(self, pred, gt):
        """
        Calculates the BerHu loss.
        Parameters
        ----------
        pred : torch.Tensor [B,1,H,W]
            Predicted inverse depth map
        gt : torch.Tensor [B,1,H,W]
            Ground-truth inverse depth map
        Returns
        -------
        loss : torch.Tensor [1]
            BerHu loss
        """
        huber_c = torch.max(pred - gt)
        huber_c = self.threshold * huber_c
        diff = (pred - gt).abs()

        # Remove
        # mask = (gt > 0).detach()
        # diff = gt - pred
        # diff = diff[mask]
        # diff = diff.abs()

        huber_mask = (diff > huber_c).detach()
        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2
        return torch.cat((diff, diff2)).mean()


class SilogLoss(nn.Module):
    def __init__(self, ratio=10, ratio2=0.85):
        super().__init__()
        self.ratio = ratio
        self.ratio2 = ratio2

    def forward(self, pred, gt):
        log_diff = torch.log(pred * self.ratio) - \
                   torch.log(gt * self.ratio)
        silog1 = torch.mean(log_diff ** 2)
        silog2 = self.ratio2 * (log_diff.mean() ** 2)
        silog_loss = torch.sqrt(silog1 - silog2) * self.ratio
        return silog_loss


def get_loss_func(supervised_method):
    """Determines the supervised loss to be used, given the supervised method."""
    if supervised_method.endswith('l1'):
        return nn.L1Loss()
    elif supervised_method.endswith('mse'):
        return nn.MSELoss()
    elif supervised_method.endswith('berhu'):
        return BerHuLoss()
    elif supervised_method.endswith('silog'):
        return SilogLoss()
    elif supervised_method.endswith('abs_rel'):
        return lambda x, y: torch.mean(torch.abs(x - y) / x)
    elif supervised_method.endswith('mysim'):
        return lambda x, y: torch.mean(torch.abs(x - y) / y)
    elif supervised_method.endswith('unc_sim'):
        return lambda x, y, z: torch.mean(torch.exp(-z) * (torch.abs(x - y) / y) + z , dim=(1, 2, 3))
    else:
        raise ValueError('Unknown supervised loss {}'.format(supervised_method))


class SupervisedLoss(nn.Module):
    """
    Supervised loss for inverse depth maps.
    Parameters
    """
    def __init__(self, cfg, is_train):
        super().__init__()
        self.supervised_method = cfg.LOSS.SUPERVISED.METHOD
        self.loss_func = get_loss_func(self.supervised_method)
        self.scales = cfg.DATASET.SCALES
        self.min_depth=cfg.MODEL.DEPTH.MIN_DEPTH
        self.max_depth=cfg.MODEL.DEPTH.MAX_DEPTH
        self.teacher_num = cfg.LOSS.SUPERVISED.TEACHER_NUM
        self.unc_miner = UncertaintyMiner(pseudo_num=self.teacher_num, pseudo_input_dim=32)

        # Clear/daytime teacher: provides reliable paired pseudo-labels (used as the single
        # pseudo-label provider in expert-training mode, teacher_num == 1).
        self.teacher_net = DepthNet(cfg)
        if is_train and cfg.LOAD.DAYTIME_TRANSLATION_TEACHER_PATH is not None:
            self.teacher_net.load_state_dict(state_dict={key.replace("depth_model.", ""): weight for key, weight in torch.load(cfg.LOAD.DAYTIME_TRANSLATION_TEACHER_PATH)['state_dict'].items() if ('pose_model' not in key and 'loss' not in key)}, strict=False)
        for param in self.teacher_net.parameters():
            param.requires_grad = False

        # Weather-expert teachers: only needed for multi-teacher UAMTD (teacher_num >= 2).
        self.teacher_nets = nn.ModuleList()
        if self.teacher_num >= 2:
            for i in range(self.teacher_num):
                teacher_net = DepthNet(cfg)
                if is_train:
                    teacher_net.load_state_dict(state_dict={key.replace("depth_model.", ""): weight for key, weight in torch.load(cfg.LOAD.DAYTIME_TRANSLATION_TEACHER_PATHS[i])['state_dict'].items() if ('pose_model' not in key and 'loss' not in key)}, strict=False)
                for param in teacher_net.parameters():
                    param.requires_grad = False
                self.teacher_nets.append(teacher_net)

    def calculate_loss(self, inv_depths, gt_inv_depths, unc_map=None):
        """
        Calculate the supervised loss.
        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            List of predicted inverse depth maps
        gt_inv_depths : list of torch.Tensor [B,1,H,W]
            List of ground-truth inverse depth mapsss
        Returns
        -------
        loss : torch.Tensor [1]
            Average supervised loss for all scales
        """
        # If using a sparse loss, mask invalid pixels for all scales
        # import pdb; pdb.set_trace()
        if self.supervised_method.startswith('sparse'):
            for i in self.scales:
                mask = (gt_inv_depths[i] > 0.).detach()
                inv_depths[i] = inv_depths[i][mask]
                gt_inv_depths[i] = gt_inv_depths[i][mask]

        pred = inv_depths
        gt = gt_inv_depths
        # import pdb; pdb.set_trace()
        if self.supervised_method.endswith('mysim'):
            for i in self.scales:
                pred[i] = disp_to_depth(pred[i], self.min_depth, self.max_depth)[1]
                gt[i] = disp_to_depth(gt[i], self.min_depth, self.max_depth)[1]

        # Only the uncertainty-aware loss consumes the per-pixel uncertainty map; plain
        # Single-Teacher Distillation (e.g. 'mysim' = mean(|teacher-student|/|teacher|)) does not.
        if self.supervised_method.endswith('unc_sim'):
            return sum([self.loss_func(pred[i], gt[i], F.interpolate(unc_map, (gt[i].size(-2), gt[i].size(-1)), mode='nearest')) for i in self.scales]) / len(self.scales)
        return sum([self.loss_func(pred[i], gt[i]) for i in self.scales]) / len(self.scales)

    def forward(self, inputs, outputs):
        """
        Calculates training supervised loss.

        Two modes (selected by cfg.LOSS.SUPERVISED.TEACHER_NUM):
          * teacher_num == 1  -> expert-training mode: a single clear/daytime teacher provides
            pseudo-labels on inputs[("color", 0)]. For synthetic (translated) samples the
            base image is clear, so the label is reliable (paired); for real adverse samples the
            label is noisy, and the per-pixel uncertainty down-weights it automatically.
          * teacher_num >= 2  -> UAMTD multi-teacher distillation (unchanged core method).
        """
        inv_depths = [outputs[("disp", 0, scale)] for scale in self.scales]
        student_feats = [outputs['student_feats'][0], outputs['student_feats'][3]]

        # ----- Expert-training mode: single clear teacher -----
        if self.teacher_num == 1:
            with torch.no_grad():
                gt_inv_depth, _ = self.teacher_net(inputs[("color", 0)], inputs['weather'], add_noise=True)
                gt_inv_depths = [gt_inv_depth[("disp", 0, scale)] for scale in self.scales]
            # Single-Teacher Distillation (STD, paper eq.4): plain similarity loss, NO uncertainty.
            # Uncertainty is reserved for the UAMTD multi-teacher step (teacher_num >= 2).
            if self.supervised_method.endswith('unc_sim'):
                unc_in_list = [gt_inv_depths[0].clone()]
                unc_map_list = self.unc_miner(unc_in_list, student_feats)
                return self.calculate_loss(inv_depths, gt_inv_depths, unc_map=unc_map_list[0]).mean()
            return self.calculate_loss(inv_depths, gt_inv_depths).mean()

        # ----- UAMTD multi-teacher distillation -----
        loss = 0.
        unc_in_list = []
        gt_inv_depths_list = []
        base_median = 0.
        for i, teacher_net in enumerate(self.teacher_nets):
            with torch.no_grad():
                gt_inv_depth, _ = teacher_net(inputs[("color", 0)], inputs['weather'], add_noise=True)
                gt_inv_depths = [gt_inv_depth[("disp", 0, scale)] for scale in self.scales]
                gt_inv_depths_list.append(gt_inv_depths)
                unc_in_depth = gt_inv_depths[0].clone()

                median = unc_in_depth.view(unc_in_depth.size(0), -1).median(dim=1).values
                median += 1e-6
                # align teachers to a common scale
                if i == 0:
                    base_median = median
                else:
                    unc_in_depth = unc_in_depth * (base_median.view(-1, 1, 1, 1) / median.view(-1, 1, 1, 1))
            unc_in_list.append(unc_in_depth)

        unc_map_list = self.unc_miner(unc_in_list, student_feats)
        for i in range(len(self.teacher_nets)):
            loss = loss + self.calculate_loss(inv_depths, gt_inv_depths_list[i], unc_map=unc_map_list[i])
        return loss.mean()
