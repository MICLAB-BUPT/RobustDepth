# Adapted from https://github.com/TRI-ML/packnet-sfm/blob/c03e4bf929f202ff67819340135c53778d36047f/packnet_sfm/models/model_wrapper.py

import copy

from collections import OrderedDict
from utils.depth import compute_depth_metrics, post_process_inv_depth, inv2depth
from utils.image import flip_lr
from losses.TotalLoss import TotalLoss


import torch
import torch.nn.functional as F
from models.uncertainty_miner import UncertaintyMiner
from models.depth_net import DepthNet
import torch.nn as nn




class Evaluator:
    def __init__(self, model, cfg):
        assert cfg.DATASET.LOAD.GT.DEPTH

        self.model = model
        self.min_depth = cfg.EVALUATION.DEPTH.MIN_DEPTH
        self.max_depth = cfg.EVALUATION.DEPTH.MAX_DEPTH
        self.temp_context = cfg.DATASET.TEMP_CONTEXT
        self.vis_unc = cfg.EVALUATION.UNC.VIS_UNC
        if self.vis_unc:

            self.teacher_num = 2
            # import pdb; pdb.set_trace()
            self.unc_miner = UncertaintyMiner(pseudo_num=self.teacher_num, pseudo_input_dim=32)
            self.teacher_nets = nn.ModuleList()
            self.unc_miner.load_state_dict(state_dict={key.replace("loss.supervised_loss.unc_miner.", ""): weight for key, weight in torch.load(cfg.LOAD.CHECKPOINT_PATH)['state_dict'].items() if ('teacher_net' not in key and 'pose_model' not in key and 'depth' not in key)})
            self.unc_miner.to('cuda')
            for i in range(self.teacher_num):
                teacher_net = DepthNet(cfg)
                teacher_net.load_state_dict(state_dict={key.replace("depth_model.", ""): weight for key, weight in torch.load(cfg.LOAD.DAYTIME_TRANSLATION_TEACHER_PATHS[i])['state_dict'].items() if ('pose_model' not in key and 'loss' not in key)})
                for param in teacher_net.parameters():
                    param.requires_grad = False
                self.teacher_nets.append(teacher_net.to('cuda'))

        #     self.loss = TotalLoss(cfg, False)


        # Task metrics
        self.metrics_name = 'depth'
        self.metrics_keys = ('abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3')
        self.metrics_modes = ('', '_pp', '_gt', '_pp_gt')

        self.metric_conditions = ('all-conditions', 'day', 'night', 'clear', 'rain', 'day-clear', 'day-rain',
                                  'night-clear', 'night-rain') if cfg.EVALUATION.CONDITION_WISE else ('all-conditions',)

        # Dictionary for metrics in different conditions
        self.metrics = OrderedDict({condition: {mode: {metric: 0.0 for metric in self.metrics_keys} for mode in self.metrics_modes} for condition in self.metric_conditions})
        for condition in self.metrics.keys():
            self.metrics[condition]['count'] = 0

    @property
    def metrics_dicts(self):
        return self.metrics

    def reset_metrics(self):
        for condition in self.metric_conditions:
            for mode in self.metrics_modes:
                for metric in self.metrics_keys:
                    self.metrics[condition][mode][metric] = 0.0
            self.metrics[condition]['count'] = 0

    def compute_average_metrics(self):
        intermediate_res = copy.deepcopy(self.metrics)
        for condition in self.metric_conditions:
            for mode in self.metrics_modes:
                for metric in self.metrics_keys:
                    intermediate_res[condition][mode][metric] = self.metrics[condition][mode][metric] / self.metrics[condition]['count'] if self.metrics[condition]['count'] != 0 else 0.0

        return intermediate_res

    def compute_average_metrics_and_export(self):
        results = self.compute_average_metrics()
        export_dict = OrderedDict({condition: {'everything': {}} for condition in self.metric_conditions})
        for condition in self.metric_conditions:
            for mode in self.metrics_modes:
                for metric in self.metrics_keys:
                    export_dict[condition]['everything'][metric + mode] = results[condition][mode][metric]
            export_dict[condition]['everything']['count'] = float(results[condition]['count'])

        return export_dict

    def evaluate_depth(self, batch):
        """Evaluate batch to produce depth metrics."""
        # Get predicted depth
        # import pdb; pdb.set_trace()
        outputs, encoder_feats = self.model(batch["color", 0], batch["weather"])
        outputs['student_feats'] = encoder_feats

        if self.vis_unc:
            base_median  = 0.
            unc_in_list = []
            for i, teacher_net in enumerate(self.teacher_nets):
                with torch.no_grad():
                    gt_inv_depth, _ = teacher_net(batch[("color", 0)], batch['weather'], add_noise=True)
                    unc_in_depth = gt_inv_depth[("disp", 0, 0)] 
        
                    median = unc_in_depth.view(unc_in_depth.size(0), -1).median(dim=1).values
                    median += 1e-6
                    # align to the same scale
                    # import pdb; pdb.set_trace()
                    if i==0:
                        base_median = median
                    else:
                        unc_in_depth = unc_in_depth * (base_median.view(-1, 1, 1, 1) / median.view(-1, 1, 1, 1))
                unc_in_list.append(unc_in_depth)
            unc_map_list = self.unc_miner(unc_in_list, [outputs['student_feats'][0], outputs['student_feats'][3]])
            
            outputs["unc_maps"] = unc_map_list
            
        
        depth = inv2depth(outputs[("disp", 0, 0)])
        # Post-process predicted depth
        batch[("color", 0)] = flip_lr(batch[("color", 0)])
        inv_depths_flipped = self.model(batch["color", 0], batch["weather"])[0][("disp", 0, 0)]
        inv_depth_pp = post_process_inv_depth(outputs[("disp", 0, 0)], inv_depths_flipped, method='mean')
        depth_pp = inv2depth(inv_depth_pp)
        batch[("color", 0)] = flip_lr(batch[("color", 0)])
        # Count conditions in batch
        for i, weather in enumerate(batch["weather"]):
            gt_i = batch["depth_gt"][i]
            valid = (gt_i > self.min_depth) & (gt_i < self.max_depth)
            if valid.sum() == 0:
                continue
            for metric_condition in self.metric_conditions:
                if metric_condition in weather:
                    self.metrics[metric_condition]['count'] += 1
            self.metrics['all-conditions']['count'] += 1

        # Calculate predicted metrics
        for mode in self.metrics_modes:
            compute_depth_metrics(
                gt=batch["depth_gt"], pred=depth_pp if 'pp' in mode else depth, weather=batch["weather"],
                metrics=self.metrics, mode=mode, min_depth=self.min_depth,
                max_depth=self.max_depth, use_gt_scale='gt' in mode
            )

        return outputs

    def evaluate_batch(self, batch):
        return self.evaluate_depth(batch)
