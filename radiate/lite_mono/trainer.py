from __future__ import absolute_import, division, print_function


import time
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import json

from utils import *
from kitti_utils import *
from layers import *


import datasets
import networks
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler
from utils import tensor2array

import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import get_dist_info, DistributedSampler
import os
import copy
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
# torch.backends.cudnn.benchmark = True


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        
        self.opt.distill = self.opt.distill_weight is not None

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.models_pose = {}
        self.parameters_to_train = []
        self.parameters_to_train_pose = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        
        # ddp setting
        self.local_rank = self.opt.local_rank
        torch.cuda.set_device(self.local_rank)
        if self.opt.ddp:
            print("using ddp")
            dist.init_process_group(backend='nccl', )
        print("norm: {}".format(self.opt.norm))
        print("use_radar_gate: {}".format(self.opt.use_radar_gate))
        
        self.profile = self.opt.profile
        self.num_scales = len(self.opt.scales)
        self.frame_ids = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.LiteMono(model=self.opt.model,
                                                   drop_path_rate=self.opt.drop_path,
                                                   width=self.opt.width, height=self.opt.height,
                                                   norm=self.opt.norm)
        if self.opt.ddp:
            self.models["encoder"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["encoder"])
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc,
                                                     self.opt.scales)
        if self.opt.ddp:
            self.models["depth"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["depth"])
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        
        if self.opt.use_radar_bev:
            self.models["radar_bev_encoder"] = networks.RadarResnetEncoder(num_layers=18, pretrained=False, num_input_images=1, n_img_channels=1)
            self.models["pov_bev_attn_down8"] = networks.PBCrossAttention(mono_height=self.opt.height, mono_width=self.opt.width,
                                                                          downsample_rate=8,
                                                                          mono_c=128, radar_c=128,
                                                                          num_heads=4,
                                                                          qkv_bias=True, attn_drop=0.,
                                                                          use_radar_gate=self.opt.use_radar_gate)
            
            self.models["pov_bev_attn_down16"] = networks.PBCrossAttention(mono_height=self.opt.height, mono_width=self.opt.width,
                                                                          downsample_rate=16,
                                                                          mono_c=224, radar_c=256,
                                                                          num_heads=4,
                                                                          qkv_bias=True, attn_drop=0.,
                                                                          use_radar_gate=self.opt.use_radar_gate)
            self.models["pov_bev_attn_down8"]  = self.models["pov_bev_attn_down8"].to(self.device)
            self.models["pov_bev_attn_down16"]  = self.models["pov_bev_attn_down16"].to(self.device)
            self.models["radar_bev_encoder"]  = self.models["radar_bev_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["pov_bev_attn_down8"].parameters())
            self.parameters_to_train += list(self.models["pov_bev_attn_down16"].parameters())
            self.parameters_to_train += list(self.models["radar_bev_encoder"].parameters())
            
            self.models["radar_fusion"] = networks.RadarFusion(mono_c=224, radar_c=256)
            self.models["radar_fusion"] = self.models["radar_fusion"].to(self.device)
            self.parameters_to_train += list(self.models["radar_fusion"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models_pose["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames,
                    norm=self.opt.norm)

                self.models_pose["pose_encoder"].to(self.device)
                self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())

                self.models_pose["pose"] = networks.PoseDecoder(
                    self.models_pose["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models_pose["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models_pose["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)
            if self.opt.ddp:
                self.models_pose["pose"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models_pose["pose"])
            self.models_pose["pose"].to(self.device)
            self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())


        # config the new module
        pseduo_input_dim = 32
        pseduo_num = 2
        self.models["unc_miner"] = networks.UncertaintyMiner(pseudo_num=pseduo_num, pseudo_input_dim=pseduo_input_dim).to(self.device)
        self.parameters_to_train += list(self.models["unc_miner"].parameters())

        if self.opt.ddp:
            for key in self.models.keys():
                self.models[key] = DDP(self.models[key], device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False, find_unused_parameters=True,)
            for key in self.models_pose.keys():
                self.models_pose[key] = DDP(self.models_pose[key], device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False, find_unused_parameters=True,)
        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
        if self.use_pose_net:
            self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opt.lr[3], weight_decay=self.opt.weight_decay)

        self.model_lr_scheduler = ChainedScheduler(
                            self.model_optimizer,
                            T_0=int(self.opt.lr[2]),
                            T_mul=1,
                            eta_min=self.opt.lr[1],
                            last_epoch=-1,
                            max_lr=self.opt.lr[0],
                            warmup_steps=0,
                            gamma=0.9
                        )
        self.model_pose_lr_scheduler = ChainedScheduler(
            self.model_pose_optimizer,
            T_0=int(self.opt.lr[5]),
            T_mul=1,
            eta_min=self.opt.lr[4],
            last_epoch=-1,
            max_lr=self.opt.lr[3],
            warmup_steps=0,
            gamma=0.9
        )

        if self.opt.distill:
            assert self.opt.load_weights_folder is not None
        if self.opt.load_weights_folder is not None:
            self.load_model_only()

        if self.opt.distill:
            assert self.opt.mypretrain is  None
        if self.opt.mypretrain is not None:
            self.load_pretrain()
        
              
        if self.opt.distill:
            self.models_teachers = []
            
            self.models_teachers.append(self.get_homoteacher(self.opt.t1_path))  
            self.models_teachers.append(self.get_homoteacher(self.opt.t2_path))  
            
            for models_teacher in self.models_teachers:
                for model in models_teacher.values():
                    for param in model.parameters():
                        param.requires_grad = False


            # Freeze all parameters in self.models_pose
            for model in self.models_pose.values():
                for param in model.parameters():
                    param.requires_grad = False
                          
            #----- config the uncertainy minig block ----------
            if self.opt.resume_weights_folder is not None:
                self.load_model_resume()

        self.local_rank0_print("Training model named:\n  ", self.opt.model_name)
        self.local_rank0_print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        self.local_rank0_print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "radiate_mono":datasets.RadiateMonoDataset,
                         "radiate_mono_radar":datasets.RadiateMonoRdarDataset}
        # self.dataset = datasets_dict[self.opt.dataset]

        train_mono_radar_dataset = datasets.RadiateMonoRdarDataset(
            data_path=self.opt.data_path, mode=self.opt.train_mode, height=self.opt.height, width=self.opt.width,
            frame_idxs=self.opt.frame_ids, num_scales=4, load_depth=False, load_unc=self.opt.distill_unc, load_radar_bev=True, load_radar_pov=True)
        train_mono_dataset = datasets.RadiateMonoDataset(
            data_path=self.opt.data_path, mode=self.opt.train_mode, height=self.opt.height, width=self.opt.width,
            frame_idxs=self.opt.frame_ids, num_scales=4, load_depth=False, load_unc=self.opt.distill_unc)
        self.num_total_steps = len(train_mono_dataset) // self.opt.batch_size * self.opt.num_epochs
        
        if self.opt.ddp:
            train_mono_sampler = torch.utils.data.distributed.DistributedSampler(train_mono_dataset, shuffle=True)
            self.train_mono_loader = DataLoader(
                train_mono_dataset, self.opt.batch_size, shuffle=False,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, sampler=train_mono_sampler)  
            
            train_mono_radar_sampler = torch.utils.data.distributed.DistributedSampler(train_mono_radar_dataset, shuffle=True)
            self.train_mono_radar_loader = DataLoader(
                train_mono_radar_dataset, self.opt.batch_size, shuffle=False,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, sampler=train_mono_radar_sampler)  
        else:
            self.train_mono_loader = DataLoader(
                train_mono_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,
                worker_init_fn=seed_worker)     
            self.train_mono_radar_loader = DataLoader(
                train_mono_radar_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,
                worker_init_fn=seed_worker)     
 
        
        val_dataset = datasets.RadiateMonoRdarDataset(
            data_path=self.opt.data_path, mode='val', height=self.opt.height, width=self.opt.width,
            frame_idxs=self.opt.frame_ids, num_scales=4, load_depth=True, load_unc=self.opt.distill_unc)
        val_fog_dataset = datasets.RadiateMonoRdarDataset(
            data_path=self.opt.data_path, sequence='fog_6_0', mode='val', height=self.opt.height, width=self.opt.width,
            frame_idxs=self.opt.frame_ids, num_scales=4, load_depth=True, load_unc=self.opt.distill_unc)
        if self.opt.ddp:
            rank, world_size = get_dist_info()
            self.world_size = world_size
            val_sampler = DistributedSampler(val_dataset, world_size, rank, shuffle=False)
            self.val_loader = DataLoader(
                val_dataset, self.opt.batch_size, shuffle=False,
                num_workers=4, pin_memory=True, drop_last=False, sampler=val_sampler)
        else:
            self.val_loader = DataLoader(
                val_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
            self.val_fog_loader = DataLoader(
                val_fog_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)
        self.val_fog_iter = iter(self.val_fog_loader)

        self.writers = {}
        for mode in ["train", "val", "val_fog"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        # self.local_rank0_print("Using split:\n  ", self.opt.split)
        # if self.local_rank == 0:
        self.local_rank0_print("There are {:d}, {:d} training items and {:d} validation items\n".format(
            len(train_mono_dataset), len(train_mono_radar_dataset), len(val_dataset)))
        if self.opt.ddp:
            self.opt.log_frequency = self.opt.log_frequency//self.world_size

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
        
            
            if self.opt.ddp:
                self.train_mono_loader.sampler.set_epoch(self.epoch)
                self.train_mono_radar_loader.sampler.set_epoch(self.epoch)
                
            if self.epoch % 2 == 0:
                train_loader = self.train_mono_loader
            else:
                train_loader = self.train_mono_radar_loader
            self.run_epoch(train_loader)
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
            # if self.opt.distill:
            #     for key in self.models_teachers[0].keys():
            #         state_dict = copy.deepcopy(self.models[key].state_dict())
            #         self.models_teachers[0][key].load_state_dict(state_dict)
                    
            #     for models_teacher in self.models_teachers:
            #         for model in models_teacher.values():
            #             for param in model.parameters():
            #                 param.requires_grad = False


    def run_epoch(self, train_loader):
        """Run a single epoch of training and validation
        """

        self.local_rank0_print("Training")
        self.set_train()

        self.model_lr_scheduler.step()
        if self.use_pose_net:
            self.model_pose_lr_scheduler.step()

        for batch_idx, inputs in enumerate(train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            if self.use_pose_net:
                self.model_pose_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            if self.use_pose_net and not self.opt.distill:    
                self.model_pose_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 20000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()
                self.val_fog()
                

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if isinstance(ipt, torch.Tensor):
                inputs[key] = ipt.to(self.device)


        if ('radar_bev',  0) in inputs:
            # get the last feats and use the cross fusion
            bsz = inputs["color_aug", 0, 0].size(0)
            pov_inputs = torch.cat([ inputs["color_aug", 0, 0],
                                    #  inputs["color_aug", -1, 0],
                                    #  inputs["color_aug", 1, 0],
                                     ])
            
            bev_inputs = torch.cat([ inputs["radar_bev", 0][:, :, :320, :],
                                    #  inputs["radar_bev", -1][:, :, :320, :],
                                    #  inputs["radar_bev", 1][:, :, :320, :],
                                     ])
            
            radar_pov_inputs = torch.cat([ inputs["radar_pov", 0],
                                    #  inputs["radar_bev", -1][:, :, :320, :],
                                    #  inputs["radar_bev", 1][:, :, :320, :],
                                     ])

            mono_feats_list = self.models["encoder"](pov_inputs)
            features = [mono_feats_list[i][:bsz, ...] for i in range(len(mono_feats_list))]
            mono_feats_down8 = mono_feats_list[-2]
            mono_feats_down16 = mono_feats_list[-1]
            
            
            radar_bev_feats_list = self.models["radar_bev_encoder"](bev_inputs)
            radar_bev_feats_down8 = radar_bev_feats_list[-3]
            radar_bev_feats_down16 = radar_bev_feats_list[-2]   
            
            bp_feats_down8 = self.models["pov_bev_attn_down8"](mono_feats_down8, radar_bev_feats_down8)
            bp_feats_down16 = self.models["pov_bev_attn_down16"](mono_feats_down16, radar_bev_feats_down16)
            

            features[-2] = features[-2] + bp_feats_down8[:bsz, ...]
            features[-1] = features[-1] + bp_feats_down16[:bsz, ...]
            
            fusion_feats = self.models["radar_fusion"](radar_pov_inputs, features[-1])
            features[-1] = fusion_feats
            
            # contrastive_loss_down8, sim_acc_down8 = self.compute_contrastive_loss(mono_feats_down8, bp_feats_down8)
            # contrastive_loss_down16, sim_acc_down16 = self.compute_contrastive_loss(mono_feats_down16, bp_feats_down16)
            
        else:
            features = self.models["encoder"](inputs["color_aug", 0, 0])
        
        outputs = self.models["depth"](features)    

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs, student_features=features)
        
        # if ('radar_bev', 0) in inputs:
        #     losses["loss_contrastive_down8"] = contrastive_loss_down8
        #     losses["loss_contrastive_down16"] = contrastive_loss_down16
            
        #     losses["sim_acc_down8"] = sim_acc_down8
        #     losses["sim_acc_down16"] = sim_acc_down16
            
        #     losses["loss"] += ( contrastive_loss_down8 + contrastive_loss_down16 )

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models_pose["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models_pose["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()
        
        
    def val_fog(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_fog_iter.next()
        except StopIteration:
            self.val_fog_iter = iter(self.val_fog_loader)
            inputs = self.val_fog_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val_fog", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs, student_features):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0
        
        if self.opt.distill:  
            pseudo_outputs_list = []  
            with torch.no_grad():
                base_medians = []
                for i, models_teacher in enumerate(self.models_teachers):
                    # always use the init input (w/o aug) for the teacher
                    features = models_teacher["encoder"](inputs["color", 0, 0])
                    pseudo_outputs = models_teacher["depth"](features)
                    outputs["teacher_{}_pred".format(i)] = pseudo_outputs["disp", 0]
                    for scale in self.opt.scales:
                        pseudo_output = pseudo_outputs[('disp', scale)]
                        
                        median = pseudo_output.view(pseudo_output.size(0), -1).median(dim=1).values
                        median += 1e-6
                        # align to the same scale
                        if i==0:
                            base_medians.append(median)
                        else:
                            pseudo_output = pseudo_output * (base_medians[scale].view(-1, 1, 1, 1) / median.view(-1, 1, 1, 1))
                        pseudo_outputs[scale] = pseudo_output
                        pseudo_outputs.pop(('disp', scale))
                    pseudo_outputs_list.append(pseudo_outputs)
                   
            uncs = self.models["unc_miner"](pseudo_outputs_list, [student_features[0], student_features[2]])
            for i in range(len(uncs)):
                outputs["teacher_{}_unc_map".format(i)] = uncs[i]
            

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            
            if self.opt.distill:
                distill_loss = 0.
                for i in range(len(pseudo_outputs_list)):
                    distill_loss += self.compute_pseudo_dist(disp_to_depth(outputs[("disp", scale)], 0.1, 100)[1], disp_to_depth(pseudo_outputs_list[i][scale], 0.1, 100)[1], uncs[i])
                    # distill_loss += self.compute_pseudo_dist(disp_to_depth(outputs[("disp", scale)], 0.1, 100)[1], disp_to_depth(pseudo_outputs_list[i][scale], 0.1, 100)[1], None)     
            else:
                distill_loss = torch.tensor(0.0, device=self.device)
            
            total_loss += (loss + self.opt.distill_weight*distill_loss)
            losses["loss/{}".format(scale)] = loss
            losses["loss_distill/{}".format(scale)] = distill_loss
            

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses
    
    
    def compute_pseudo_dist(self, pred_depth, pseudo_depth, unc_map=None):
        
        pseudo_depth = pseudo_depth.detach() # double ensure no grad
        if unc_map is None:
            return torch.mean(torch.abs(pred_depth - pseudo_depth) / pseudo_depth)
        else:
            unc_map = F.interpolate(unc_map, (pred_depth.size(-2), pred_depth.size(-1)), mode='nearest')
            loss = torch.mean(torch.exp(-unc_map) * (torch.abs(pred_depth - pseudo_depth) / pseudo_depth) + unc_map)
            # loss = torch.mean(torch.abs(pred_depth - pseudo_depth) / pseudo_depth)
            
            return loss
            # means = torch.mean(unc_maps, dim=(1, 2, 3))
            # scale_factors = torch.clamp(torch.exp(means), min=1, max=2).view(-1, 1, 1, 1)
            # unc_maps =  1 + self.norm_01(unc_maps)
            # unc_maps = scale_factors * unc_maps
            
            # return torch.mean(unc_maps * (torch.abs(pred_depth - pseudo_depth) / pseudo_depth))
            

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [372, 672], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        # TODO may be radiate also need to mask
        # crop_mask = torch.zeros_like(mask)
        # crop_mask[:, :, 153:371, 44:1197] = 1
        # mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """self.local_rank0_print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | lr {:.6f} |lr_p {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        self.local_rank0_print(print_string.format(self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
                                  self.model_pose_optimizer.state_dict()['param_groups'][0]['lr'],
                                  batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    tensor2array(normalize_image(disp_to_depth(outputs[("disp", s)][j], 0.1, 100)[0]),  colormap='inferno'), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)
                    
            for i in range(len(self.models_teachers)):
                
                writer.add_image(
                    "teacher_{}_pred/{}".format(i, j),
                    tensor2array(normalize_image(disp_to_depth(  outputs["teacher_{}_pred".format(i)][j], 0.1, 100  )[0]),  colormap='inferno'), self.step)
                writer.add_image(
                    "teacher_{}_unc_map/{}".format(i, j),
                    tensor2array(normalize_image(outputs["teacher_{}_unc_map".format(i)][j]),  colormap='inferno'), self.step)
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        if self.local_rank!=0:
            return
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            if self.opt.ddp:
                to_save = model.module.state_dict()
            else:
                to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        for model_name, model in self.models_pose.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            if self.opt.ddp:
                to_save = model.module.state_dict()
            else:
                to_save = model.state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam_pose"))
        if self.use_pose_net:
            torch.save(self.model_pose_optimizer.state_dict(), save_path)

    def load_pretrain(self):
        self.opt.mypretrain = os.path.expanduser(self.opt.mypretrain)
        path = self.opt.mypretrain
        model_dict = self.models["encoder"].state_dict()
        pretrained_dict = torch.load(path)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
        model_dict.update(pretrained_dict)
        self.models["encoder"].load_state_dict(model_dict)
        self.local_rank0_print('mypretrain loaded.')

    def load_model_only(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        self.local_rank0_print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            self.local_rank0_print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))

            if n in ['pose_encoder', 'pose']:
                model_dict = self.models_pose[n].state_dict()
                if self.opt.ddp:
                    pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
                else:
                    pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models_pose[n].load_state_dict(model_dict)
            else:
                model_dict = self.models[n].state_dict()
                if self.opt.ddp:
                    pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
                else:
                    pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)
            

    def load_model_resume(self):
        """Load model(s) from disk
        """
        self.opt.resume_weights_folder = os.path.expanduser(self.opt.resume_weights_folder)

        assert os.path.isdir(self.opt.resume_weights_folder), \
            "Cannot find folder {}".format(self.opt.resume_weights_folder)
        self.local_rank0_print("loading model from folder {}".format(self.opt.resume_weights_folder))

        for n in self.opt.models_to_load:
            self.local_rank0_print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.resume_weights_folder, "{}.pth".format(n))

            if n in ['pose_encoder', 'pose']:
                model_dict = self.models_pose[n].state_dict()
                if self.opt.ddp:
                    pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
                else:
                    pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models_pose[n].load_state_dict(model_dict)
            else:
                model_dict = self.models[n].state_dict()
                if self.opt.ddp:
                    pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
                else:
                    pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

        # loading adam state

        optimizer_load_path = os.path.join(self.opt.resume_weights_folder, "adam.pth")
        optimizer_pose_load_path = os.path.join(self.opt.resume_weights_folder, "adam_pose.pth")
        if os.path.isfile(optimizer_load_path):
            self.local_rank0_print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            optimizer_pose_dict = torch.load(optimizer_pose_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
            self.model_pose_optimizer.load_state_dict(optimizer_pose_dict)
        else:
            self.local_rank0_print("Cannot find Adam weights so Adam is randomly initialized")
            
    def local_rank0_print(self, *args, **kwargs):
        if self.local_rank == 0:
            print(*args, **kwargs)

    @staticmethod
    def norm_01(tensor):
        """
        Normalize the input tensor to the range (0, 1).
        
        Args:
        - tensor: Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
        - Tensor: Normalized tensor with values in range (0, 1).
        """
        # Find the minimum and maximum values in the tensor
        min_val = tensor.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]  # Min over h, w
        max_val = tensor.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # Max over h, w
        
        # Perform the min-max normalization
        normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-7)  # Add small epsilon to avoid division by zero
        
        return normalized_tensor

    def get_md2model(self):
        models = {}
        models["encoder"] = networks.ResnetEncoder(
                50, False).to(self.device)
        models["depth"] = networks.MD2DepthDecoder(
            models["encoder"].num_ch_enc, self.opt.scales).to(self.device)
        
        
        load_weights_folder = os.path.expanduser(self.opt.md2_load_weights_folder)

        assert os.path.isdir(load_weights_folder), \
            "Cannot find folder {}".format(load_weights_folder)
        print("loading model from folder {}".format(load_weights_folder))

        for n in self.opt.models_to_load:
            if not (n == "encoder" or n == 'depth'):
                continue
            print("Loading {} weights...".format(n))
            path = os.path.join(load_weights_folder, "{}.pth".format(n))
            model_dict = models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            models[n].load_state_dict(model_dict)
        
        return models
    
    def get_monovit(self):
        models = {}
        models["encoder"] = networks.mpvit_small().to(self.device)
        models["depth"] = networks.MonovitDepthDecoder().to(self.device)
        
        load_weights_folder = os.path.expanduser(self.opt.monovit_load_weights_folder)

        assert os.path.isdir(load_weights_folder), \
            "Cannot find folder {}".format(load_weights_folder)
        print("loading model from folder {}".format(load_weights_folder))
        
        for n in self.opt.models_to_load:
            if not (n == "encoder" or n == 'depth'):
                continue
            print("Loading {} weights...".format(n))
            path = os.path.join(load_weights_folder, "{}.pth".format(n))
            model_dict = models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            models[n].load_state_dict(model_dict)
        
        return models
    
    def get_homoteacher(self, ckpt_path):
        models = {}
        models["encoder"] = networks.LiteMono(model=self.opt.model,
                                                drop_path_rate=self.opt.drop_path,
                                                width=self.opt.width, height=self.opt.height,
                                                norm=self.opt.norm)
        models["encoder"].to(self.device)
        models["depth"] = networks.DepthDecoder(models["encoder"].num_ch_enc,
                                                        self.opt.scales)
        models["depth"].to(self.device)
        
        load_weights_folder = os.path.expanduser(ckpt_path)

        assert os.path.isdir(load_weights_folder), \
            "Cannot find folder {}".format(load_weights_folder)
        print("loading model from folder {}".format(load_weights_folder))
        
        for n in self.opt.models_to_load:
            if not (n == "encoder" or n == 'depth'):
                continue
            print("Loading {} weights...".format(n))
            path = os.path.join(load_weights_folder, "{}.pth".format(n))
            model_dict = models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            models[n].load_state_dict(model_dict)
        
        return models