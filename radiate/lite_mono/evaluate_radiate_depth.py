from __future__ import absolute_import, division, print_function
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from layers import disp_to_depth
from utils import readlines
from options import LiteMonoOptions
import datasets
import networks
import time
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from PIL import Image
from matplotlib.cm import get_cmap
from matplotlib import cm



# from thop import clever_format
# from thop import profile


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def profile_once(encoder, decoder, x):
    x_e = x[0, :, :, :].unsqueeze(0)
    x_d = encoder(x_e)
    flops_e, params_e = profile(encoder, inputs=(x_e, ), verbose=False)
    flops_d, params_d = profile(decoder, inputs=(x_d, ), verbose=False)

    flops, params = clever_format([flops_e + flops_d, params_e + params_d], "%.3f")
    flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
    flops_d, params_d = clever_format([flops_d, params_d], "%.3f")

    return flops, params, flops_e, params_e, flops_d, params_d


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def is_tensor(data):
    """Checks if data is a torch tensor."""
    return type(data) == torch.Tensor
def viz_inv_depth(inv_depth, normalizer=None, percentile=95,
                  colormap='plasma', filter_zeros=False):
    """
    Converts an inverse depth map to a colormap for visualization.

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map to be converted
    normalizer : float
        Value for inverse depth map normalization
    percentile : float
        Percentile value for automatic normalization
    colormap : str
        Colormap to be used
    filter_zeros : bool
        If True, do not consider zero values during normalization

    Returns
    -------
    colormap : np.array [H,W,3]
        Colormap generated from the inverse depth map
    """
    # If a tensor is provided, convert to numpy
    if is_tensor(inv_depth):
        # Squeeze if depth channel exists
        if len(inv_depth.shape) == 3:
            inv_depth = inv_depth.squeeze(0)
        inv_depth = inv_depth.detach().cpu().numpy()
    cm = get_cmap(colormap)
    if normalizer is None:
        normalizer = np.percentile(
            inv_depth[inv_depth > 0] if filter_zeros else inv_depth, percentile)
    inv_depth /= (normalizer + 1e-6)
    return cm(np.clip(inv_depth, 0., 1.0))[:, :, :3]


def save_pred(mono_img, pred_disp, depth_gt, out_file_path):
    
    colour_depth = viz_inv_depth(1./pred_disp, percentile=95).astype(np.float32) * 255
    colour_depth = colour_depth.astype(np.uint8)
    depth_img = Image.fromarray(colour_depth)
    depth_img = depth_img.resize((depth_gt.shape[1], depth_gt.shape[0]), Image.Resampling.LANCZOS)

    colour_depth_gt = cm.get_cmap('inferno')(depth_gt/depth_gt.max()).astype(np.float32).transpose(2, 0, 1)[:3, ...]
    colour_depth_gt = colour_depth_gt.transpose(1, 2, 0) * 255
    colour_depth_gt = colour_depth_gt.astype(np.uint8)

    img_uint8 = mono_img.mul(255).byte().cpu().numpy().transpose(1, 2, 0)
    img_pil = Image.fromarray(img_uint8)
    img_pil = img_pil.resize((depth_gt.shape[1], depth_gt.shape[0]), Image.Resampling.LANCZOS)
    

    img_overlay = np.array(img_pil)
    img_overlay[np.nonzero(depth_gt)]  =  colour_depth_gt[np.nonzero(depth_gt)]
    overlay_img = Image.fromarray(img_overlay.astype(np.uint8))

    # Concatenate the overlay with the predicted depth for comparison
    combined_img = Image.new('RGB', (img_pil.width * 2, img_pil.height))
    combined_img.paste(overlay_img, (0, 0))
    combined_img.paste(depth_img, (img_pil.width, 0))

    combined_img.save(out_file_path)                        


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    
    
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)
    
    encoder = networks.LiteMono(model=opt.model,
                                height=encoder_dict['height'],
                                width=encoder_dict['width'])
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    model_dict = encoder.state_dict()
    depth_model_dict = depth_decoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    scene_list_path = Path(opt.data_path)/'test_dep.txt'
    scene_names = [folder.strip()
                    for folder in open(scene_list_path) if not folder.strip().startswith("#")]
    
    columns = ["scene_name", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "samples"]
    results_df = pd.DataFrame(columns=columns)
    batch_size = 16
    
    
    results_dir = Path(opt.eval_out_dir)  # /args.sequence
    results_dir.mkdir(parents=True, exist_ok=True)
    for scene_name in scene_names:
        
        print("=> Processing:", scene_name)

        results_depth_dir = results_dir/scene_name/'depth'
        results_depth_dir.mkdir(parents=True, exist_ok=True)

        # all setting must use the RadiateMonoRdarDataset 
        # to ensure the number of test sample is same
        dataset = datasets.RadiateMonoRdarDataset(opt.data_path, mode='test',
                                           height = encoder_dict['height'], width = encoder_dict['width'],
                                           frame_idxs = [0, -1, 1], num_scales=4, load_depth=True,
                                           depth_gt_dir='depth', sequence=scene_name)
        nframes = len(dataset)
        dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)


        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        errors = []
        ratios = []
        
        # on-the-fly evaluating
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                
                input_color = data[("color", 0, 0)].cuda()
                output = depth_decoder(encoder(input_color))
                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                gt_depths = data["depth_gt"].cpu()[:, 0].numpy()
                
                pred_disp = F.interpolate(pred_disp, (gt_depths.shape[-2], gt_depths.shape[-1]), mode='nearest')
                pred_disps = pred_disp.detach().cpu()[:, 0].numpy()

                mono_imgs = input_color
                
                for j, (mono_img, pred_disp, gt_depth) in enumerate(zip(mono_imgs, pred_disps, gt_depths)):
                    if (i * batch_size + j) % 100 == 0:
                        out_file_path = results_depth_dir / f'{i * batch_size + j:06d}.png'
                        save_pred(mono_img, pred_disp, gt_depth, out_file_path)
                        
                        
                    pred_depth = 1 / pred_disp
                    mask = gt_depth > 0
                    
                    pred_depth = pred_depth[mask]
                    gt_depth = gt_depth[mask]
                    pred_depth *= opt.pred_depth_scale_factor

                    ratio = np.median(gt_depth) / np.median(pred_depth)
                    ratios.append(ratio)
                    pred_depth *= ratio

                    pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
                    pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
                    errors.append(compute_errors(gt_depth, pred_depth))
                    
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

            mean_errors = np.array(errors).mean(0)
            print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
            print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
            # print("\n  " + ("flops: {0}, params: {1}, flops_e: {2}, params_e:{3}, flops_d:{4}, params_d:{5}").format(flops, params, flops_e, params_e, flops_d, params_d))
            print("\n-> Done!")
            
            scene_results = [scene_name] + mean_errors.tolist() + [nframes]
            results_df.loc[len(results_df)] = scene_results  # 将结果添加到 DataFrame

    print(results_df)
    results_df.to_csv(Path('{}'.format(opt.eval_out_dir))/'results.csv', index=False)


if __name__ == "__main__":
    options = LiteMonoOptions()
    evaluate(options.parse())
