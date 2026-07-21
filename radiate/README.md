# RADIATE track — UAMTD + radar (PBCA / PBCRF)

This directory contains the **RADIATE (camera + Navtech radar)** track of UAMTD.
It is **fully self-contained**: the complete Lite-Mono training codebase is
vendored under [`lite_mono/`](./lite_mono), so you do **not** need a separate
Lite-Mono checkout or any external `PYTHONPATH`.

The novel all-weather radar modules are part of the vendored Lite-Mono copy:

| Novel module | File (under `lite_mono/`) | What it does |
|---|---|---|
| **PBCA** (point-to-BEV cross-attention) | `networks/cross_attn.py` (`PBCrossAttention`) | projects camera features into the radar BEV frustum and fuses via cross-attention |
| **Radar POV-BEV encoder** | `networks/radar_bev_encoder.py` (`RadarResnetEncoder`) | ResNet encoder for the single-channel radar BEV input |
| **PBCRF** (radar fusion) | `networks/radar_fusion.py` (`RadarFusion`, `RadarSMFusion`) | sparse multi-layer camera-radar fusion |
| RADIATE loaders | `datasets/radiate_mono_dataset.py`, `datasets/radiate_mono_radar_dataset.py`, `datasets/mono_radar_seq_dataset.py` | camera + radar (POV/BEV) + uncertainty sequence loaders |

## Dependencies

Install the RADIATE-specific extras on top of the root environment:

```bash
pip install -r lite_mono/requirements.txt
```

Key extras vs. the nuScenes track: `timm` (used by `networks/depth_encoder.py`
and `networks/mpvit.py`), `tensorboardX`, `scikit-image`, `imageio`, `tqdm`,
`six`. The LR scheduler `linear_warmup_cosine_annealing_warm_restarts_weight_decay`
is vendored inside `lite_mono/` so no extra install is needed.

## Data preparation

Prepare RADIATE with, per scene, the camera images, Navtech radar Cartesian (BEV)
and POV maps, and (optionally) depth ground truth, laid out as expected by
`datasets/mono_radar_seq_dataset.py` (`crawl_folders`). The loader reads a
`<mode>.txt` file from the dataset root listing the scene folders, e.g.:

```
${RADIATE_ROOT}/train_all.txt   # one scene folder per line
${RADIATE_ROOT}/val.txt
${RADIATE_ROOT}/test.txt
```

Expected per-scene layout (defaults; see `MonoSeqDataset.__init__` for overrides):

```
scene_xxx/
  stereo_undistorted/left/*.png        # camera images (mono_folder)
  Navtech_Cartesian/                    # radar BEV (radar_folder)
  radar_pov/                            # radar POV (radar_pov_folder)
  depth_ac/*_depth.tiff                 # depth GT (depth_gt_dir, optional)
  zed_left.txt                          # camera timestamps (stereo_timestamps)
  Navtech_Polar.txt                     # radar timestamps (radar_timestamps)
  unc/                                  # uncertainty maps (unc_folder, for --distill_unc)
```

## Training

```bash
RADIATE_ROOT=/path/to/radiate_f \
TEACHER1=/path/to/teacher_night \
TEACHER2=/path/to/teacher_rain \
[STUDENT_INIT=/path/to/uamtd_nuscenes_init] \
bash tools/train_radiate_distill_unc.sh
```

Notes:
- `--distill_weight 1.` fixes the distillation loss at `1:1`, matching the
  self-supervised photometric loss weight used in the nuScenes track.
- `--lr 0.0001 5e-6 31 0.0001 1e-5 3` is the **cosine-annealing-with-warm-restarts**
  schedule consumed by `ChainedScheduler` (depth net: max/eta_min/T_0; pose net:
  max/eta_min/T_0). It is the legitimate scheduler config, not a tuning trick.
- `--norm` enables input normalisation; `--distill_unc` weights the pixel-level
  distillation loss by the teachers' uncertainty.

## Evaluation

```bash
cd lite_mono
export PYTHONPATH=$(pwd):${PYTHONPATH}
python evaluate_radiate_depth.py \
  --data_path /path/to/radiate_f \
  --model_name distill_unc \
  --model lite-mono-8m \
  --dataset radiate_mono_radar \
  --load_weights_folder /path/to/trained_model \
  --norm
```

## License

The vendored `lite_mono/` code is released under the **Monodepth2 / Lite-Mono
license** (non-commercial); see `lite_mono/LICENSE_LITE_MONO`. The UAMTD
contributions in this track follow the repository-wide **CC BY-NC-SA 4.0** license.
