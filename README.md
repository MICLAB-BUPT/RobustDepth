# UAMTD: Boosting Robustness for All-Weather Self-Supervised Depth Estimation in Autonomous Driving

**Xiaoyang Bi**, et al.

## Highlights

- **UAMTD (Uncertainty-Aware Multi-Teacher Distillation)** — a self-supervised monocular
  depth estimation framework that distills multiple *weather-expert teachers* (each
  specialized for an adverse condition such as night / rain) into a single student, using
  per-teacher uncertainty weighting so unreliable pseudo-labels are down-weighted.
- **PBCA / PBCRF radar fusion** — a point-to-BEV cross-attention (PBCA) and a sparse
  multi-layer fusion (PBCRF) that inject RADIATE Navtech radar (POV + BEV) into the camera
  branch for further all-weather robustness.
- **Two benchmarks** — nuScenes (camera-only) and RADIATE (camera + radar), with
  state-of-the-art all-weather results across clear / night / rain / fog / snow.

## Benchmark

| Benchmark | Modality | What is released here |
|---|---|---|
| **nuScenes** | camera-only | full self-contained training & evaluation pipeline (repository root) |
| **RADIATE** | camera + radar | full self-contained pipeline (Lite-Mono vendored under `radiate/lite_mono/`) |

## Results

### Table 1. RADIATE self-supervised depth estimation (absRel ↓, RMSE ↓, δ₁ ↑).

`tr.data`: `clear` = day-clear only; `all` = all-weather. **Bold** = best.
`Ours` = UAMTD applied to each backbone; `Ours*` = UAMTD + PBCRF camera-radar fusion.

| Method | tr.data | clear | night | rain | fog | snow |
|---|---|---|---|---|---|---|
| Monodepth2 | clear | 0.2120 / 6.643 / 70.35 | 0.1822 / 6.052 / 74.14 | 0.2702 / 6.383 / 61.28 | 0.3265 / 6.741 / 57.46 | 0.6945 / 10.24 / 29.37 |
| Monodepth2 | all | 0.2258 / 6.762 / 66.78 | 1.2211 / 17.13 / 14.13 | 0.2413 / 5.721 / 60.71 | 0.2965 / 3.662 / 45.15 | 0.5284 / 7.732 / 40.46 |
| **Ours** | all | **0.2104 / 6.614 / 71.03** | **0.1647 / 5.788 / 77.37** | 0.2488 / 5.406 / 61.57 | 0.2832 / 4.544 / 58.12 | 0.6812 / 10.22 / 36.42 |
| MonoVit | clear | 0.1963 / 6.394 / 76.98 | 0.1626 / 5.744 / 76.94 | 0.2332 / 5.889 / 67.41 | 0.2884 / 4.586 / 54.80 | 0.5821 / 9.362 / 55.30 |
| MonoVit | all | 0.2132 / 6.182 / 69.82 | 0.9185 / 12.62 / 16.55 | 0.2274 / 5.073 / 67.05 | 0.2928 / 3.576 / 45.74 | 0.5430 / 7.572 / 33.17 |
| **Ours** | all | **0.1843 / 5.935 / 77.58** | **0.1598 / 5.735 / 78.53** | **0.2116 / 5.162 / 69.71** | **0.2576 / 3.671 / 59.05** | 0.5577 / 9.124 / 59.27 |
| ManyDepth2 | clear | 0.1780 / 5.797 / 77.70 | 0.1500 / 5.623 / 79.40 | 0.2560 / 5.712 / 62.90 | 0.2410 / 3.277 / 63.60 | 0.4600 / 7.333 / 55.90 |
| ManyDepth2 | all | 0.1890 / 5.798 / 77.30 | 0.2080 / 6.338 / 66.30 | 0.2590 / 6.121 / 65.10 | 0.3940 / 4.846 / 59.40 | 0.4780 / 7.068 / 47.60 |
| **Ours** | all | **0.1690 / 5.417 / 78.30** | **0.1480 / 5.755 / 79.70** | **0.2330 / 5.331 / 65.00** | **0.2190 / 2.992 / 66.40** | **0.4530 / 7.513 / 57.00** |
| Lite-Mono | clear | 0.1802 / 6.271 / 79.27 | 0.1630 / 5.870 / 78.15 | 0.2616 / 6.475 / 64.45 | 0.3318 / 6.029 / 60.15 | 0.6426 / 11.14 / 52.46 |
| Lite-Mono | all | 0.2133 / 6.450 / 71.67 | 0.9498 / 13.05 / 15.71 | 0.2541 / 6.191 / 65.45 | 0.3802 / 4.356 / 39.14 | 0.7159 / 10.06 / 28.69 |
| **Ours** | all | **0.1706 / 5.810 / 80.65** | **0.1462 / 5.628 / 81.06** | **0.2158 / 5.011 / 69.88** | **0.2514 / 3.706 / 63.31** | **0.5165 / 8.562 / 58.70** |
| Ours + CaFNet | all | 0.1749 / 5.916 / 79.86 | 0.1475 / 5.620 / 80.61 | 0.2194 / 5.102 / 69.09 | 0.2490 / 3.663 / 63.17 | 0.5597 / 9.165 / 57.23 |
| Ours + R4dyn | all | 0.1660 / 5.614 / 80.67 | 0.1478 / 5.722 / 80.10 | 0.2031 / 4.560 / 71.43 | 0.2414 / 3.453 / 63.39 | 0.5115 / 8.371 / 58.37 |
| **Ours\*** | all | **0.1630 / 5.534 / 80.76** | 0.1482 / 5.805 / 79.96 | **0.1991 / 4.483 / 72.12** | **0.2400 / 3.391 / 63.19** | **0.5025 / 8.120 / 57.85** |

## Getting Started

```bash
conda create -n uamtd python=3.8
conda activate uamtd
pip install -r requirements.txt
# or build the provided Docker image
make docker-build
```

Key dependencies: PyTorch 1.13 + CUDA 11.3, pytorch-lightning, fvcore, nuScenes-devkit,
opencv, tensorboard (see `requirements.txt` / `requirements_w_version.txt`).

## Preparing Datasets

### nuScenes (camera-only track)

Download [nuScenes](https://www.nuscenes.org/) and symlink it next to the repo root:

```bash
ln -s /path/to/your/nuscenes ../nuscenes
```

(Optional, only for training the weather-expert teachers) ForkGAN day→night / day→rain
translated images:

```bash
ln -s /path/to/nuScenes_translated ../nuScenes_translated
# expected layout: ../nuScenes_translated/night , ../nuScenes_translated/rain
```

If you skip synthetic translation, train the teachers with the
`DAY_*_TRANSLATION_*` augmentation probability set to `0.0`.

### RADIATE (camera + radar track)

Prepare the RADIATE dataset with camera images, Navtech radar Cartesian (BEV) and POV
maps, and (optionally) depth ground truth, laid out per scene as expected by
`radiate/datasets/mono_radar_seq_dataset.py` (`crawl_folders`). Point the training script
at it via `RADIATE_ROOT` (see `radiate/tools/train_radiate_distill_unc.sh`).

## Pretrained weights

This release contains **training scripts only**; no checkpoints are hosted. Reproduce all
models by following the pipeline below. Each stage writes its checkpoint under
`log/checkpoints/` and later stages load them through the relative paths defined in the
config files.

## Training & Evaluation

Run from the repository root with `export PYTHONPATH=$(pwd)`. Stages must be executed in
order; each stage saves its checkpoint to `log/checkpoints/`, which the next stage consumes.
After a stage finishes, symlink or copy its best checkpoint to the filename expected by the
downstream config (use `tools/select_best_teacher.py`, or simply take `last.ckpt`).

### nuScenes track (self-contained)

1. **Strong day-clear init model (md4allDDa)**
   ```bash
   python train.py --config config/train_uncdistill_dd.yaml
   # link output -> log/checkpoints/md4allDDa_nuscenes.ckpt
   ```

2. **Weather-expert teachers (night / rain)**
   ```bash
   python train.py --config config/train_teacher_night_syn.yaml
   python train.py --config config/train_teacher_rain_syn.yaml

   # pick the best-epoch ckpt and link to the fixed names used by the student config
   python tools/select_best_teacher.py \
       --ckpt_dir log/checkpoints/train-teacher-night-syn \
       --tb_dir   log/training/train-teacher-night-syn/lightning_logs \
       --out      log/checkpoints/teacher-night-syn.ckpt

   python tools/select_best_teacher.py \
       --ckpt_dir log/checkpoints/train-teacher-rain-syn \
       --tb_dir   log/training/train-teacher-rain-syn/lightning_logs \
       --metric   'metrics_day-rain/everything/abs_rel_pp' \
       --out      log/checkpoints/teacher-rain-syn.ckpt
   ```

3. **UAMTD student (final model)**
   ```bash
   python train.py --config config/train_uncdistill_strongteacher.yaml
   # -> log/checkpoints/train-unc-distll-strongteacher/<epoch>.ckpt
   ```

4. **Evaluation**
   ```bash
   python evaluate_depth.py --config config/eval_unc_distill.yaml
   # condition-wise metrics (day-clear / day-rain / night-clear / night-rain)
   # are printed and saved under results/
   ```

### RADIATE track (self-contained — Lite-Mono vendored)

The RADIATE track is **fully self-contained**: the complete Lite-Mono training
codebase is vendored under `radiate/lite_mono/`, so no external Lite-Mono checkout
or `PYTHONPATH` is needed. See `radiate/README.md` for data preparation, the novel
radar modules (PBCA / PBCRF), dependencies, and train/eval commands.

```bash
RADIATE_ROOT=/path/to/radiate_f \
TEACHER1=/path/to/teacher_night \
TEACHER2=/path/to/teacher_rain \
bash radiate/tools/train_radiate_distill_unc.sh
```

The distillation loss weight is fixed at `1:1` (`--distill_weight 1.`), matching the
self-supervised photometric loss weight used in the nuScenes track.

## Repository structure

```
uamtd_release/
├── config/            # training & evaluation YAML configs (relative paths)
├── data/              # nuScenes / RobotCar dataset loaders
├── losses/            # UAMTD: SupervisedLoss (unc_sim) + TotalLoss + photometric/smoothness/velocity
├── models/            # depth_net, pose_net, uncertainty_miner, uncertainty_block, md2 backbone
├── utils/             # helpers
├── evaluation/        # metric computation
├── visualization/     # depth / uncertainty visualization
├── tools/             # select_best_teacher.py, measure_uncertainty_branch.py, sanity_supervised_loss.py
├── radiate/           # RADIATE track (self-contained): camera + radar
│   ├── lite_mono/     # vendored Lite-Mono training codebase (backbone + novel modules)
│   │   ├── networks/  # cross_attn.py (PBCA), radar_bev_encoder.py, radar_fusion.py (PBCRF)
│   │   ├── datasets/  # RADIATE camera+radar loaders
│   │   ├── evaluate_radiate_depth.py
│   │   └── requirements.txt
│   ├── tools/         # train_radiate_distill_unc.sh
│   └── README.md
├── train.py           # training entry point (nuScenes track)
├── trainer.py         # PyTorch Lightning module (Md4All)
├── evaluate_depth.py  # evaluation entry point (nuScenes track)
├── requirements.txt   # dependencies
├── Dockerfile         # training/eval environment
├── Makefile           # docker convenience targets
├── LICENSE            # CC BY-NC-SA 4.0 (inherited from md4all)
└── README.md
```

## License & Acknowledgement

Released under **CC BY-NC-SA 4.0**, the same license as
[md4all](https://github.com/...). This code adapts the md4all (ICCV 2023) framework; please
also cite md4all when you use this repository. The RADIATE track reuses novel radar modules
originally developed within the Lite-Mono framework.

## Citation

```bibtex
@article{bi2025uamtd,
  title     = {Boosting Robustness for All-Weather Self-Supervised Depth Estimation in Autonomous Driving},
  author    = {Bi, Xiaoyang and others},
  journal   = {IEEE Transactions on Image Processing},
  year      = {2025}
}
```
