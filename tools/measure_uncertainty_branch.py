#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure the per-teacher cost of the uncertainty-mining branch (UncertaintyMiner).

Reports, for pseudo_num (= #teachers) in {1,2,3,4}:
  * #Params (CPU-only, always reliable)
  * training-time forward latency on GPU (dummy inputs; needs CUDA)
and prints the incremental cost per additional teacher branch, to fill Tab.R2 /
hAEi-Q2 in the rebuttal.

Run (GPU machine):
  cd <UAMTD_REPO_ROOT> && export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  CUDA_VISIBLE_DEVICES=0 python tools/measure_uncertainty_branch.py
  # params-only (no GPU needed):
  python tools/measure_uncertainty_branch.py --params_only

NOTE on dummy shapes (derived from UncertaintyMiner.forward):
  - x[i]            : teacher inverse-depth, [1,1,H,W]; pseudo_encoder is Conv(1->32,k4,s4) -> H/4
  - features[0]     : student feat (64ch) at [1,64,H/2,W/2]  (forward does interpolate x0.5 -> H/4)
  - features[1]     : student feat (256ch) at [1,256,H/16,W/16] (forward does upsample x4 -> H/4)
  With H=320,W=576 (md4all nuScenes): x=[1,1,320,576], f0=[1,64,160,288], f1=[1,256,20,36].
  If your backbone differs, adjust --H/--W and the feature channels below.
"""
import argparse
import time

import torch

from models.uncertainty_miner import UncertaintyMiner

PSEUDO_INPUT_DIM = 32


def count_params(n):
    miner = UncertaintyMiner(pseudo_num=n, pseudo_input_dim=PSEUDO_INPUT_DIM)
    return sum(p.numel() for p in miner.parameters()), miner


def make_dummy(n, H, W, device):
    x = [torch.randn(1, 1, H, W, device=device) for _ in range(n)]
    # features[0] -> 64ch at H/2,W/2 ; features[1] -> 256ch at H/16,W/16
    f0 = torch.randn(1, 64, H // 2, W // 2, device=device)
    f1 = torch.randn(1, 256, H // 16, W // 16, device=device)
    return x, [f0, f1]


@torch.no_grad()
def measure_latency(miner, n, H, W, device, warmup=30, iters=100):
    miner.eval().to(device)
    for _ in range(warmup):
        x, feats = make_dummy(n, H, W, device)
        miner(x, feats)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        x, feats = make_dummy(n, H, W, device)
        if device.type == "cuda":
            torch.cuda.synchronize(); t0 = time.time()
            miner(x, feats); torch.cuda.synchronize()
        else:
            t0 = time.time(); miner(x, feats)
        ts.append((time.time() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2], min(ts)  # median, min (ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nums", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--H", type=int, default=320)
    ap.add_argument("--W", type=int, default=576)
    ap.add_argument("--params_only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.params_only) else "cpu")
    print(f"device={device}, dummy x=[1,1,{args.H},{args.W}]")
    print(f"{'#teachers':>9} | {'#Params(M)':>10} | {'Latency med/min (ms)':>22}")
    print("-" * 50)

    rows = []
    for n in args.nums:
        params, miner = count_params(n)
        lat = None
        if not args.params_only:
            try:
                lat = measure_latency(miner, n, args.H, args.W, device)
            except Exception as e:
                print(f"  [warn] latency failed for n={n}: {e!r}")
        rows.append((n, params, lat))
        lat_s = f"{lat[0]:.2f} / {lat[1]:.2f}" if lat else "n/a"
        print(f"{n:>9} | {params / 1e6:>10.4f} | {lat_s:>22}")

    print("-" * 50)
    print("Incremental cost per additional teacher branch:")
    for i in range(1, len(rows)):
        n0, p0, l0 = rows[i - 1]
        n1, p1, l1 = rows[i]
        dp = (p1 - p0) / 1e6
        dl = f"{l1[0] - l0[0]:+.2f} ms" if (l0 and l1) else "n/a"
        print(f"  {n0}->{n1}: +{dp:.4f}M params, {dl}")


if __name__ == "__main__":
    main()
