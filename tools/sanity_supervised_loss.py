"""CPU sanity check for the refactored SupervisedLoss (both modes).

Run from the repo root (needs torch + repo deps, CPU only):
    python tools/sanity_supervised_loss.py config/train_teacher_night_syn.yaml 1
    python tools/sanity_supervised_loss.py config/train_uncdistill_strongteacher.yaml 2

It builds a reference DepthNet to produce shape-consistent student outputs, instantiates
SupervisedLoss with is_train=False (so no teacher ckpt files are required), runs forward,
and checks the loss is a finite scalar.
"""
import sys
import torch

from config.config import get_cfg
from models.depth_net import DepthNet
from losses.SupervisedLoss import SupervisedLoss


def build_cfg(config_path, teacher_num):
    class _Args:
        config = config_path
        opts = []
    cfg = get_cfg(_Args())
    cfg.defrost()
    cfg.MODEL.DEPTH.ENCODER.PRETRAINED = False   # avoid imagenet download in sanity
    cfg.MODEL.POSE.ENCODER.PRETRAINED = False
    cfg.LOSS.SUPERVISED.WEIGHT = 1.0
    cfg.LOSS.SUPERVISED.TEACHER_NUM = teacher_num
    cfg.freeze()
    return cfg


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/train_teacher_night_syn.yaml"
    teacher_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    cfg = build_cfg(config_path, teacher_num)
    print(f"[cfg] config={config_path} TEACHER_NUM={cfg.LOSS.SUPERVISED.TEACHER_NUM} "
          f"METHOD={cfg.LOSS.SUPERVISED.METHOD} scales={list(cfg.DATASET.SCALES)}")

    torch.manual_seed(0)
    B, H, W = 2, 320, 576
    device = "cpu"

    # Reference student to get shape-consistent disps + encoder features
    ref = DepthNet(cfg).to(device).eval()
    color0 = torch.rand(B, 3, H, W, device=device)
    with torch.no_grad():
        ref_out, ref_feats = ref(color0, ["day-clear"] * B, add_noise=False)

    outputs = {("disp", 0, s): ref_out[("disp", 0, s)].clone().requires_grad_(True)
               for s in cfg.DATASET.SCALES}
    outputs["student_feats"] = [f.clone() for f in ref_feats]
    print("[shapes] student_feats:", [tuple(f.shape) for f in outputs["student_feats"]])
    print("[shapes] disp scales  :", {s: tuple(outputs[("disp", 0, s)].shape) for s in cfg.DATASET.SCALES})

    inputs = {
        ("color", 0): color0,
        "weather": ["day-clear"] * B,
        "weather_depth": ["day-clear"] * B,   # kept for backward-compat (no longer used)
    }

    loss_mod = SupervisedLoss(cfg, is_train=False).to(device)
    loss = loss_mod(inputs, outputs)

    print(f"\n[result] loss={loss.item():.6f} shape={tuple(loss.shape)} "
          f"requires_grad={loss.requires_grad}")
    assert loss.dim() == 0, "loss must be a scalar"
    assert torch.isfinite(loss), "loss must be finite"
    loss.backward()
    grad_ok = outputs[("disp", 0, 0)].grad is not None
    print("[result] backward OK, student disp grad:", grad_ok)
    print("\nSANITY PASSED for TEACHER_NUM =", cfg.LOSS.SUPERVISED.TEACHER_NUM)


if __name__ == "__main__":
    main()
