"""Auto-select the best-epoch expert checkpoint and symlink it to a fixed name.

The expert configs save one checkpoint per epoch (TOP_K=-1) named like
    <EXPERIMENT_NAME>-epoch=XX.ckpt
plus a `last.ckpt`. This tool parses the TensorBoard logs to find the epoch with the
best (min) monitored metric, locates the matching checkpoint, and creates/updates a
symlink (or copy) at a stable path used by the UAMTD config.

Examples (run from repo root):
  # night expert (monitors night abs_rel by default)
  python tools/select_best_teacher.py \
      --ckpt_dir log/checkpoints/train-teacher-night-syn \
      --tb_dir   log/training/train-teacher-night-syn/lightning_logs \
      --out      log/checkpoints/teacher-night-syn.ckpt

  # rain expert (pick a rain metric; use --list to see available tags first)
  python tools/select_best_teacher.py \
      --ckpt_dir log/checkpoints/train-teacher-rain-syn \
      --tb_dir   log/training/train-teacher-rain-syn/lightning_logs \
      --metric   'metrics_day-rain/everything/abs_rel_pp' \
      --out      log/checkpoints/teacher-rain-syn.ckpt
"""
import argparse
import glob
import os
import re
import shutil
import sys


def find_event_files(tb_dir):
    files = glob.glob(os.path.join(tb_dir, "**", "events.out.tfevents.*"), recursive=True)
    return sorted(files)


def load_scalars(tb_dir):
    """Return {tag: [(step, value), ...]} by reading all event files under tb_dir."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:
        print("[ERROR] tensorboard not available to parse logs:", repr(e))
        return None
    event_files = find_event_files(tb_dir)
    if not event_files:
        print(f"[ERROR] no event files found under {tb_dir}")
        return None
    scalars = {}
    for ef in event_files:
        acc = EventAccumulator(ef)
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            scalars.setdefault(tag, [])
            scalars[tag].extend((e.step, e.value) for e in acc.Scalars(tag))
    for tag in scalars:
        scalars[tag].sort(key=lambda x: x[0])
    return scalars


def build_step_to_epoch(scalars):
    """PL logs an 'epoch' scalar; map step -> epoch."""
    mapping = {}
    for tag in ("epoch", "epoch_step", "epoch/epoch"):
        if tag in scalars:
            for step, val in scalars[tag]:
                mapping[step] = int(round(val))
            break
    return mapping


def parse_epoch_from_filename(path):
    name = os.path.basename(path)
    m = re.search(r"epoch[=_-]?(\d+)", name)
    if m:
        return int(m.group(1))
    nums = re.findall(r"(\d+)", name)
    return int(nums[-1]) if nums else None


def main():
    ap = argparse.ArgumentParser(description="Select best-epoch expert ckpt and symlink it.")
    ap.add_argument("--ckpt_dir", required=True, help="experiment checkpoint folder (contains *-epoch=XX.ckpt)")
    ap.add_argument("--tb_dir", required=True, help="lightning_logs dir of the experiment")
    ap.add_argument("--metric", default="metrics_night/everything/abs_rel_pp",
                    help="monitored metric tag (default: night abs_rel; use --list to inspect)")
    ap.add_argument("--mode", default="min", choices=["min", "max"], help="optimize direction")
    ap.add_argument("--out", required=True, help="output stable path (symlink/copy target)")
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    ap.add_argument("--list", action="store_true", help="only list available scalar tags and exit")
    args = ap.parse_args()

    scalars = load_scalars(args.tb_dir)

    # All per-epoch checkpoints (exclude last.ckpt)
    ckpts = [p for p in glob.glob(os.path.join(args.ckpt_dir, "*.ckpt"))
             if os.path.basename(p) != "last.ckpt"]
    epoch2ckpt = {}
    for p in ckpts:
        ep = parse_epoch_from_filename(p)
        if ep is not None:
            epoch2ckpt[ep] = p

    if scalars is not None and args.list:
        print("Available scalar tags:")
        for t in sorted(scalars.keys()):
            print("   ", t)
        return

    best_ckpt = None
    if scalars is not None and args.metric in scalars and epoch2ckpt:
        step2epoch = build_step_to_epoch(scalars)
        series = scalars[args.metric]
        # choose best step
        best_step, best_val = (min(series, key=lambda x: x[1]) if args.mode == "min"
                               else max(series, key=lambda x: x[1]))
        best_epoch = step2epoch.get(best_step)
        if best_epoch is None:
            # fall back: rank epochs by averaging metric values that fall in each epoch order
            # (if no 'epoch' scalar) -> use index order
            ordered = sorted(series, key=lambda x: x[1], reverse=(args.mode == "max"))
            # try each candidate step until one maps to an available epoch ckpt
            for st, _ in ordered:
                ep = step2epoch.get(st)
                if ep in epoch2ckpt:
                    best_epoch = ep
                    break
        print(f"[metric] {args.metric} best={best_val:.5f} at step={best_step} epoch={best_epoch}")
        if best_epoch in epoch2ckpt:
            best_ckpt = epoch2ckpt[best_epoch]
        else:
            print(f"[warn] no checkpoint found for epoch {best_epoch}; "
                  f"available epochs={sorted(epoch2ckpt)}")

    # Fallbacks
    if best_ckpt is None:
        last = os.path.join(args.ckpt_dir, "last.ckpt")
        if scalars is None or args.metric not in (scalars or {}):
            print(f"[warn] metric '{args.metric}' unavailable; falling back to last.ckpt")
        if os.path.isfile(last):
            best_ckpt = last
        elif epoch2ckpt:
            best_ckpt = epoch2ckpt[max(epoch2ckpt)]
            print(f"[warn] last.ckpt missing; using highest-epoch ckpt: {best_ckpt}")
        else:
            print("[ERROR] no checkpoints found at all."); sys.exit(1)

    best_ckpt = os.path.abspath(best_ckpt)
    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if os.path.islink(out) or os.path.exists(out):
        os.remove(out)
    if args.copy:
        shutil.copy2(best_ckpt, out)
        print(f"[done] copied {best_ckpt} -> {out}")
    else:
        os.symlink(best_ckpt, out)
        print(f"[done] symlinked {out} -> {best_ckpt}")


if __name__ == "__main__":
    main()
