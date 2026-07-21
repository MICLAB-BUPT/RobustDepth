#!/bin/bash
# Train UAMTD + radar (PBCA / PBCRF) on the RADIATE dataset.
#
# This RADIATE track is FULLY SELF-CONTAINED: the complete Lite-Mono training
# codebase is vendored under `radiate/lite_mono/`, so no external PYTHONPATH or
# Lite-Mono checkout is required.
#
# Before running:
#   1. Prepare the RADIATE dataset and point RADIATE_ROOT to it (see radiate/README.md).
#      The loader expects, per scene, a `<mode>.txt` file listing the scene folders
#      (e.g. train_all.txt / val.txt / test.txt).
#   2. Obtain the two weather-expert teacher checkpoints and point TEACHER1 / TEACHER2
#      to their weight folders (these are produced by the nuScenes track, or trained
#      separately). The student is optionally initialised from a nuScenes UAMTD model
#      via STUDENT_INIT.
#
# Usage:
#   RADIATE_ROOT=/path/to/radiate_f \
#   TEACHER1=/path/to/teacher_night \
#   TEACHER2=/path/to/teacher_rain \
#   [STUDENT_INIT=/path/to/uamtd_nuscenes_init] \
#   bash tools/train_radiate_distill_unc.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LITE_MONO_DIR="${SCRIPT_DIR}/../lite_mono"

export PYTHONPATH="${LITE_MONO_DIR}:${PYTHONPATH}"

: "${RADIATE_ROOT:?Set RADIATE_ROOT to your prepared RADIATE directory}"
: "${TEACHER1:?Set TEACHER1 to weather-expert teacher 1 weights folder}"
: "${TEACHER2:?Set TEACHER2 to weather-expert teacher 2 weights folder}"

STUDENT_INIT_FLAG=""
if [ -n "${STUDENT_INIT}" ]; then
  STUDENT_INIT_FLAG="--load_weights_folder ${STUDENT_INIT}"
fi

python "${LITE_MONO_DIR}/train.py" \
  --data_path "${RADIATE_ROOT}" \
  --model_name distill_unc \
  --model lite-mono-8m \
  --num_epochs 30 \
  --batch_size 12 \
  --train_mode train_all \
  --dataset radiate_mono_radar \
  --distill_weight 1. \
  --distill_unc \
  ${STUDENT_INIT_FLAG} \
  --t1_path "${TEACHER1}" \
  --t2_path "${TEACHER2}" \
  --norm \
  --lr 0.0001 5e-6 31 0.0001 1e-5 3
