#!/usr/bin/env bash
set -euo pipefail

TEACHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "${TEACHER_DIR}/.." && pwd)}"

ROBOTWIN_LEROBOT_ROOT="${ROBOTWIN_LEROBOT_ROOT:-${WORKSPACE}/fastwam/data/robotwin2.0/robotwin2.0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TARGET="${1:-both}" # action, state, or both
FRAME_STRIDE="${ROBOTWIN_IDM_FRAME_STRIDE:-2}"
BATCH_SIZE="${ROBOTWIN_IDM_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${ROBOTWIN_IDM_EVAL_BATCH_SIZE:-32}"
NUM_WORKERS="${ROBOTWIN_IDM_NUM_WORKERS:-8}"
NUM_ITERATIONS="${ROBOTWIN_IDM_NUM_ITERATIONS:-200000}"
EVAL_INTERVAL="${ROBOTWIN_IDM_EVAL_INTERVAL:-5000}"

cd "${TEACHER_DIR}"

if [[ ! -f "${ROBOTWIN_LEROBOT_ROOT}/meta/info.json" ]]; then
  echo "ROBOTWIN_LEROBOT_ROOT must point to a LeRobot dataset root with meta/info.json." >&2
  echo "Current value: ${ROBOTWIN_LEROBOT_ROOT}" >&2
  exit 1
fi

run_idm() {
  local run_suffix="$1"
  local target_key="$2"
  local cuda_devices="$3"

  CUDA_VISIBLE_DEVICES="${cuda_devices}" accelerate launch --num_processes "${NPROC_PER_NODE}" main.py \
    +name="robotwin_lerobot_idm_${run_suffix}" \
    experiment=exp_idm \
    algorithm=idm_resnet_plus \
    dataset=robotwin_idm \
    dataset.data_root="${ROBOTWIN_LEROBOT_ROOT}" \
    dataset.input_source=lerobot \
    dataset.view_mode=stitched \
    dataset.target_key="${target_key}" \
    'dataset.image_keys=[observation.images.cam_high,observation.images.cam_left_wrist,observation.images.cam_right_wrist]' \
    dataset.frame_stride="${FRAME_STRIDE}" \
    dataset.dino_size=512 \
    dataset.do_flip=false \
    algorithm.output_dim=14 \
    experiment.training.batch_size="${BATCH_SIZE}" \
    experiment.training.eval_batch_size="${EVAL_BATCH_SIZE}" \
    experiment.training.num_workers="${NUM_WORKERS}" \
    experiment.training.num_iterations="${NUM_ITERATIONS}" \
    experiment.training.eval_interval="${EVAL_INTERVAL}"
}

case "${TARGET}" in
  action)
    run_idm "action14" "action" "${ACTION_CUDA_VISIBLE_DEVICES:-0}"
    ;;
  state|proprio)
    run_idm "state14" "observation.state" "${STATE_CUDA_VISIBLE_DEVICES:-0}"
    ;;
  both)
    run_idm "action14" "action" "${ACTION_CUDA_VISIBLE_DEVICES:-0}"
    run_idm "state14" "observation.state" "${STATE_CUDA_VISIBLE_DEVICES:-0}"
    ;;
  *)
    echo "Usage: bash scripts/train_robotwin_lerobot_idm.sh [action|state|both]" >&2
    exit 1
    ;;
esac
