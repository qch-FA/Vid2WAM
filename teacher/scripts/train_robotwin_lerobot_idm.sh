#!/usr/bin/env bash
set -euo pipefail

TEACHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "${TEACHER_DIR}/.." && pwd)}"

ROBOTWIN_LEROBOT_ROOT="${ROBOTWIN_LEROBOT_ROOT:-${WORKSPACE}/data/robotwin}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
IDM_TARGET="${ROBOTWIN_IDM_TARGET:-action}" # action or state
FRAME_STRIDE="${ROBOTWIN_IDM_FRAME_STRIDE:-1}"
DINO_SIZE="${ROBOTWIN_IDM_DINO_SIZE:-null}"
BATCH_SIZE="${ROBOTWIN_IDM_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${ROBOTWIN_IDM_EVAL_BATCH_SIZE:-32}"
NUM_WORKERS="${ROBOTWIN_IDM_NUM_WORKERS:-8}"
NUM_ITERATIONS="${ROBOTWIN_IDM_NUM_ITERATIONS:-200000}"
EVAL_INTERVAL="${ROBOTWIN_IDM_EVAL_INTERVAL:-5000}"
WARMUP_STEPS="${ROBOTWIN_IDM_WARMUP_STEPS:-5000}"

EXTRA_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    idm_target=*|target=*)
      IDM_TARGET="${arg#*=}"
      ;;
    *)
      EXTRA_ARGS+=("${arg}")
      ;;
  esac
done

cd "${TEACHER_DIR}"

if [[ ! -f "${ROBOTWIN_LEROBOT_ROOT}/meta/info.json" ]]; then
  echo "ROBOTWIN_LEROBOT_ROOT must point to a LeRobot dataset root with meta/info.json." >&2
  echo "Current value: ${ROBOTWIN_LEROBOT_ROOT}" >&2
  exit 1
fi

case "${IDM_TARGET}" in
  action)
    RUN_SUFFIX="action14"
    TARGET_KEY="action"
    ;;
  state|proprio)
    RUN_SUFFIX="state14"
    TARGET_KEY="observation.state"
    ;;
  *)
    echo "Usage: bash scripts/train_robotwin_lerobot_idm.sh idm_target=<action|state> [hydra_overrides...]" >&2
    exit 1
    ;;
esac

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NPROC_PER_NODE - 1)))}" python -m torch.distributed.run --standalone --nproc-per-node "${NPROC_PER_NODE}" main.py \
  +name="robotwin_aloha_joint_idm_${RUN_SUFFIX}" \
  experiment=exp_idm \
  algorithm=idm_resnet_plus \
  dataset=robotwin_idm \
  dataset.data_root="${ROBOTWIN_LEROBOT_ROOT}" \
  dataset.input_source=lerobot \
  dataset.view_mode=stitched \
  ++dataset.stitch_resize=false \
  ++dataset.target_key="${TARGET_KEY}" \
  ++dataset.target_space=joint \
  ++dataset.expected_robot_type=aloha \
  '++dataset.image_keys=[observation.images.cam_high,observation.images.cam_left_wrist,observation.images.cam_right_wrist]' \
  dataset.frame_stride="${FRAME_STRIDE}" \
  dataset.dino_size="${DINO_SIZE}" \
  dataset.do_flip=false \
  algorithm.output_dim=14 \
  algorithm.normalization_stats=auto \
  experiment.training.use_normalization=true \
  experiment.training.batch_size="${BATCH_SIZE}" \
  experiment.training.eval_batch_size="${EVAL_BATCH_SIZE}" \
  experiment.training.num_workers="${NUM_WORKERS}" \
  experiment.training.num_iterations="${NUM_ITERATIONS}" \
  experiment.training.eval_interval="${EVAL_INTERVAL}" \
  experiment.training.scheduler=warmup_cosine \
  experiment.training.warmup_steps="${WARMUP_STEPS}" \
  experiment.training.warmup_ratio=0.0 \
  experiment.training.min_lr_ratio=0.0 \
  "${EXTRA_ARGS[@]}"
