#!/usr/bin/env bash
set -euo pipefail

TEACHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "${TEACHER_DIR}/.." && pwd)}"

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${WORKSPACE}/data/robotwin_low_20}"
ROBOTWIN_METADATA="${ROBOTWIN_METADATA:-robotwin_videos.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${TEACHER_DIR}/outputs/robotwin_low20_sft}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TEXT_ENCODER_CKPT="${TEXT_ENCODER_CKPT:-${TEACHER_DIR}/data/ckpts/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth}"
TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-${TEACHER_DIR}/data/ckpts/Wan2.1-I2V-14B-480P/google/umt5-xxl}"
TUNED_CKPT_PATH="${TUNED_CKPT_PATH:-${TEACHER_DIR}/data/ckpts/checkpoints/lvp_14B.ckpt}"

DEFAULT_VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-${DEFAULT_VISIBLE_GPUS}}"
TEXT_CUDA_VISIBLE_DEVICES="${TEXT_CUDA_VISIBLE_DEVICES:-${DEFAULT_VISIBLE_GPUS%%,*}}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-65}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-13742}"
RUN_NAME="${RUN_NAME:-robotwin_low20_teacher_i2v}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -r -a visible_gpus <<< "${TRAIN_CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#visible_gpus[@]}"
fi

cd "${TEACHER_DIR}"

for path in \
  "${ROBOTWIN_ROOT}/meta/info.json" \
  "${ROBOTWIN_ROOT}/meta/episodes.jsonl" \
  "${TEXT_ENCODER_CKPT}" \
  "${TEXT_ENCODER_NAME}" \
  "${TUNED_CKPT_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" - <<'PY'
import importlib

for module in ("av", "cv2", "decord", "hydra", "lightning", "torch"):
    importlib.import_module(module)
print("Teacher SFT Python dependencies are available.")
PY

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  "${PYTHON_BIN}" datasets/generate_robotwin_lerobot_csv.py \
    --root-dir "${ROBOTWIN_ROOT}" \
    --output-csv "${ROBOTWIN_ROOT}/${ROBOTWIN_METADATA}"
fi

if [[ ! -f "${ROBOTWIN_ROOT}/${ROBOTWIN_METADATA}" ]]; then
  echo "Prepared metadata does not exist: ${ROBOTWIN_ROOT}/${ROBOTWIN_METADATA}" >&2
  exit 1
fi

if [[ "${SKIP_PROMPT_EMBED:-0}" != "1" ]]; then
  CUDA_VISIBLE_DEVICES="${TEXT_CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" main.py \
    +name=process_robotwin_low20_embeds \
    experiment=process_data \
    algorithm=wan_i2v \
    dataset=robotwin_lerobot \
    dataset.data_root="${ROBOTWIN_ROOT}" \
    dataset.metadata_path="${ROBOTWIN_METADATA}" \
    algorithm.text_encoder.ckpt_path="${TEXT_ENCODER_CKPT}" \
    algorithm.text_encoder.name="${TEXT_ENCODER_NAME}" \
    'experiment.tasks=[cache_prompt_embed]' \
    experiment.new_data_root="${ROBOTWIN_ROOT}" \
    experiment.cache_prompt_embed.batch_size="${PROMPT_BATCH_SIZE}"
fi

if [[ "${SKIP_SFT:-0}" != "1" ]]; then
  CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes="${NUM_NODES}" \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    main.py \
    +name="${RUN_NAME}" \
    experiment=exp_video \
    algorithm=wan_i2v \
    dataset=robotwin_lerobot \
    dataset.data_root="${ROBOTWIN_ROOT}" \
    dataset.metadata_path="${ROBOTWIN_METADATA}" \
    experiment.num_nodes="${NUM_NODES}" \
    algorithm.lang_guidance=0 \
    algorithm.hist_guidance=0 \
    experiment.training.batch_size="${TRAIN_BATCH_SIZE}" \
    experiment.training.max_epochs="${MAX_EPOCHS}" \
    experiment.training.max_steps=-1 \
    algorithm.gradient_checkpointing_rate=1.0 \
    algorithm.model.tuned_ckpt_path="${TUNED_CKPT_PATH}" \
    algorithm.text_encoder.name="${TEXT_ENCODER_NAME}" \
    hydra.run.dir="${OUTPUT_DIR}" \
    "$@"
fi
