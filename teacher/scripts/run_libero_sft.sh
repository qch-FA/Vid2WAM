#!/usr/bin/env bash
set -euo pipefail

TEACHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "${TEACHER_DIR}/.." && pwd)}"

LIBERO_RAW_ROOT="${LIBERO_RAW_ROOT:-${WORKSPACE}/data/libero_mujoco3.3.2}"
LIBERO_TEACHER_ROOT="${LIBERO_TEACHER_ROOT:-${WORKSPACE}/data/libero_teacher_sft_2cam224}"
LIBERO_METADATA="${LIBERO_METADATA:-libero_videos.csv}"

TEXT_ENCODER_CKPT="${TEXT_ENCODER_CKPT:-${TEACHER_DIR}/data/ckpts/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth}"
TUNED_CKPT_PATH="${TUNED_CKPT_PATH:-${TEACHER_DIR}/data/ckpts/checkpoints/lvp_14B.ckpt}"
TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-${TEACHER_DIR}/data/ckpts/Wan2.1-I2V-14B-480P/google/umt5-xxl}"
RUN_NAME="${RUN_NAME:-libero_final_i2v}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
NUM_NODES="${NUM_NODES:-1}"

cd "${TEACHER_DIR}"
mkdir -p "${LIBERO_TEACHER_ROOT}"

if [[ ! -d "${LIBERO_RAW_ROOT}" ]]; then
  echo "LIBERO_RAW_ROOT does not exist: ${LIBERO_RAW_ROOT}" >&2
  echo "Expected extracted LeRobot dataset folders under LIBERO_RAW_ROOT." >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  libero_dataset_names=("$@")
else
  read -r -a libero_dataset_names <<< "${LIBERO_DATASETS:-libero_goal_no_noops_lerobot}"
fi

dataset_args=()
for dataset_name in "${libero_dataset_names[@]}"; do
  if [[ "${dataset_name}" = /* || "${dataset_name}" == *"/"* ]]; then
    dataset_dir="${dataset_name}"
  else
    dataset_dir="${LIBERO_RAW_ROOT}/${dataset_name}"
  fi

  if [[ -d "${dataset_dir}" ]]; then
    dataset_args+=(--dataset-dir "${dataset_dir}")
  else
    echo "LIBERO dataset directory does not exist: ${dataset_dir}" >&2
    echo "Pass dataset names explicitly, for example:" >&2
    echo "  bash scripts/run_libero_sft.sh libero_goal_no_noops_lerobot" >&2
    exit 1
  fi
done

if [[ ${#dataset_args[@]} -eq 0 ]]; then
  echo "No LIBERO datasets specified." >&2
  exit 1
fi

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  python datasets/prepare_libero_sft_data.py \
    "${dataset_args[@]}" \
    --output-root "${LIBERO_TEACHER_ROOT}" \
    --metadata-path "${LIBERO_TEACHER_ROOT}/${LIBERO_METADATA}" \
    ${PREPARE_EXTRA_ARGS:-}
fi

if [[ "${SKIP_PROMPT_EMBED:-0}" != "1" ]]; then
  CUDA_VISIBLE_DEVICES="${TEXT_CUDA_VISIBLE_DEVICES:-0}" python main.py \
    +name=process_libero_embeds \
    experiment=process_data \
    dataset=libero \
    dataset.data_root="${LIBERO_TEACHER_ROOT}" \
    dataset.metadata_path="${LIBERO_METADATA}" \
    algorithm=wan_i2v \
    algorithm.text_encoder.ckpt_path="${TEXT_ENCODER_CKPT}" \
    algorithm.text_encoder.name="${TEXT_ENCODER_NAME}" \
    'experiment.tasks=[cache_prompt_embed]' \
    experiment.new_data_root="${LIBERO_TEACHER_ROOT}" \
    experiment.cache_prompt_embed.batch_size="${PROMPT_BATCH_SIZE}"
fi

if [[ "${SKIP_SFT:-0}" != "1" ]]; then
  python -m torch.distributed.run --standalone --nproc-per-node="${NPROC_PER_NODE:-8}" main.py \
    +name="${RUN_NAME}" \
    experiment=exp_video \
    algorithm=wan_i2v \
    dataset=libero \
    dataset.data_root="${LIBERO_TEACHER_ROOT}" \
    dataset.metadata_path="${LIBERO_METADATA}" \
    experiment.num_nodes="${NUM_NODES}" \
    algorithm.text_encoder.name="${TEXT_ENCODER_NAME}" \
    algorithm.lang_guidance=0 \
    algorithm.hist_guidance=0 \
    experiment.training.batch_size="${TRAIN_BATCH_SIZE}" \
    algorithm.gradient_checkpointing_rate=1.0 \
    algorithm.model.tuned_ckpt_path="${TUNED_CKPT_PATH}"
fi
