#!/usr/bin/env bash
set -euo pipefail

TEACHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "${TEACHER_DIR}/.." && pwd)}"

PIPER_RAW_ROOT="${PIPER_RAW_ROOT:-${WORKSPACE}/data/realworld_merged}"
PIPER_TEACHER_ROOT="${PIPER_TEACHER_ROOT:-${WORKSPACE}/data/piper_teacher_sft_cam_high_native_640x480}"
PIPER_METADATA="${PIPER_METADATA:-piper_videos.csv}"
PIPER_DATASETS="${PIPER_DATASETS:-piper_realworld11}"

TEXT_ENCODER_CKPT="${TEXT_ENCODER_CKPT:-${TEACHER_DIR}/data/ckpts/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth}"
TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-${TEACHER_DIR}/data/ckpts/Wan2.1-I2V-14B-480P/google/umt5-xxl}"
TUNED_CKPT_PATH="${TUNED_CKPT_PATH:-${TEACHER_DIR}/data/ckpts/checkpoints/lvp_14B.ckpt}"
PREPARE_NUM_WORKERS="${PREPARE_NUM_WORKERS:-4}"
PREPARE_FFMPEG_ENCODER="${PREPARE_FFMPEG_ENCODER:-libx264}"
RUN_NAME="${RUN_NAME:-piper_cam_high_only_i2v}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
NUM_NODES="${NUM_NODES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"
PIPER_HEIGHT="${PIPER_HEIGHT:-480}"
PIPER_WIDTH="${PIPER_WIDTH:-640}"

cd "${TEACHER_DIR}"
mkdir -p "${PIPER_TEACHER_ROOT}"

read -r -a piper_dataset_names <<< "${PIPER_DATASETS}"
dataset_args=()
for dataset_name in "${piper_dataset_names[@]}"; do
  if [[ "${dataset_name}" = /* || "${dataset_name}" == *"/"* ]]; then
    dataset_dir="${dataset_name}"
  else
    dataset_dir="${PIPER_RAW_ROOT}/${dataset_name}"
  fi

  if [[ -d "${dataset_dir}" ]]; then
    dataset_args+=(--dataset-dir "${dataset_dir}")
  else
    echo "Piper dataset directory does not exist: ${dataset_dir}" >&2
    exit 1
  fi
done

if [[ ${#dataset_args[@]} -eq 0 ]]; then
  echo "No Piper datasets specified." >&2
  exit 1
fi

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  python datasets/prepare_piper_sft_data_cam_high.py \
    "${dataset_args[@]}" \
    --output-root "${PIPER_TEACHER_ROOT}" \
    --metadata-path "${PIPER_TEACHER_ROOT}/${PIPER_METADATA}" \
    --preserve-source-resolution \
    --num-workers "${PREPARE_NUM_WORKERS}" \
    --ffmpeg-encoder "${PREPARE_FFMPEG_ENCODER}" \
    ${PREPARE_THREADS_PER_WORKER:+--threads-per-worker "${PREPARE_THREADS_PER_WORKER}"} \
    ${PREPARE_ENCODE_TIMEOUT:+--encode-timeout "${PREPARE_ENCODE_TIMEOUT}"} \
    ${PREPARE_EXTRA_ARGS:-}
fi

if [[ "${SKIP_PROMPT_EMBED:-0}" != "1" ]]; then
  CUDA_VISIBLE_DEVICES="${TEXT_CUDA_VISIBLE_DEVICES:-0}" python main.py \
    +name=process_piper_cam_high_embeds \
    experiment=process_data \
    dataset=piper \
    dataset.data_root="${PIPER_TEACHER_ROOT}" \
    dataset.metadata_path="${PIPER_METADATA}" \
    dataset.height="${PIPER_HEIGHT}" \
    dataset.width="${PIPER_WIDTH}" \
    algorithm=wan_i2v \
    algorithm.text_encoder.ckpt_path="${TEXT_ENCODER_CKPT}" \
    algorithm.text_encoder.name="${TEXT_ENCODER_NAME}" \
    'experiment.tasks=[cache_prompt_embed]' \
    experiment.new_data_root="${PIPER_TEACHER_ROOT}" \
    experiment.cache_prompt_embed.batch_size="${PROMPT_BATCH_SIZE}"
fi

if [[ "${SKIP_SFT:-0}" != "1" ]]; then
  accelerate launch --main_process_port "${MAIN_PROCESS_PORT}" main.py \
    +name="${RUN_NAME}" \
    experiment=exp_video \
    algorithm=wan_i2v \
    dataset=piper \
    dataset.data_root="${PIPER_TEACHER_ROOT}" \
    dataset.metadata_path="${PIPER_METADATA}" \
    dataset.height="${PIPER_HEIGHT}" \
    dataset.width="${PIPER_WIDTH}" \
    experiment.num_nodes="${NUM_NODES}" \
    algorithm.lang_guidance=0 \
    algorithm.hist_guidance=0 \
    experiment.training.batch_size="${TRAIN_BATCH_SIZE}" \
    algorithm.gradient_checkpointing_rate=1.0 \
    algorithm.text_encoder.name="${TEXT_ENCODER_NAME}" \
    algorithm.model.tuned_ckpt_path="${TUNED_CKPT_PATH}"
fi
