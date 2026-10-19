# Teacher and IDM Training

Follow the Vid2WAM environment setup, then install the training dependencies from the repository root:

```bash
pip install -e ".[teacher]"
```

Teacher training additionally requires FlashAttention 2 and a CUDA toolkit matching the installed PyTorch build:

```bash
pip install packaging ninja
pip install "flash-attn>=2.7,<3" --no-build-isolation
```

The teacher is a Wan2.1 I2V 14B model with the training examples below using eight GPUs on one node. Adjust `--nproc_per_node`, per-GPU batch size, and gradient accumulation for your hardware. IDM training supports a single GPU and does not require FlashAttention or the Wan checkpoints.

### Initial Teacher Checkpoints

Download the Wan2.1 configuration, VAE, text encoder, CLIP encoder, and tokenizer, together with the [LVP initialization checkpoint](https://huggingface.co/KempnerInstituteAI/LVP).

```bash
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download

snapshot_download(
    repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
    local_dir="teacher/data/ckpts/Wan2.1-I2V-14B-480P",
    allow_patterns=[
        "config.json", "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "google/umt5-xxl/*",
    ],
)
hf_hub_download(
    repo_id="KempnerInstituteAI/LVP",
    filename="checkpoints/lvp_14B.ckpt",
    local_dir="teacher/data/ckpts",
)
PY

cd teacher
export WAN_DIR="$PWD/data/ckpts/Wan2.1-I2V-14B-480P"
export TEACHER_INIT="$PWD/data/ckpts/checkpoints/lvp_14B.ckpt"
```

Run all remaining training commands from `teacher/` in the same shell.

### LIBERO Teacher

Use extracted LIBERO datasets in LeRobot format. Each dataset directory should contain `meta/`, `data/`, and `videos/`. Prepare the videos by stitching the main and wrist camera images horizontally, with each camera resized to 224 × 224:

```bash
export LIBERO_RAW_ROOT=/path/to/libero_lerobot
export LIBERO_TEACHER_ROOT="$PWD/data/libero_teacher"

python datasets/prepare_libero_sft_data.py \
  --dataset-dir "$LIBERO_RAW_ROOT/libero_goal_no_noops_lerobot" \
  --output-root "$LIBERO_TEACHER_ROOT" \
  --metadata-path "$LIBERO_TEACHER_ROOT/libero_videos.csv"
```

Repeat `--dataset-dir` to include additional suites. Cache the prompt embeddings before training:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  +name=libero_prompt_embeddings \
  experiment=process_data algorithm=wan_i2v dataset=libero \
  dataset.data_root="$LIBERO_TEACHER_ROOT" \
  algorithm.text_encoder.ckpt_path="$WAN_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  algorithm.text_encoder.name="$WAN_DIR/google/umt5-xxl" \
  'experiment.tasks=[cache_prompt_embed]' \
  experiment.new_data_root="$LIBERO_TEACHER_ROOT"
```

Start teacher training:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py \
  +name=libero_teacher \
  experiment=exp_video algorithm=wan_i2v dataset=libero \
  dataset.data_root="$LIBERO_TEACHER_ROOT" \
  algorithm.model.tuned_ckpt_path="$TEACHER_INIT" \
  algorithm.text_encoder.name="$WAN_DIR/google/umt5-xxl" \
  algorithm.lang_guidance=0 algorithm.hist_guidance=0 \
  experiment.training.batch_size=1 \
  experiment.training.max_epochs=500 experiment.training.max_steps=-1 \
  hydra.run.dir=outputs/libero_teacher
```

### RoboTwin Teacher

Use a RoboTwin LeRobot dataset with `meta/info.json`, `meta/episodes.jsonl`, episode parquet files, and the three camera streams `observation.images.cam_high`, `observation.images.cam_left_wrist`, and `observation.images.cam_right_wrist`. 

Generate the metadata CSV. The loader samples matching frames from all three cameras and stitches a 320 × 256 head image above two 160 × 128 wrist images, producing 384 × 320 frames:

```bash
export ROBOTWIN_ROOT=/path/to/robotwin_lerobot

python datasets/generate_robotwin_lerobot_csv.py \
  --root-dir "$ROBOTWIN_ROOT" \
  --output-csv "$ROBOTWIN_ROOT/robotwin_videos.csv"

CUDA_VISIBLE_DEVICES=0 python main.py \
  +name=robotwin_prompt_embeddings \
  experiment=process_data algorithm=wan_i2v dataset=robotwin_lerobot \
  dataset.data_root="$ROBOTWIN_ROOT" \
  algorithm.text_encoder.ckpt_path="$WAN_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  algorithm.text_encoder.name="$WAN_DIR/google/umt5-xxl" \
  'experiment.tasks=[cache_prompt_embed]' \
  experiment.new_data_root="$ROBOTWIN_ROOT"

torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py \
  +name=robotwin_teacher \
  experiment=exp_video algorithm=wan_i2v dataset=robotwin_lerobot \
  dataset.data_root="$ROBOTWIN_ROOT" \
  algorithm.model.tuned_ckpt_path="$TEACHER_INIT" \
  algorithm.text_encoder.name="$WAN_DIR/google/umt5-xxl" \
  algorithm.lang_guidance=0 algorithm.hist_guidance=0 \
  experiment.training.batch_size=1 \
  experiment.training.max_epochs=500 experiment.training.max_steps=-1 \
  hydra.run.dir=outputs/robotwin_teacher
```

### IDM

Run IDM commands from `Vid2WAM/teacher/`.

```bash
export LIBERO_RAW_ROOT=/path/to/libero_lerobot

CUDA_VISIBLE_DEVICES=0 python main.py \
  +name=libero_action_idm \
  experiment=exp_idm algorithm=idm_resnet_plus dataset=libero_idm \
  dataset.data_root="$LIBERO_RAW_ROOT/libero_goal_no_noops_lerobot" \
  dataset.target_key=action dataset.is_abs=true \
  algorithm.output_dim=7 \
  experiment.training.batch_size=32 \
  experiment.training.num_iterations=200000 \
  hydra.run.dir=outputs/libero_action_idm
```

For the LIBERO state IDM, use the same command with `dataset.target_key=observation.state dataset.is_abs=true algorithm.output_dim=8` for datasets with eight-dimensional states. Match `algorithm.output_dim` to the target feature dimension in `meta/info.json`.

For the RoboTwin action IDM:

```bash
export ROBOTWIN_ROOT=/path/to/robotwin_lerobot

ROBOTWIN_LEROBOT_ROOT="$ROBOTWIN_ROOT" \
  CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
  bash scripts/train_robotwin_lerobot_idm.sh idm_target=action \
  hydra.run.dir=outputs/robotwin_action_idm
```

Use `idm_target=state` to train the 14-dimensional joint state IDM.

Torchvision downloads the ResNet initialization weights on first use and caches them under `TORCH_HOME`. For offline training, set `algorithm.pretrained_path=/path/to/resnet50-11ad3fa6.pth`.


## Acknowledgements

The Wan implementation retains its original copyright notices and is based on [Wan2.1](https://github.com/Wan-Video/Wan2.1). Teacher initialization uses [LVP](https://huggingface.co/KempnerInstituteAI/LVP).
