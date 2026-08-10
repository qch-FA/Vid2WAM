"""Evaluation-only factory for the public Vid2WAM checkpoint architecture."""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf


def _to_dict(value, name: str, *, required: bool = False) -> dict:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        if required:
            raise ValueError(f"`{name}` is required.")
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}.")
    return value


def create_vid2wam_action_adapter(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = True,
    video_scheduler=None,
    action_scheduler=None,
    domain_adapter=None,
    mot_checkpoint_mixed_attn: bool = False,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from vid2wam.engine.action_adapter import Vid2WAMActionAdapter

    video_dit_config = _to_dict(video_dit_config, "video_dit_config", required=True)
    action_dit_config = _to_dict(action_dit_config, "action_dit_config", required=True)
    video_scheduler = _to_dict(video_scheduler, "video_scheduler")
    action_scheduler = _to_dict(action_scheduler, "action_scheduler", required=True)
    domain_adapter = _to_dict(domain_adapter, "domain_adapter")

    required_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing = required_scheduler_keys - set(action_scheduler)
    if missing:
        raise ValueError(f"`action_scheduler` missing required keys: {sorted(missing)}")

    return Vid2WAMActionAdapter.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        domain_adapter_config=domain_adapter,
    )
