from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from vid2wam.support.logging import get_logger

from .action_backbone import ActionDiT
from .world_action_model import Vid2WAM
from vid2wam.loading.wan_loader import load_wan22_ti2v_5b_components
from .mixture_transformer import MoT

logger = get_logger(__name__)


class ActionTokenAdapter(nn.Module):
    """Small residual adapter for source-specific action-token corrections."""

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int = 128,
        dropout: float = 0.0,
        scale: float = 1.0,
        eps: float = 1.0e-6,
    ):
        super().__init__()
        bottleneck_dim = int(bottleneck_dim)
        if bottleneck_dim <= 0:
            raise ValueError(f"`bottleneck_dim` must be positive, got {bottleneck_dim}.")
        self.scale = float(scale)
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.act = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(float(dropout))
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.dropout(self.act(self.down(self.norm(x)))))
        return x + self.scale * residual


class Vid2WAMActionAdapter(Vid2WAM):
    """Evaluation architecture for Vid2WAM checkpoints with action adapters."""

    def __init__(self, *args, domain_adapters: nn.ModuleDict, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_adapters = domain_adapters
        self.dit = nn.ModuleDict({
            "mot": self.mot,
            "domain_adapters": self.domain_adapters,
        })

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = True,
        mot_checkpoint_mixed_attn: bool = False,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        domain_adapter_config: dict[str, Any] | None = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required.")
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required.")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )
        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        cls._validate_action_expert_compatible(video_expert, action_expert, "action_expert")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        domain_adapters = cls._build_domain_adapters(
            hidden_dim=int(action_dit_config["hidden_dim"]),
            eps=float(action_dit_config.get("eps", 1.0e-6)),
            config=domain_adapter_config or {},
            device=device,
            dtype=torch_dtype,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            domain_adapters=domain_adapters,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    @staticmethod
    def _validate_action_expert_compatible(video_expert, action_expert: ActionDiT, name: str) -> None:
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError(f"{name} `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError(f"{name} `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError(f"{name} `num_layers` must match video expert.")

    @staticmethod
    def _build_domain_adapters(
        hidden_dim: int,
        eps: float,
        config: dict[str, Any],
        device: str,
        dtype: torch.dtype,
    ) -> nn.ModuleDict:
        bottleneck_dim = int(config.get("bottleneck_dim", 128))
        dropout = float(config.get("dropout", 0.0))
        scale = float(config.get("scale", 1.0))

        def make_pair() -> nn.ModuleDict:
            return nn.ModuleDict({
                "input": ActionTokenAdapter(
                    hidden_dim=hidden_dim,
                    bottleneck_dim=bottleneck_dim,
                    dropout=dropout,
                    scale=scale,
                    eps=eps,
                ),
                "output": ActionTokenAdapter(
                    hidden_dim=hidden_dim,
                    bottleneck_dim=bottleneck_dim,
                    dropout=dropout,
                    scale=scale,
                    eps=eps,
                ),
            })

        adapters = nn.ModuleDict({"gt": make_pair(), "pseudo": make_pair()})
        return adapters.to(device=device, dtype=dtype)

    def _apply_action_adapter(self, tokens: torch.Tensor, domain: str, stage: str) -> torch.Tensor:
        if domain not in self.domain_adapters:
            raise ValueError(f"Unknown action-adapter domain: {domain}.")
        return self.domain_adapters[domain][stage](tokens)

    def _mot_forward_with_domain(
        self,
        video_pre: dict[str, Any],
        action_pre: dict[str, Any],
        domain: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_tokens = self._apply_action_adapter(action_pre["tokens"], domain=domain, stage="input")
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        adapted_action = self._apply_action_adapter(tokens_out["action"], domain=domain, stage="output")
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(adapted_action, action_pre)
        return pred_video, pred_action

    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        return self._mot_forward_with_domain(video_pre, action_pre, domain="gt")

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        domain: str = "gt",
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self._apply_action_adapter(action_pre["tokens"], domain=domain, stage="input")
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_tokens,
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        action_tokens = self._apply_action_adapter(action_tokens, domain=domain, stage="output")
        return self.action_expert.post_dit(action_tokens, action_pre)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "gt_mot" in payload:
            logger.warning("Loading dual-action `gt_mot` checkpoint into shared adapter branch.")
            self.mot.load_state_dict(payload["gt_mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing `mot`, `gt_mot`, and `dit` keys: {path}")

        if "domain_adapters" in payload:
            self.domain_adapters.load_state_dict(payload["domain_adapters"], strict=False)
        else:
            logger.warning("Checkpoint has no `domain_adapters`; keeping initialized adapters.")

        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder`; keeping current params.")
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
