import torch
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from algorithms.wan import WanImageToVideo, WanTextToVideo
from datasets.libero import LiberoDataset
from datasets.piper import PiperDataset
from datasets.robotwin import RobotwinDataset
from datasets.robotwin_lerobot import RobotwinLerobotDataset
from datasets.video_base import SingleFrameVideoDataset
from .exp_base import BaseLightningExperiment


class VideoPredictionExperiment(BaseLightningExperiment):
    """
    A video prediction experiment
    """

    compatible_algorithms = dict(
        wan_i2v=WanImageToVideo,
        wan_t2v=WanTextToVideo,
        wan_toy=WanImageToVideo,
    )

    compatible_datasets = dict(
        image_csv=SingleFrameVideoDataset,
        ours_test=SingleFrameVideoDataset,
        libero=LiberoDataset,
        piper=PiperDataset,
        robotwin=RobotwinDataset,
        robotwin_lerobot=RobotwinLerobotDataset,
    )

    def _build_strategy(self):
        from lightning.pytorch.strategies.fsdp import FSDPStrategy

        if self.cfg.strategy == "ddp":
            return super()._build_strategy()
        elif self.cfg.strategy == "fsdp":
            return FSDPStrategy(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                ),
                cpu_offload=True, # Enable CPU Offload to save GPU memory for VAE activations
                auto_wrap_policy=ModuleWrapPolicy(self.algo.classes_to_shard()),
                sharding_strategy="FULL_SHARD",
                # sharding_strategy="HYBRID_SHARD",
                device_mesh=None,
            )

        else:
            return self.cfg.strategy

    def download_dataset(self):
        dataset = self._build_dataset("training")
