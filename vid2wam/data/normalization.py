"""Normalization utilities required by Vid2WAM evaluation."""

import json
from typing import Annotated, Any, Dict, Literal, Union

import numpy as np
import torch

from vid2wam.support.tensors import dict_apply

ConstRange = Annotated[str, "format: const_min/const_max"]
NormMode = Union[Literal["min/max", "q01/q99", "z-score"], ConstRange]


class LinearNormalizer:
    def __init__(
        self,
        shape_meta,
        use_stepwise_action_norm: bool,
        default_mode: NormMode,
        exception_mode: Dict[str, Dict[str, NormMode]] | None,
        stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
    ):
        self.normalizers = {"action": {}, "state": {}}
        self.stats = stats

        for meta in shape_meta["action"]:
            key = meta["key"]
            prefix = "stepwise_" if use_stepwise_action_norm else "global_"
            current_stats = {
                name.removeprefix(prefix): value
                for name, value in stats["action"][key].items()
                if name.startswith(prefix)
            }
            mode = default_mode
            if exception_mode is not None and key in exception_mode.get("action", {}):
                mode = exception_mode["action"][key]
            self.normalizers["action"][key] = SingleFieldLinearNormalizer(
                stats=current_stats,
                mode=mode,
            )

        for meta in shape_meta["state"]:
            key = meta["key"]
            current_stats = {
                name.removeprefix("global_"): value
                for name, value in stats["state"][key].items()
                if name.startswith("global_")
            }
            mode = default_mode
            if exception_mode is not None and key in exception_mode.get("state", {}):
                mode = exception_mode["state"][key]
            self.normalizers["state"][key] = SingleFieldLinearNormalizer(
                stats=current_stats,
                mode=mode,
            )

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]):
        if "action" in batch:
            for key, normalizer in self.normalizers["action"].items():
                batch["action"][key] = normalizer.forward(batch["action"][key])
        for key, normalizer in self.normalizers["state"].items():
            batch["state"][key] = normalizer.forward(batch["state"][key])
        return batch

    def backward(self, batch: Dict[str, Dict[str, torch.Tensor]]):
        if "action" in batch:
            for key, normalizer in self.normalizers["action"].items():
                batch["action"][key] = normalizer.backward(batch["action"][key])
        if "state" in batch:
            for key, normalizer in self.normalizers["state"].items():
                batch["state"][key] = normalizer.backward(batch["state"][key])
        return batch


class SingleFieldLinearNormalizer:
    std_reg = 1.0e-8
    range_tol = 1.0e-4
    output_max = 1.0
    output_min = -1.0

    def __init__(self, stats, mode: NormMode = "min/max"):
        self.stats = stats
        self.mode = mode

        if mode == "z-score":
            input_mean, input_std = stats["mean"], stats["std"]
            self.scale = 1.0 / (input_std + self.std_reg)
            self.offset = -input_mean / (input_std + self.std_reg)
            return

        if mode == "min/max":
            input_min, input_max = stats["min"], stats["max"]
        elif mode == "q01/q99":
            input_min, input_max = stats["q01"], stats["q99"]
        else:
            const_min, const_max = map(float, str(mode).split("/"))
            input_min = torch.full_like(stats["min"], const_min)
            input_max = torch.full_like(stats["max"], const_max)

        input_range = input_max - input_min
        ignored_dimensions = input_range < self.range_tol
        input_range = input_range.clone()
        input_range[ignored_dimensions] = self.output_max - self.output_min
        self.scale = (self.output_max - self.output_min) / input_range
        self.offset = self.output_min - self.scale * input_min
        self.offset[ignored_dimensions] = (
            (self.output_max + self.output_min) / 2 - input_min[ignored_dimensions]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(value * self.scale + self.offset, -5.0, 5.0)

    def backward(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.offset) / self.scale


def load_dataset_stats_from_json(
    file_path: str,
    try_convert_tensor: bool = True,
) -> Dict[str, Any]:
    def is_numeric_list(value) -> bool:
        if not isinstance(value, list):
            return False
        if not value:
            return True
        if isinstance(value[0], (int, float)):
            return all(isinstance(item, (int, float)) for item in value)
        if isinstance(value[0], list):
            return all(is_numeric_list(item) for item in value)
        return False

    def convert_to_tensor(value):
        if isinstance(value, dict):
            return {key: convert_to_tensor(item) for key, item in value.items()}
        if isinstance(value, list):
            if is_numeric_list(value):
                try:
                    return torch.from_numpy(np.asarray(value))
                except Exception:
                    pass
            return [convert_to_tensor(item) for item in value]
        return value

    with open(file_path, "r", encoding="utf-8") as stats_file:
        data = json.load(stats_file)

    if try_convert_tensor:
        data = convert_to_tensor(data)

    return dict_apply(
        data,
        lambda value: value.to(torch.float32) if isinstance(value, torch.Tensor) else value,
    )
