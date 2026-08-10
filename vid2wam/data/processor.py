"""Minimal processor state required by the public Vid2WAM evaluators."""

from typing import Any, Dict, List, Optional

from .normalization import LinearNormalizer, NormMode


class Vid2WAMProcessor:
    def __init__(
        self,
        shape_meta: Dict[str, Any],
        num_obs_steps: int,
        num_output_cameras: int,
        action_output_dim: int,
        proprio_output_dim: int,
        action_state_transforms: Optional[List[Any]],
        use_stepwise_action_norm: bool,
        norm_default_mode: NormMode,
        norm_exception_mode: Optional[Dict[str, Dict[str, NormMode]]],
    ):
        self.shape_meta = shape_meta
        self.num_obs_steps = int(num_obs_steps)
        self.num_output_cameras = int(num_output_cameras)
        self.action_output_dim = int(action_output_dim)
        self.proprio_output_dim = int(proprio_output_dim)
        self.action_state_transforms = action_state_transforms
        self.use_stepwise_action_norm = bool(use_stepwise_action_norm)
        self.norm_default_mode = norm_default_mode
        self.norm_exception_mode = norm_exception_mode
        self._normalizer: Optional[LinearNormalizer] = None
        self._is_train: Optional[bool] = None

    @property
    def is_train(self) -> bool:
        if self._is_train is None:
            raise ValueError("Processor mode is unset. Call eval() first.")
        return self._is_train

    @property
    def normalizer(self) -> LinearNormalizer:
        if self._normalizer is None:
            raise ValueError("Normalizer is unset. Call set_normalizer_from_stats() first.")
        return self._normalizer

    def eval(self):
        self._is_train = False
        return self

    def set_normalizer_from_stats(self, dataset_stats: Dict[str, Any]) -> None:
        self._normalizer = LinearNormalizer(
            use_stepwise_action_norm=self.use_stepwise_action_norm,
            shape_meta=self.shape_meta,
            default_mode=self.norm_default_mode,
            exception_mode=self.norm_exception_mode,
            stats=dataset_stats,
        )

    def action_state_transform(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if "action" in batch:
            for meta in self.shape_meta["action"]:
                key = meta["key"]
                actual_dim = batch["action"][key].shape[-1]
                expected_dim = int(meta["raw_shape"])
                if actual_dim != expected_dim:
                    raise ValueError(
                        f"Action key {key} has raw dim {actual_dim}, expected {expected_dim}."
                    )

        for meta in self.shape_meta["state"]:
            key = meta["key"]
            actual_dim = batch["state"][key].shape[-1]
            expected_dim = int(meta["raw_shape"])
            if actual_dim != expected_dim:
                raise ValueError(
                    f"State key {key} has raw dim {actual_dim}, expected {expected_dim}."
                )

        if self.action_state_transforms is not None:
            for transform in self.action_state_transforms:
                batch = transform.forward(batch)

        if "action" in batch:
            for meta in self.shape_meta["action"]:
                key = meta["key"]
                actual_dim = batch["action"][key].shape[-1]
                expected_dim = int(meta["shape"])
                if actual_dim != expected_dim:
                    raise ValueError(
                        f"Action key {key} has transformed dim {actual_dim}, expected {expected_dim}."
                    )

        for meta in self.shape_meta["state"]:
            key = meta["key"]
            actual_dim = batch["state"][key].shape[-1]
            expected_dim = int(meta["shape"])
            if actual_dim != expected_dim:
                raise ValueError(
                    f"State key {key} has transformed dim {actual_dim}, expected {expected_dim}."
                )

        return batch
