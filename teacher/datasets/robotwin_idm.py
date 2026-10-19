from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from idm import DinoPreprocessor, flip_action

try:
    import decord
except ImportError:  # pragma: no cover - exercised via cv2 fallback if unavailable
    decord = None

try:
    import av
except ImportError:  # pragma: no cover - exercised by cv2/decord fallback
    av = None

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - raised with a clearer message at runtime
    pq = None


FASTWAM_FRAME_WIDTH = 320
FASTWAM_FRAME_HEIGHT = 384
FASTWAM_TOP_HEIGHT = 256
FASTWAM_BOTTOM_HEIGHT = 128
FASTWAM_WRIST_WIDTH = 160
EPISODE_RE = re.compile(r"episode_(\d+)")
ALOHA_JOINT_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _episode_index(path: Path) -> int:
    match = EPISODE_RE.search(path.stem)
    if match is None:
        raise ValueError(f"Could not parse episode index from {path}")
    return int(match.group(1))


def _parse_split(split: str | None) -> set[int] | None:
    if not split:
        return None
    episode_indices: set[int] = set()
    for part in split.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start, end = part.split(":", 1)
            episode_indices.update(range(int(start), int(end)))
        else:
            episode_indices.add(int(part))
    return episode_indices


def compose_fastwam_frame(head: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    head_resized = cv2.resize(
        head,
        (FASTWAM_FRAME_WIDTH, FASTWAM_TOP_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    left_resized = cv2.resize(
        left,
        (FASTWAM_WRIST_WIDTH, FASTWAM_BOTTOM_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    right_resized = cv2.resize(
        right,
        (FASTWAM_WRIST_WIDTH, FASTWAM_BOTTOM_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    bottom = np.concatenate([left_resized, right_resized], axis=1)
    frame = np.concatenate([head_resized, bottom], axis=0)
    if frame.shape != (FASTWAM_FRAME_HEIGHT, FASTWAM_FRAME_WIDTH, 3):
        raise ValueError(f"Unexpected stitched frame shape: {frame.shape}")
    return frame


def compose_robotwin_frame_no_resize(
    head: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Stack RobotWin's 3 views without changing any camera resolution."""
    if head.ndim != 3 or left.ndim != 3 or right.ndim != 3:
        raise ValueError(
            f"Expected HWC RGB frames, got head={head.shape}, left={left.shape}, right={right.shape}"
        )
    if head.shape[2] != 3 or left.shape[2] != 3 or right.shape[2] != 3:
        raise ValueError(
            f"Expected RGB frames, got head={head.shape}, left={left.shape}, right={right.shape}"
        )
    if left.shape[0] != right.shape[0]:
        raise ValueError(f"Left/right wrist heights must match, got {left.shape} and {right.shape}")

    bottom = np.concatenate([left, right], axis=1)
    out_h = head.shape[0] + bottom.shape[0]
    out_w = max(head.shape[1], bottom.shape[1])
    frame = np.zeros((out_h, out_w, 3), dtype=head.dtype)

    head_x = (out_w - head.shape[1]) // 2
    frame[: head.shape[0], head_x: head_x + head.shape[1]] = head
    frame[head.shape[0] :, : bottom.shape[1]] = bottom
    return frame


def robotwin_idm_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([sample["images"] for sample in batch], dim=0)
    actions = torch.stack([sample["actions"] for sample in batch], dim=0)

    masks = None
    if all(sample["mask"] is not None for sample in batch):
        masks = torch.stack([sample["mask"] for sample in batch], dim=0)

    return {
        "images": images,
        "mask": masks,
        "actions": actions,
        "metadata": [sample["metadata"] for sample in batch],
    }


class RobotwinIDMDataset(Dataset):
    def __init__(self, cfg, split: str = "training") -> None:
        self.cfg = cfg
        self.split = split
        self.data_root = Path(cfg.data_root)
        self.video_root = (
            Path(cfg.video_root) if cfg.get("video_root", None) else None
        )
        self.input_source = cfg.get(
            "input_source",
            "video" if self.video_root is not None else "hdf5",
        )
        self.view_mode = cfg.get(
            "view_mode",
            "stitched" if self.video_root is not None else "single",
        )
        self.split_name = cfg.get("split_name", "train")
        self.task_config = cfg.get("task_config", "demo_clean")
        self.target_key = cfg.get("target_key", cfg.get("action_key", "action"))
        self.target_space = cfg.get("target_space", None)
        self.expected_robot_type = cfg.get("expected_robot_type", None)
        self.stitch_resize = cfg.get("stitch_resize", self.input_source != "lerobot")
        self.action_key = cfg.get("action_key", "joint_action/vector")
        self.single_rgb_key = cfg.get(
            "single_rgb_key",
            cfg.get("rgb_key", "observation/head_camera/rgb"),
        )
        self.stitched_rgb_keys = list(
            cfg.get(
                "stitched_rgb_keys",
                [
                    "observation/head_camera/rgb",
                    "observation/left_camera/rgb",
                    "observation/right_camera/rgb",
                ],
            )
        )
        self.image_keys = list(
            cfg.get(
                "image_keys",
                [
                    "observation.images.cam_high",
                    "observation.images.cam_left_wrist",
                    "observation.images.cam_right_wrist",
                ],
            )
        )
        self.rgb_key = self.single_rgb_key
        self.mask_key = cfg.get("mask_key", "observation/head_camera/left_arm_mask")
        self.use_gt_mask = cfg.get("use_gt_mask", False)
        self.num_frames = cfg.get("num_frames", 3)
        self.frame_stride = cfg.get("frame_stride", 2)
        self.do_flip = split == "training" and cfg.get("do_flip", False)
        # Kept for config compatibility. IDM now relies on DinoPreprocessor
        # for the only spatial resize step instead of a legacy pre-resize here.
        self.crop_and_resize = cfg.get("crop_and_resize", True)
        self.preprocessor = DinoPreprocessor(
            {
                "use_transform": split == "training" and cfg.get("use_transform", True),
                "dino_size": cfg.get("dino_size", 512),
            }
        )
        self.preprocessor.set_augmentation_progress(0.0)

        self.h5_paths: list[Path | None] = []
        self.video_paths: list[Path | tuple[Path, ...] | None] = []
        self.video_lengths: list[int | None] = []
        self.lengths: list[int] = []
        self.actions: list[torch.Tensor] = []
        self.task_names: list[str] = []

        if self.input_source not in {"hdf5", "video", "lerobot"}:
            raise ValueError(f"Unsupported input_source={self.input_source}")
        if self.view_mode not in {"single", "stitched"}:
            raise ValueError(f"Unsupported view_mode={self.view_mode}")
        if self.input_source == "lerobot" and self.view_mode != "stitched":
            raise ValueError("input_source=lerobot requires view_mode=stitched.")
        if self.input_source == "video" and self.video_root is None:
            raise ValueError("input_source=video requires dataset.video_root to be set.")
        if self.view_mode == "stitched" and self.use_gt_mask:
            raise ValueError(
                "use_gt_mask=True is not supported when view_mode=stitched."
            )

        self._scan_metadata()

    def _scan_metadata(self) -> None:
        if self.input_source == "lerobot":
            self._scan_lerobot_metadata()
            return

        h5_files: list[Path] = []
        for task_dir in sorted(self.data_root.iterdir()):
            if not task_dir.is_dir():
                continue
            cfg_dir = task_dir / self.task_config
            if not cfg_dir.exists():
                continue
            for path in sorted(cfg_dir.rglob("*.hdf5")):
                h5_files.append(path)

        if not h5_files:
            raise RuntimeError(
                f"No HDF5 files found under {self.data_root} with task_config={self.task_config}"
            )

        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as handle:
                if self.action_key not in handle:
                    continue
                action_ds = handle[self.action_key]
                length = len(action_ds)
                video_path = None
                video_len = None
                if self.input_source == "hdf5":
                    if self.view_mode == "single":
                        if self.single_rgb_key not in handle:
                            continue
                        length = min(length, len(handle[self.single_rgb_key]))
                        if self.use_gt_mask and self.mask_key not in handle:
                            continue
                    else:
                        if any(key not in handle for key in self.stitched_rgb_keys):
                            continue
                        length = min(
                            length,
                            *(len(handle[key]) for key in self.stitched_rgb_keys),
                        )
                else:
                    video_path = self._resolve_video_path(h5_path)
                    video_len = self._get_video_length(video_path)
                    length = min(length, video_len)
                if length <= 0:
                    continue
                self.h5_paths.append(h5_path)
                self.video_paths.append(video_path)
                self.video_lengths.append(video_len)
                self.lengths.append(length)
                self.actions.append(torch.from_numpy(action_ds[:length]).float())
                self.task_names.append(h5_path.relative_to(self.data_root).parts[0])

        if not self.h5_paths:
            raise RuntimeError(
                f"Found {len(h5_files)} HDF5 files under {self.data_root}, but none matched "
                f"the required keys for the configured IDM dataset."
            )

        self.data_begin = np.cumsum([0] + self.lengths[:-1])
        self.data_end = np.cumsum(self.lengths)

    def _scan_lerobot_metadata(self) -> None:
        if pq is None:
            raise ImportError(
                "RobotwinIDMDataset input_source=lerobot requires pyarrow. "
                "Install it with `pip install pyarrow`."
            )

        info_path = self.data_root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")

        info = _load_json(info_path)
        if self.expected_robot_type is not None:
            robot_type = info.get("robot_type", None)
            if robot_type != self.expected_robot_type:
                raise ValueError(
                    f"Expected robot_type={self.expected_robot_type!r}, got {robot_type!r} "
                    f"in {info_path}."
                )

        features = info.get("features", {})
        if self.target_key not in features:
            available = ", ".join(sorted(features))
            raise KeyError(
                f"dataset.target_key={self.target_key!r} is not in LeRobot features. "
                f"Available fields: {available}"
            )
        self._validate_lerobot_target_feature(features[self.target_key], self.target_key)
        missing_images = [key for key in self.image_keys if key not in features]
        if missing_images:
            raise KeyError(f"Missing LeRobot image keys: {missing_images}")

        split_indices = _parse_split(info.get("splits", {}).get(self.split_name))
        parquet_paths = sorted(
            (self.data_root / "data").glob("chunk-*/episode_*.parquet"),
            key=_episode_index,
        )
        if not parquet_paths:
            raise RuntimeError(f"No LeRobot parquet episodes found under {self.data_root / 'data'}")

        chunks_size = int(info.get("chunks_size", 1000))
        video_template = info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )

        for parquet_path in parquet_paths:
            episode_index = _episode_index(parquet_path)
            if split_indices is not None and episode_index not in split_indices:
                continue

            target = self._load_lerobot_target(parquet_path, self.target_key)
            if target.shape[0] == 0:
                continue

            episode_chunk = episode_index // chunks_size
            video_paths = tuple(
                self.data_root
                / video_template.format(
                    episode_chunk=episode_chunk,
                    video_key=image_key,
                    episode_index=episode_index,
                )
                for image_key in self.image_keys
            )
            for video_path in video_paths:
                if not video_path.exists():
                    raise FileNotFoundError(f"Missing LeRobot video: {video_path}")

            video_len = min(self._get_video_length(video_path) for video_path in video_paths)
            length = min(int(target.shape[0]), video_len)
            if length <= 0:
                continue

            self.h5_paths.append(None)
            self.video_paths.append(video_paths)
            self.video_lengths.append(video_len)
            self.lengths.append(length)
            self.actions.append(torch.from_numpy(target[:length]).float())
            self.task_names.append("robotwin_lerobot")

        if not self.actions:
            raise RuntimeError(
                f"No usable LeRobot RobotWin episodes found under {self.data_root} "
                f"for split={self.split_name!r} target_key={self.target_key!r}."
            )

        self.data_begin = np.cumsum([0] + self.lengths[:-1])
        self.data_end = np.cumsum(self.lengths)

    def _validate_lerobot_target_feature(self, feature: dict, key: str) -> None:
        shape = feature.get("shape", None)
        expected_target_dim = self.cfg.get("expected_target_dim", None)
        if expected_target_dim is not None:
            expected_shape = [int(expected_target_dim)]
            if shape != expected_shape:
                raise ValueError(
                    f"Expected {key!r} shape {expected_shape}, got {shape}."
                )
        elif shape is None:
            raise ValueError(f"Missing shape metadata for {key!r}: {feature!r}")

        expected_target_names = self.cfg.get("expected_target_names", None)
        if expected_target_names is not None:
            names = feature.get("names", None)
            if not names or not isinstance(names, list) or not isinstance(names[0], list):
                raise ValueError(f"Expected names for {key!r}, got {names!r}.")
            if names[0] != list(expected_target_names):
                raise ValueError(
                    f"Expected names for {key!r} to be {list(expected_target_names)!r}, got {names[0]!r}."
                )
            return

        if self.target_space != "joint":
            return

        names = feature.get("names", None)
        if not names or not isinstance(names, list) or not isinstance(names[0], list):
            raise ValueError(f"Expected joint names for {key!r}, got {names!r}.")
        if names[0] != ALOHA_JOINT_NAMES:
            raise ValueError(
                f"Expected ALOHA joint-space names for {key!r}, got {names[0]!r}. "
                "This script is for joint action/state targets, not EE pose targets."
            )

    def _load_lerobot_target(self, parquet_path: Path, key: str) -> np.ndarray:
        table = pq.read_table(parquet_path, columns=[key])
        values = table[key].to_pylist()
        target = np.asarray(values, dtype=np.float32)
        if target.ndim == 1:
            target = target[:, None]
        return target

    def compute_target_stats(
        self,
        indices: list[int] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = torch.cat(self.actions, dim=0)
        if indices is not None:
            targets = targets[torch.as_tensor(indices, dtype=torch.long)]
        return targets.mean(dim=0), targets.std(dim=0, unbiased=False)

    def __len__(self) -> int:
        return int(self.data_end[-1])

    def _decode_image(self, data):
        if isinstance(data, np.bytes_):
            data = bytes(data)

        if isinstance(data, (bytes, bytearray)):
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode encoded RGB image.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)

        if isinstance(data, np.ndarray):
            image = data
            return Image.fromarray(image)

        raise TypeError(f"Unsupported RGB payload type: {type(data)}")

    def _decode_mask(self, data):
        if isinstance(data, np.bytes_):
            data = bytes(data)

        if isinstance(data, (bytes, bytearray)):
            mask = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError("Failed to decode encoded mask.")
            return Image.fromarray(mask)

        if isinstance(data, np.ndarray):
            mask = data
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask = mask[:, :, 0]
            return Image.fromarray(mask)

        raise TypeError(f"Unsupported mask payload type: {type(data)}")

    def _load_hdf5_frames(self, handle, frame_indices: list[int]) -> list[np.ndarray]:
        if self.view_mode == "single":
            rgb_ds = handle[self.single_rgb_key]
            return [np.asarray(self._decode_image(rgb_ds[frame_idx])) for frame_idx in frame_indices]

        rgb_datasets = [handle[key] for key in self.stitched_rgb_keys]
        frames: list[np.ndarray] = []
        for frame_idx in frame_indices:
            head, left, right = (
                np.asarray(self._decode_image(dataset[frame_idx]))
                for dataset in rgb_datasets
            )
            frames.append(compose_fastwam_frame(head, left, right))
        return frames

    def _resolve_video_path(self, h5_path: Path) -> Path:
        relative = h5_path.relative_to(self.data_root)
        if len(relative.parts) < 4 or relative.parts[-2] != "data":
            raise ValueError(
                f"Expected HDF5 path under <task>/<task_config>/data/*.hdf5, got {h5_path}"
            )
        task_name, task_config = relative.parts[0], relative.parts[1]
        video_path = self.video_root / task_name / task_config / f"{h5_path.stem}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(
                f"Missing stitched video for {h5_path}: expected {video_path}"
            )
        return video_path

    def _get_video_length(self, video_path: Path) -> int:
        if decord is not None:
            try:
                return len(decord.VideoReader(video_path.as_posix()))
            except Exception:
                pass
        if av is not None:
            container = av.open(video_path.as_posix())
            try:
                stream = container.streams.video[0]
                if stream.frames:
                    return int(stream.frames)
                return sum(1 for _ in container.decode(video=0))
            finally:
                container.close()

        cap = cv2.VideoCapture(str(video_path))
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        if frame_count <= 0:
            raise RuntimeError(f"Failed to read frame count from {video_path}")
        return frame_count

    def _load_video_frames(self, video_path: Path, frame_indices: list[int]) -> list[np.ndarray]:
        if decord is not None:
            try:
                frames = decord.VideoReader(video_path.as_posix()).get_batch(frame_indices)
                if hasattr(frames, "asnumpy"):
                    frames = frames.asnumpy()
                elif isinstance(frames, torch.Tensor):
                    frames = frames.cpu().numpy()
                else:
                    frames = np.asarray(frames)
                return [frame for frame in frames]
            except Exception:
                pass
        if av is not None:
            wanted = set(frame_indices)
            frames = {}
            container = av.open(video_path.as_posix())
            try:
                max_idx = max(wanted)
                for frame_idx, frame in enumerate(container.decode(video=0)):
                    if frame_idx in wanted:
                        frames[frame_idx] = frame.to_ndarray(format="rgb24")
                    if frame_idx >= max_idx and len(frames) == len(wanted):
                        break
            finally:
                container.close()
            missing = wanted - set(frames)
            if missing:
                raise RuntimeError(f"Failed to read frames {sorted(missing)} from {video_path}")
            return [frames[idx] for idx in frame_indices]

        cap = cv2.VideoCapture(str(video_path))
        frames: list[np.ndarray] = []
        try:
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        return frames

    def _load_lerobot_frames(
        self,
        video_paths: tuple[Path, ...],
        frame_indices: list[int],
    ) -> list[np.ndarray]:
        if len(video_paths) == 1:
            return self._load_video_frames(video_paths[0], frame_indices)
        if len(video_paths) != 3:
            raise ValueError(
                f"RobotWin LeRobot IDM expects either 1 camera video (single-view) "
                f"or 3 camera videos (stitched), got {len(video_paths)}"
            )
        frame_groups = [
            self._load_video_frames(video_path, frame_indices)
            for video_path in video_paths
        ]
        return [
            (
                compose_fastwam_frame(head, left, right)
                if self.stitch_resize
                else compose_robotwin_frame_no_resize(head, left, right)
            )
            for head, left, right in zip(*frame_groups)
        ]

    def _frame_indices(self, local_idx: int, video_len: int) -> list[int]:
        if self.num_frames == 1:
            indices = [local_idx]
        elif self.num_frames == 2:
            indices = [local_idx - self.frame_stride, local_idx]
        elif self.num_frames == 3:
            indices = [
                local_idx - self.frame_stride,
                local_idx,
                local_idx + self.frame_stride,
            ]
        else:
            raise ValueError(f"Unsupported num_frames={self.num_frames}")
        return [max(0, min(idx, video_len - 1)) for idx in indices]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        video_idx = int(np.searchsorted(self.data_end, idx, side="right"))
        local_idx = int(idx - self.data_begin[video_idx])
        h5_path = self.h5_paths[video_idx]
        mask = None

        if self.input_source == "lerobot":
            video_len = self.video_lengths[video_idx]
            image_indices = self._frame_indices(local_idx, video_len)
            video_paths = self.video_paths[video_idx]
            if not isinstance(video_paths, tuple):
                raise TypeError(f"Expected tuple video paths for LeRobot sample, got {type(video_paths)}")
            frames = self._load_lerobot_frames(video_paths, image_indices)
            images = [self.preprocessor.process_image(frame) for frame in frames]
        elif self.input_source == "hdf5":
            with h5py.File(h5_path, "r") as handle:
                if self.view_mode == "single":
                    video_len = len(handle[self.single_rgb_key])
                else:
                    video_len = min(len(handle[key]) for key in self.stitched_rgb_keys)
                image_indices = self._frame_indices(local_idx, video_len)
                frames = self._load_hdf5_frames(handle, image_indices)
                images = [self.preprocessor.process_image(frame) for frame in frames]

                if self.use_gt_mask:
                    mask_ds = handle[self.mask_key]
                    mask = self.preprocessor.process_mask(
                        self._decode_mask(mask_ds[local_idx])
                    )
        else:
            video_len = self.video_lengths[video_idx]
            image_indices = self._frame_indices(local_idx, video_len)
            frames = self._load_video_frames(self.video_paths[video_idx], image_indices)
            images = [self.preprocessor.process_image(frame) for frame in frames]

        images = torch.stack(images, dim=0)
        actions = self.actions[video_idx][local_idx].clone()

        is_flipped = self.do_flip and np.random.rand() < 0.5
        if is_flipped:
            images = torch.flip(images, dims=[-1])
            if mask is not None:
                mask = torch.flip(mask, dims=[-1])
            actions = flip_action(actions)

        metadata = {
            "task": self.task_names[video_idx],
            "file_path": str(h5_path) if h5_path is not None else None,
            "video_path": (
                str(self.video_paths[video_idx]) if self.video_paths[video_idx] else None
            ),
            "frame_idx": local_idx,
            "video_idx": video_idx,
            "split": self.split,
            "input_source": self.input_source,
            "view_mode": self.view_mode,
            "target_key": self.target_key,
        }
        return {
            "images": images,
            "mask": mask,
            "actions": actions,
            "metadata": metadata,
        }
