from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from idm import DinoPreprocessor

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - raised with a clearer message at runtime
    pq = None

try:
    import decord
except ImportError:  # pragma: no cover - exercised by cv2/av fallback
    decord = None

try:
    import av
except ImportError:  # pragma: no cover - exercised by cv2 fallback
    av = None

try:
    import cv2
except ImportError:  # pragma: no cover - only needed if decord/av are unavailable
    cv2 = None


EPISODE_RE = re.compile(r"episode_(\d+)")
DEFAULT_LIBERO_DATASET_DIRS = [
    "libero_10_no_noops_lerobot",
    "libero_spatial_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
]
EEF_ACTION_DIM = 6
FEATURE_ALIASES = {
    "actions": "action",
    "obs/ee_states": "observation.states.ee_state",
    "obs/ee_state": "observation.states.ee_state",
    "obs/state": "observation.state",
}


@dataclass(frozen=True)
class LiberoEpisode:
    dataset_dir: Path
    parquet_path: Path
    video_paths: tuple[Path, ...]
    episode_index: int
    length: int
    action_offset: int


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def _episode_index(path: Path) -> int:
    match = EPISODE_RE.search(path.stem)
    if match is None:
        raise ValueError(f"Could not parse episode index from {path}")
    return int(match.group(1))


def _resolve_feature_key(features: dict, key: str) -> str:
    candidates = [key]
    if key in FEATURE_ALIASES:
        candidates.append(FEATURE_ALIASES[key])
    slash_to_dot = key.replace("/", ".")
    if slash_to_dot != key:
        candidates.append(slash_to_dot)

    for candidate in candidates:
        if candidate in features:
            return candidate

    available = ", ".join(sorted(features))
    raise KeyError(f"{key!r} is not in LeRobot features. Available fields: {available}")


def _resolve_dataset_dirs(cfg) -> list[Path]:
    data_root = cfg.get("data_root", None)
    data_root_path = Path(data_root).expanduser() if data_root else None
    dataset_dirs = cfg.get("dataset_dirs", None)

    if dataset_dirs:
        paths = []
        for dataset_dir in dataset_dirs:
            path = Path(dataset_dir).expanduser()
            if not path.is_absolute():
                if data_root_path is None:
                    raise ValueError(
                        "Relative dataset_dirs require dataset.data_root to be set."
                    )
                path = data_root_path / path
            paths.append(path)
        return paths

    if data_root_path is None:
        raise ValueError("Set dataset.data_root or dataset.dataset_dirs.")
    if (data_root_path / "meta" / "info.json").exists():
        return [data_root_path]

    paths = [data_root_path / name for name in DEFAULT_LIBERO_DATASET_DIRS]
    missing = [path for path in paths if not (path / "meta" / "info.json").exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing expected LIBERO LeRobot datasets under {data_root_path}:\n"
            f"{missing_text}"
        )
    return paths


def _delta_eef_action_to_absolute(
    actions: np.ndarray,
    states: np.ndarray,
    source: Path,
) -> np.ndarray:
    if actions.shape[1] < EEF_ACTION_DIM or states.shape[1] < EEF_ACTION_DIM:
        raise ValueError(
            "Cannot convert LIBERO delta EEF action to absolute EEF action "
            f"for {source}: action shape {actions.shape}, state shape {states.shape}."
        )

    abs_actions = actions.copy()
    abs_actions[:, :EEF_ACTION_DIM] = (
        states[:, :EEF_ACTION_DIM] + actions[:, :EEF_ACTION_DIM]
    )
    return abs_actions


class LiberoLerobotIDMDataset(Dataset):
    def __init__(self, cfg, split: str = "training") -> None:
        if pq is None:
            raise ImportError(
                "LiberoLerobotIDMDataset requires pyarrow to read LeRobot parquet "
                "files. Install it with `pip install pyarrow`."
            )

        self.cfg = cfg
        self.split = split
        self.dataset_dirs = _resolve_dataset_dirs(cfg)
        self.split_name = cfg.get("split_name", "train")
        self.input_source = cfg.get("input_source", "video")
        if self.input_source != "video":
            raise ValueError("LiberoLerobotIDMDataset currently supports input_source=video.")
        self.target_key = cfg.get("target_key", cfg.get("action_key", "action"))
        self.is_abs = cfg.get("is_abs", True)
        self.abs_action_state_key = cfg.get(
            "abs_action_state_key",
            "observation.states.ee_state",
        )
        self.target_representation = "dataset"
        if not self.is_abs:
            self.target_representation = "absolute_eef"
        self.image_keys = self._resolve_image_keys(cfg)
        self.view_resize_size = self._resolve_view_resize_size(cfg)
        self.video_backend = cfg.get("video_backend", "av")
        self.num_frames = cfg.get("num_frames", 3)
        self.frame_stride = cfg.get("frame_stride", 2)
        self.do_flip = split == "training" and cfg.get("do_flip", False)
        if self.do_flip:
            raise ValueError(
                "do_flip is not supported for LIBERO actions. Keep dataset.do_flip=false."
            )

        self.preprocessor = DinoPreprocessor(
            {
                "use_transform": split == "training" and cfg.get("use_transform", True),
                "dino_size": cfg.get("dino_size", None),
            }
        )
        self.preprocessor.set_augmentation_progress(0.0)

        self.episodes: list[LiberoEpisode] = []
        self.actions: list[torch.Tensor] = []
        self.lengths: list[int] = []
        self._scan_metadata()

    def _resolve_image_keys(self, cfg) -> list[str]:
        image_keys = cfg.get("image_keys", None)
        if image_keys:
            return list(image_keys)
        return [
            cfg.get("image_key", "observation.images.image"),
            cfg.get("wrist_image_key", "observation.images.wrist_image"),
        ]

    def _resolve_view_resize_size(self, cfg) -> tuple[int, int] | None:
        size = cfg.get("view_resize_size", cfg.get("view_size", 224))
        if size is None:
            return None
        if isinstance(size, int):
            return (size, size)
        height, width = list(size)
        return (int(height), int(width))

    def _scan_metadata(self) -> None:
        for dataset_dir in self.dataset_dirs:
            info_path = dataset_dir / "meta" / "info.json"
            if not info_path.exists():
                raise FileNotFoundError(f"Missing {info_path}")

            info = _load_json(info_path)
            features = info.get("features", {})
            target_key = _resolve_feature_key(features, self.target_key)
            abs_action_state_key = None
            if not self.is_abs:
                if target_key != "action":
                    raise ValueError(
                        "dataset.is_abs=false is only supported when "
                        "dataset.target_key resolves to 'action'."
                    )
                abs_action_state_key = _resolve_feature_key(
                    features,
                    self.abs_action_state_key,
                )
            image_keys = [
                _resolve_feature_key(features, image_key)
                for image_key in self.image_keys
            ]

            split_indices = _parse_split(info.get("splits", {}).get(self.split_name))
            parquet_paths = sorted(
                (dataset_dir / "data").glob("chunk-*/episode_*.parquet"),
                key=_episode_index,
            )

            action_offset = len(self.actions)
            for parquet_path in parquet_paths:
                episode_index = _episode_index(parquet_path)
                if split_indices is not None and episode_index not in split_indices:
                    continue

                actions = self._load_actions(
                    parquet_path,
                    target_key=target_key,
                    abs_action_state_key=abs_action_state_key,
                )
                if actions.shape[0] == 0:
                    continue

                video_paths = tuple(
                    self._resolve_video_path(
                        dataset_dir,
                        info,
                        episode_index,
                        image_key,
                    )
                    for image_key in image_keys
                )
                length = actions.shape[0]
                episode = LiberoEpisode(
                    dataset_dir=dataset_dir,
                    parquet_path=parquet_path,
                    video_paths=video_paths,
                    episode_index=episode_index,
                    length=length,
                    action_offset=action_offset,
                )
                self.episodes.append(episode)
                self.actions.append(torch.from_numpy(actions).float())
                self.lengths.append(length)
                action_offset += 1

        if not self.episodes:
            raise RuntimeError(
                f"No LIBERO LeRobot episodes found for split={self.split_name!r}."
            )

        self.data_begin = np.cumsum([0] + self.lengths[:-1])
        self.data_end = np.cumsum(self.lengths)

    def compute_target_stats(
        self,
        indices: list[int] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = torch.cat(self.actions, dim=0)
        if indices is not None:
            targets = targets[torch.as_tensor(indices, dtype=torch.long)]
        return targets.mean(dim=0), targets.std(dim=0, unbiased=False)

    def _load_actions(
        self,
        parquet_path: Path,
        target_key: str,
        abs_action_state_key: str | None = None,
    ) -> np.ndarray:
        columns = [target_key]
        if abs_action_state_key is not None:
            columns.append(abs_action_state_key)

        table = pq.read_table(parquet_path, columns=columns)
        values = table[target_key].to_pylist()
        actions = np.asarray(values, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[:, None]

        if abs_action_state_key is not None:
            states = np.asarray(table[abs_action_state_key].to_pylist(), dtype=np.float32)
            if states.ndim == 1:
                states = states[:, None]
            actions = _delta_eef_action_to_absolute(actions, states, parquet_path)
        return actions

    def _resolve_video_path(
        self,
        dataset_dir: Path,
        info: dict,
        episode_index: int,
        image_key: str,
    ) -> Path:
        chunks_size = int(info.get("chunks_size", 1000))
        episode_chunk = episode_index // chunks_size
        template = info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        video_rel = template.format(
            episode_chunk=episode_chunk,
            video_key=image_key,
            episode_index=episode_index,
        )
        video_path = dataset_dir / video_rel
        if video_path.exists():
            return video_path

        matches = sorted(
            (dataset_dir / "videos").glob(
                f"chunk-*/{image_key}/episode_{episode_index:06d}.mp4"
            )
        )
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Missing video for episode {episode_index}: {video_path}")

    def __len__(self) -> int:
        return int(self.data_end[-1])

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

    def _load_video_frames(self, video_path: Path, frame_indices: list[int]):
        backends = {
            "av": self._load_video_frames_av,
            "decord": self._load_video_frames_decord,
            "cv2": self._load_video_frames_cv2,
        }
        if self.video_backend == "auto":
            errors = []
            for backend_name in ("av", "decord", "cv2"):
                try:
                    return backends[backend_name](video_path, frame_indices)
                except Exception as exc:
                    errors.append(f"{backend_name}: {exc}")
            raise RuntimeError(
                f"Failed to decode {video_path} with all video backends:\n"
                + "\n".join(errors)
            )

        if self.video_backend not in backends:
            raise ValueError(
                f"Unsupported video_backend={self.video_backend!r}. "
                "Use 'av', 'decord', 'cv2', or 'auto'."
            )
        return backends[self.video_backend](video_path, frame_indices)

    def _load_video_frames_decord(self, video_path: Path, frame_indices: list[int]):
        if decord is None:
            raise ImportError("decord is not installed.")
        frames = decord.VideoReader(video_path.as_posix()).get_batch(frame_indices)
        if hasattr(frames, "asnumpy"):
            frames = frames.asnumpy()
        elif isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        else:
            frames = np.asarray(frames)
        return [frame for frame in frames]

    def _load_video_frames_av(self, video_path: Path, frame_indices: list[int]):
        if av is None:
            raise ImportError("av is not installed.")
        wanted = set(frame_indices)
        frames = {}
        container = av.open(video_path.as_posix())
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
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

    def _load_video_frames_cv2(self, video_path: Path, frame_indices: list[int]):
        if cv2 is None:
            raise ImportError("opencv-python is not installed.")
        cap = cv2.VideoCapture(str(video_path))
        frames = []
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

    def _load_composed_frames(
        self,
        video_paths: tuple[Path, ...],
        frame_indices: list[int],
    ) -> list[np.ndarray]:
        frame_groups = [
            self._load_video_frames(video_path, frame_indices)
            for video_path in video_paths
        ]
        composed = []
        for frames_at_t in zip(*frame_groups):
            if self.view_resize_size is not None:
                frames_at_t = tuple(
                    self._resize_frame(frame, self.view_resize_size)
                    for frame in frames_at_t
                )
            heights = {frame.shape[0] for frame in frames_at_t}
            channels = {frame.shape[2] for frame in frames_at_t}
            if len(heights) != 1 or channels != {3}:
                shapes = [frame.shape for frame in frames_at_t]
                raise ValueError(f"Cannot concatenate view frames with shapes: {shapes}")
            composed.append(np.concatenate(frames_at_t, axis=1))
        return composed

    def _resize_frame(
        self,
        frame: np.ndarray,
        size: tuple[int, int],
    ) -> np.ndarray:
        height, width = size
        if frame.shape[:2] == (height, width):
            return frame
        image = Image.fromarray(frame)
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(image)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode_idx = int(np.searchsorted(self.data_end, idx, side="right"))
        local_idx = int(idx - self.data_begin[episode_idx])
        episode = self.episodes[episode_idx]
        image_indices = self._frame_indices(local_idx, episode.length)
        frames = self._load_composed_frames(episode.video_paths, image_indices)
        images = torch.stack(
            [self.preprocessor.process_image(frame) for frame in frames],
            dim=0,
        )
        actions = self.actions[episode.action_offset][local_idx].clone()

        metadata = {
            "dataset_dir": str(episode.dataset_dir),
            "parquet_path": str(episode.parquet_path),
            "video_path": str(episode.video_paths[0]),
            "video_paths": [str(path) for path in episode.video_paths],
            "episode_index": episode.episode_index,
            "frame_idx": local_idx,
            "split": self.split,
            "image_keys": self.image_keys,
            "target_key": self.target_key,
            "is_abs": self.is_abs,
            "abs_action_state_key": self.abs_action_state_key,
            "target_representation": self.target_representation,
        }
        return {
            "images": images,
            "mask": None,
            "actions": actions,
            "metadata": metadata,
        }

