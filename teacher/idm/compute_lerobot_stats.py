#!/usr/bin/env python3
"""Compute mean/std for numeric fields in LeRobot-format datasets.

Example:
    python3 idm/compute_lerobot_stats.py \
        /path/to/libero_10_no_noops_lerobot \
        /path/to/libero_spatial_no_noops_lerobot \
        --field action
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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


def _as_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _iter_episode_stats(dataset_dir: Path, field: str, split_name: str):
    info_path = dataset_dir / "meta" / "info.json"
    stats_path = dataset_dir / "meta" / "episodes_stats.jsonl"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing {stats_path}")

    info = _load_json(info_path)
    if field not in info.get("features", {}):
        available = ", ".join(sorted(info.get("features", {})))
        raise KeyError(f"{field!r} is not in {info_path}. Available fields: {available}")

    split_indices = _parse_split(info.get("splits", {}).get(split_name))
    with stats_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            episode_index = int(record["episode_index"])
            if split_indices is not None and episode_index not in split_indices:
                continue
            stats = record.get("stats", {}).get(field)
            if stats is None:
                raise KeyError(
                    f"{field!r} is missing in episode {episode_index} of {stats_path}"
                )
            yield episode_index, stats


def compute_stats(dataset_dirs: list[Path], field: str, split_name: str) -> dict:
    total_count = 0
    sum_values = None
    sum_squares = None
    min_values = None
    max_values = None
    episode_count = 0

    for dataset_dir in dataset_dirs:
        for _episode_index, stats in _iter_episode_stats(dataset_dir, field, split_name):
            count = int(_as_array(stats["count"])[0])
            mean = _as_array(stats["mean"])
            std = _as_array(stats["std"])
            min_value = _as_array(stats["min"])
            max_value = _as_array(stats["max"])

            if sum_values is None:
                sum_values = np.zeros_like(mean)
                sum_squares = np.zeros_like(mean)
                min_values = min_value.copy()
                max_values = max_value.copy()
            elif mean.shape != sum_values.shape:
                raise ValueError(
                    f"Field {field!r} has inconsistent shape: expected "
                    f"{sum_values.shape}, got {mean.shape} in {dataset_dir}"
                )

            total_count += count
            episode_count += 1
            sum_values += count * mean
            sum_squares += count * (np.square(std) + np.square(mean))
            min_values = np.minimum(min_values, min_value)
            max_values = np.maximum(max_values, max_value)

    if total_count == 0 or sum_values is None or sum_squares is None:
        raise ValueError(f"No episodes found for split={split_name!r}.")

    mean = sum_values / total_count
    variance = np.maximum(sum_squares / total_count - np.square(mean), 0.0)
    std = np.sqrt(variance)

    return {
        "field": field,
        "split": split_name,
        "datasets": [str(path) for path in dataset_dirs],
        "episodes": episode_count,
        "frames": total_count,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": min_values.tolist(),
        "max": max_values.tolist(),
    }


def _format_tensor(name: str, values: list[float]) -> str:
    formatted = ",\n        ".join(f"{value:.8g}" for value in values)
    return f"{name} = torch.tensor(\n    [\n        {formatted},\n    ]\n)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute LeRobot field mean/std from meta/episodes_stats.jsonl."
    )
    parser.add_argument(
        "dataset_dirs",
        nargs="+",
        type=Path,
        help="One or more LeRobot dataset directories.",
    )
    parser.add_argument(
        "--field",
        default="action",
        help='Feature field to compute, e.g. "action" or "observation.state".',
    )
    parser.add_argument(
        "--split",
        default="train",
        help='Dataset split in meta/info.json. Defaults to "train".',
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of torch.tensor snippets.",
    )
    args = parser.parse_args()

    stats = compute_stats(args.dataset_dirs, args.field, args.split)
    if args.json:
        print(json.dumps(stats, indent=2))
        return

    print(f"field: {stats['field']}")
    print(f"split: {stats['split']}")
    print(f"datasets: {len(stats['datasets'])}")
    print(f"episodes: {stats['episodes']}")
    print(f"frames: {stats['frames']}")
    print(_format_tensor("train_mean", stats["mean"]))
    print(_format_tensor("train_std", stats["std"]))


if __name__ == "__main__":
    main()
