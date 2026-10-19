import argparse
import csv
import json
import re
from pathlib import Path

from tqdm import tqdm


CAMERA_HIGH = "observation.images.cam_high"
CAMERA_LEFT_WRIST = "observation.images.cam_left_wrist"
CAMERA_RIGHT_WRIST = "observation.images.cam_right_wrist"

FRAME_WIDTH = 320
FRAME_HEIGHT = 384

CSV_COLUMNS = [
    "video_path",
    "video_path_left_wrist",
    "video_path_right_wrist",
    "fps",
    "n_frames",
    "width",
    "height",
    "caption",
    "prompt_embed_path",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_episodes(path: Path) -> dict[int, dict]:
    episodes = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            episodes[int(episode["episode_index"])] = episode
    return episodes


def first_task_text(episode: dict) -> str:
    tasks = episode.get("tasks", [])
    if isinstance(tasks, str):
        return tasks.strip()
    if isinstance(tasks, list):
        for task in tasks:
            if isinstance(task, str) and task.strip():
                return task.strip()
    return ""


def parse_episode_index(video_path: Path) -> int:
    match = re.fullmatch(r"episode_(\d+)", video_path.stem)
    if match is None:
        raise ValueError(f"Cannot parse episode index from {video_path}")
    return int(match.group(1))


def companion_path(high_path: Path, camera_key: str, root_dir: Path) -> str:
    path = high_path.parent.parent / camera_key / high_path.name
    if not path.is_file():
        raise FileNotFoundError(f"Missing companion camera video: {path}")
    return path.relative_to(root_dir).as_posix()


def collect_records(root_dir: Path) -> list[dict]:
    info_path = root_dir / "meta" / "info.json"
    episodes_path = root_dir / "meta" / "episodes.jsonl"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {episodes_path}")

    info = load_json(info_path)
    episodes = load_episodes(episodes_path)
    fps = float(info.get("fps", 50))
    high_videos = sorted(
        (root_dir / "videos").glob(f"chunk-*/{CAMERA_HIGH}/episode_*.mp4")
    )
    print(f"Found {len(high_videos)} RoboTwin LeRobot episodes")

    records = []
    for high_path in tqdm(high_videos, desc="Preparing RoboTwin SFT metadata"):
        episode_index = parse_episode_index(high_path)
        if episode_index not in episodes:
            raise KeyError(f"Episode {episode_index} is missing from {episodes_path}")
        episode = episodes[episode_index]
        high_relpath = high_path.relative_to(root_dir)
        records.append(
            {
                "video_path": high_relpath.as_posix(),
                "video_path_left_wrist": companion_path(
                    high_path, CAMERA_LEFT_WRIST, root_dir
                ),
                "video_path_right_wrist": companion_path(
                    high_path, CAMERA_RIGHT_WRIST, root_dir
                ),
                "fps": fps,
                "n_frames": int(episode["length"]),
                "width": FRAME_WIDTH,
                "height": FRAME_HEIGHT,
                "caption": first_task_text(episode),
                "prompt_embed_path": high_relpath.with_suffix(".pt").as_posix(),
            }
        )
    return records


def write_csv(records: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Prepare RoboTwin LeRobot metadata for teacher SFT. The three camera "
            "videos remain in place and are stitched to 384x320 by the dataloader."
        )
    )
    parser.add_argument("--root-dir", required=True)
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Defaults to <root-dir>/robotwin_videos.csv.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root_dir = Path(args.root_dir).expanduser().resolve()
    output_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv is not None
        else root_dir / "robotwin_videos.csv"
    )
    records = collect_records(root_dir)
    if not records:
        raise RuntimeError(f"No cam_high videos found under {root_dir / 'videos'}")
    write_csv(records, output_csv)
    print(f"Successfully wrote {len(records)} records to {output_csv}")


if __name__ == "__main__":
    main()
