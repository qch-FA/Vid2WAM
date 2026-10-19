from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


CSV_COLUMNS = [
    "video_path",
    "fps",
    "n_frames",
    "width",
    "height",
    "caption",
    "prompt_embed_path",
    "source_dataset",
    "episode_index",
]

DEFAULT_IMAGE_KEY = "observation.images.image"
DEFAULT_WRIST_IMAGE_KEY = "observation.images.wrist_image"
DEFAULT_VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_task_text(episode: dict[str, Any]) -> str:
    tasks = episode.get("tasks", [])
    if isinstance(tasks, list):
        for task in tasks:
            if isinstance(task, str) and task.strip():
                return task.strip()
    if isinstance(tasks, str):
        return tasks.strip()
    return ""


def get_feature_fps(info: dict[str, Any], video_key: str) -> float | None:
    feature = info.get("features", {}).get(video_key, {})
    video_info = feature.get("info", {})
    fps = video_info.get("video.fps")
    return float(fps) if fps else None


def resolve_video_path(
    dataset_dir: Path,
    info: dict[str, Any],
    episode_index: int,
    video_key: str,
) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    template = info.get("video_path", DEFAULT_VIDEO_PATH)
    relpath = template.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
        video_key=video_key,
    )
    return dataset_dir / relpath


def open_writer(path: Path, fps: float, width: int, height: int, codec: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    return writer


def read_video_info(path: Path) -> dict[str, Any] | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if n_frames <= 0 or width <= 0 or height <= 0:
        return None
    return {
        "fps": fps,
        "n_frames": n_frames,
        "width": width,
        "height": height,
    }


def resize_camera(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def convert_pair_with_opencv(
    image_path: Path,
    wrist_path: Path,
    output_path: Path,
    fps: float,
    camera_width: int,
    camera_height: int,
    codec: str,
) -> int:
    image_cap = cv2.VideoCapture(str(image_path))
    wrist_cap = cv2.VideoCapture(str(wrist_path))
    if not image_cap.isOpened() or not wrist_cap.isOpened():
        image_cap.release()
        wrist_cap.release()
        return 0

    writer = open_writer(
        output_path,
        fps=fps,
        width=camera_width * 2,
        height=camera_height,
        codec=codec,
    )
    n_frames = 0
    try:
        while True:
            ok_image, image = image_cap.read()
            ok_wrist, wrist = wrist_cap.read()
            if not ok_image or not ok_wrist:
                break
            image = resize_camera(image, camera_width, camera_height)
            wrist = resize_camera(wrist, camera_width, camera_height)
            writer.write(np.concatenate([image, wrist], axis=1))
            n_frames += 1
    finally:
        image_cap.release()
        wrist_cap.release()
        writer.release()
    return n_frames


def convert_pair_with_decord(
    image_path: Path,
    wrist_path: Path,
    output_path: Path,
    fps: float,
    camera_width: int,
    camera_height: int,
    codec: str,
) -> int:
    try:
        import decord
    except ImportError as exc:
        raise RuntimeError(
            f"OpenCV could not read {image_path} or {wrist_path}, and decord is not installed."
        ) from exc

    image_reader = decord.VideoReader(str(image_path))
    wrist_reader = decord.VideoReader(str(wrist_path))
    n_frames = min(len(image_reader), len(wrist_reader))
    if n_frames <= 0:
        return 0

    writer = open_writer(
        output_path,
        fps=fps,
        width=camera_width * 2,
        height=camera_height,
        codec=codec,
    )
    try:
        for frame_idx in range(n_frames):
            image = image_reader[frame_idx].asnumpy()
            wrist = wrist_reader[frame_idx].asnumpy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            wrist = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)
            image = resize_camera(image, camera_width, camera_height)
            wrist = resize_camera(wrist, camera_width, camera_height)
            writer.write(np.concatenate([image, wrist], axis=1))
    finally:
        writer.release()
    return n_frames


def resolve_ffmpeg_exe() -> str | None:
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        return ffmpeg_exe

    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def convert_pair_with_ffmpeg(
    image_path: Path,
    wrist_path: Path,
    output_path: Path,
    fps: float,
    camera_width: int,
    camera_height: int,
    ffmpeg_encoder: str,
) -> int:
    ffmpeg_exe = resolve_ffmpeg_exe()
    if ffmpeg_exe is None:
        raise RuntimeError(
            "OpenCV could not decode the LIBERO AV1 videos and no ffmpeg executable "
            "was found. Install ffmpeg or `pip install imageio-ffmpeg`, then rerun."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_complex = (
        f"[0:v]scale={camera_width}:{camera_height}:flags=lanczos,setsar=1[left];"
        f"[1:v]scale={camera_width}:{camera_height}:flags=lanczos,setsar=1[right];"
        "[left][right]hstack=inputs=2[v]"
    )

    encoders = [ffmpeg_encoder]
    if ffmpeg_encoder != "mpeg4":
        encoders.append("mpeg4")

    last_error = ""
    for encoder in encoders:
        cmd = [
            ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "none",
            "-i",
            str(image_path),
            "-i",
            str(wrist_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-an",
            "-r",
            str(fps),
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            encoder,
        ]
        if encoder == "libx264":
            cmd.extend(["-preset", "veryfast", "-crf", "18"])
        elif encoder == "mpeg4":
            cmd.extend(["-q:v", "4"])
        cmd.append(str(output_path))

        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode == 0:
            video_info = read_video_info(output_path)
            if video_info is None:
                raise RuntimeError(f"ffmpeg wrote an unreadable output video: {output_path}")
            return int(video_info["n_frames"])
        last_error = (proc.stderr or proc.stdout or "").strip()

    raise RuntimeError(
        f"ffmpeg failed to convert {image_path} and {wrist_path}. Last error:\n{last_error}"
    )


def convert_pair(
    image_path: Path,
    wrist_path: Path,
    output_path: Path,
    fps: float,
    camera_width: int,
    camera_height: int,
    codec: str,
    ffmpeg_encoder: str,
    backend: str,
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        existing_info = read_video_info(output_path)
        if existing_info is not None:
            return existing_info

    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image camera video: {image_path}")
    if not wrist_path.is_file():
        raise FileNotFoundError(f"Missing wrist_image camera video: {wrist_path}")

    n_frames = 0
    errors = []

    if backend in {"auto", "opencv"}:
        try:
            n_frames = convert_pair_with_opencv(
                image_path,
                wrist_path,
                output_path,
                fps=fps,
                camera_width=camera_width,
                camera_height=camera_height,
                codec=codec,
            )
        except Exception as exc:
            errors.append(f"OpenCV failed: {exc}")

    if n_frames == 0 and backend in {"auto", "ffmpeg"}:
        try:
            n_frames = convert_pair_with_ffmpeg(
                image_path,
                wrist_path,
                output_path,
                fps=fps,
                camera_width=camera_width,
                camera_height=camera_height,
                ffmpeg_encoder=ffmpeg_encoder,
            )
        except Exception as exc:
            errors.append(f"ffmpeg failed: {exc}")

    if n_frames == 0 and backend in {"auto", "decord"}:
        try:
            n_frames = convert_pair_with_decord(
                image_path,
                wrist_path,
                output_path,
                fps=fps,
                camera_width=camera_width,
                camera_height=camera_height,
                codec=codec,
            )
        except Exception as exc:
            errors.append(f"decord failed: {exc}")

    if n_frames == 0:
        detail = "\n".join(errors)
        raise RuntimeError(f"No frames were written for {output_path}\n{detail}")

    return {
        "fps": fps,
        "n_frames": n_frames,
        "width": camera_width * 2,
        "height": camera_height,
    }


def build_records_for_dataset(
    dataset_dir: Path,
    output_root: Path,
    image_key: str,
    wrist_image_key: str,
    camera_width: int,
    camera_height: int,
    codec: str,
    ffmpeg_encoder: str,
    backend: str,
    overwrite: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    info_path = dataset_dir / "meta" / "info.json"
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot info file: {info_path}")
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot episodes file: {episodes_path}")

    info = load_json(info_path)
    episodes = load_jsonl(episodes_path)
    if limit is not None:
        episodes = episodes[:limit]

    dataset_name = dataset_dir.name
    fps = get_feature_fps(info, image_key) or float(info.get("fps", 20))
    records = []
    for episode in tqdm(episodes, desc=dataset_name):
        episode_index = int(episode["episode_index"])
        image_path = resolve_video_path(dataset_dir, info, episode_index, image_key)
        wrist_path = resolve_video_path(dataset_dir, info, episode_index, wrist_image_key)

        rel_video_path = (
            Path("videos") / dataset_name / f"episode_{episode_index:06d}.mp4"
        )
        output_path = output_root / rel_video_path
        video_info = convert_pair(
            image_path=image_path,
            wrist_path=wrist_path,
            output_path=output_path,
            fps=fps,
            camera_width=camera_width,
            camera_height=camera_height,
            codec=codec,
            ffmpeg_encoder=ffmpeg_encoder,
            backend=backend,
            overwrite=overwrite,
        )

        records.append(
            {
                "video_path": rel_video_path.as_posix(),
                "fps": video_info["fps"],
                "n_frames": video_info["n_frames"],
                "width": video_info["width"],
                "height": video_info["height"],
                "caption": first_task_text(episode),
                "prompt_embed_path": rel_video_path.with_suffix(".pt").as_posix(),
                "source_dataset": dataset_name,
                "episode_index": episode_index,
            }
        )
    return records


def write_csv(records: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Prepare LIBERO LeRobot datasets for teacher SFT by resizing image and "
            "wrist_image videos to 224x224 and concatenating them as 224x448 mp4s."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        required=True,
        help="Path to an extracted LIBERO LeRobot dataset. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where concatenated teacher-SFT videos and metadata are written.",
    )
    parser.add_argument(
        "--metadata-path",
        default=None,
        help="Output CSV path. Defaults to <output-root>/libero_videos.csv.",
    )
    parser.add_argument("--image-key", default=DEFAULT_IMAGE_KEY)
    parser.add_argument("--wrist-image-key", default=DEFAULT_WRIST_IMAGE_KEY)
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument(
        "--ffmpeg-encoder",
        default="libx264",
        help="Encoder used by the ffmpeg fallback. Falls back to mpeg4 if unavailable.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "opencv", "ffmpeg", "decord"],
        default="auto",
        help="Video conversion backend. Use `ffmpeg` to avoid OpenCV/decord AV1 decoding.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-dataset episode limit for debugging.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dataset_dirs = [Path(p).expanduser().resolve() for p in args.dataset_dir]
    output_root = Path(args.output_root).expanduser().resolve()
    output_csv = (
        Path(args.metadata_path).expanduser().resolve()
        if args.metadata_path is not None
        else output_root / "libero_videos.csv"
    )

    all_records = []
    for dataset_dir in dataset_dirs:
        all_records.extend(
            build_records_for_dataset(
                dataset_dir=dataset_dir,
                output_root=output_root,
                image_key=args.image_key,
                wrist_image_key=args.wrist_image_key,
                camera_width=args.camera_width,
                camera_height=args.camera_height,
                codec=args.codec,
                ffmpeg_encoder=args.ffmpeg_encoder,
                backend=args.backend,
                overwrite=args.overwrite,
                limit=args.limit,
            )
        )

    if not all_records:
        raise RuntimeError("No LIBERO episodes were converted.")

    write_csv(all_records, output_csv)
    print(f"Successfully wrote {len(all_records)} records to {output_csv}")


if __name__ == "__main__":
    main()
