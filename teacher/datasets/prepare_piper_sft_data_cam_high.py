from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
import multiprocessing as mp

import cv2
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

DEFAULT_CAM_HIGH_KEY = "observation.images.cam_high"
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
            if isinstance(task, dict):
                if isinstance(task.get("task"), str) and task["task"].strip():
                    return task["task"].strip()
                if isinstance(task.get("language_instruction"), str) and task["language_instruction"].strip():
                    return task["language_instruction"].strip()
    if isinstance(tasks, str):
        return tasks.strip()
    if isinstance(episode.get("task"), str):
        return episode["task"].strip()
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


def read_video_info(path: Path) -> dict[str, Any] | None:
    ffprobe_exe = shutil.which("ffprobe")
    if ffprobe_exe is not None:
        cmd = [
            ffprobe_exe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate",
            "-of",
            "json",
            str(path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0:
            try:
                data = json.loads(proc.stdout)
                streams = data.get("streams", [])
                if streams:
                    stream = streams[0]
                    width = int(stream.get("width", 0) or 0)
                    height = int(stream.get("height", 0) or 0)
                    nb_frames = int(stream.get("nb_frames", 0) or 0)
                    rate = stream.get("r_frame_rate", "0/1")
                    num, den = rate.split("/")
                    fps = float(num) / max(float(den), 1.0)
                    if nb_frames > 0 and width > 0 and height > 0:
                        return {
                            "fps": fps,
                            "n_frames": nb_frames,
                            "width": width,
                            "height": height,
                        }
            except Exception:
                pass

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


def resolve_ffmpeg_exe() -> str | None:
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        return ffmpeg_exe
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run_ffmpeg_convert(
    ffmpeg_exe: str,
    cam_high_path: Path,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    ffmpeg_encoder: str,
    threads: int,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
    )
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(cam_high_path),
        "-vf",
        vf,
        "-r",
        str(fps),
        "-an",
        "-c:v",
        ffmpeg_encoder,
        "-threads",
        str(threads),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    try:
        return subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            cmd, returncode=-1, stdout=exc.stdout or "", stderr=f"TIMEOUT after {timeout}s (likely NVENC session contention)"
        )


def convert_single_view_with_ffmpeg(
    cam_high_path: Path,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    ffmpeg_encoder: str,
    threads: int = 1,
    timeout: float = 60.0,
) -> tuple[int, dict[str, float]]:
    ffmpeg_exe = resolve_ffmpeg_exe()
    if ffmpeg_exe is None:
        raise RuntimeError("ffmpeg is unavailable and imageio-ffmpeg is not installed.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    t0 = time.monotonic()
    proc = _run_ffmpeg_convert(
        ffmpeg_exe, cam_high_path, output_path, fps, width, height, ffmpeg_encoder, threads, timeout
    )
    timings["encode_primary_s"] = time.monotonic() - t0

    if proc.returncode != 0 and ffmpeg_encoder != "libx264":
        stderr_lower = proc.stderr.lower()
        if (
            "nvenc" in stderr_lower
            or "libnvidia-encode" in stderr_lower
            or "timeout" in stderr_lower
        ):
            t1 = time.monotonic()
            proc = _run_ffmpeg_convert(
                ffmpeg_exe, cam_high_path, output_path, fps, width, height, "libx264", threads, timeout
            )
            timings["encode_fallback_s"] = time.monotonic() - t1
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to convert {cam_high_path}\n{proc.stderr}")

    t2 = time.monotonic()
    info = read_video_info(output_path)
    timings["read_info_s"] = time.monotonic() - t2
    if info is None:
        raise RuntimeError(f"Failed to read generated video info from {output_path}")
    return int(info["n_frames"]), timings


def _process_episode(job: dict[str, Any]) -> dict[str, Any]:
    episode_index = int(job["episode"]["episode_index"])
    try:
        record = _process_episode_inner(job)
        return {"ok": True, **record}
    except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the whole batch
        print(
            f"[FAILED] episode_{episode_index:06d}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return {"ok": False, "episode_index": episode_index, "error": str(exc)}


def _process_episode_inner(job: dict[str, Any]) -> dict[str, Any]:
    dataset_name = job["dataset_name"]
    episode = job["episode"]
    dataset_dir = Path(job["dataset_dir"])
    output_root = Path(job["output_root"])
    info = job["info"]
    cam_high_key = job["cam_high_key"]
    fps = float(job["fps"])
    width = int(job["width"])
    height = int(job["height"])
    ffmpeg_encoder = job["ffmpeg_encoder"]
    overwrite = bool(job["overwrite"])
    threads = int(job.get("threads", 1))
    timeout = float(job.get("timeout", 60.0))
    preserve_source_resolution = bool(job.get("preserve_source_resolution", False))

    episode_index = int(episode["episode_index"])
    t_start = time.monotonic()
    cam_high_path = resolve_video_path(dataset_dir, info, episode_index, cam_high_key)
    if not cam_high_path.is_file():
        raise FileNotFoundError(f"Missing cam_high camera video: {cam_high_path}")

    rel_video_path = Path("videos") / dataset_name / f"episode_{episode_index:06d}.mp4"
    output_path = output_root / rel_video_path
    timings: dict[str, float] = {}
    if preserve_source_resolution:
        source_info = read_video_info(cam_high_path)
        if source_info is None:
            raise RuntimeError(f"Failed to read source video info from {cam_high_path}")

        video_info = read_video_info(output_path) if output_path.exists() and not overwrite else None
        output_matches_source = video_info is not None and all(
            video_info[key] == source_info[key]
            for key in ("n_frames", "width", "height")
        )
        if not output_matches_source:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cam_high_path, output_path)
            video_info = source_info
    elif output_path.exists() and not overwrite:
        t_check = time.monotonic()
        video_info = read_video_info(output_path)
        timings["check_existing_s"] = time.monotonic() - t_check
        if video_info is None:
            output_path.unlink(missing_ok=True)
            n_frames, timings = convert_single_view_with_ffmpeg(
                cam_high_path=cam_high_path,
                output_path=output_path,
                fps=fps,
                width=width,
                height=height,
                ffmpeg_encoder=ffmpeg_encoder,
                threads=threads,
                timeout=timeout,
            )
            video_info = {
                "fps": fps,
                "n_frames": n_frames,
                "width": width,
                "height": height,
            }
    else:
        n_frames, timings = convert_single_view_with_ffmpeg(
            cam_high_path=cam_high_path,
            output_path=output_path,
            fps=fps,
            width=width,
            height=height,
            ffmpeg_encoder=ffmpeg_encoder,
            threads=threads,
            timeout=timeout,
        )
        video_info = {
            "fps": fps,
            "n_frames": n_frames,
            "width": width,
            "height": height,
        }

    total_s = time.monotonic() - t_start
    if total_s > 15.0:
        print(
            f"[SLOW] episode_{episode_index:06d}: total={total_s:.1f}s breakdown={timings}",
            file=sys.stderr,
            flush=True,
        )

    return {
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

def build_records_for_dataset(
    dataset_dir: Path,
    output_root: Path,
    cam_high_key: str,
    width: int,
    height: int,
    ffmpeg_encoder: str,
    overwrite: bool,
    limit: int | None,
    num_workers: int,
    threads_per_worker: int,
    timeout: float,
    preserve_source_resolution: bool,
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
    fps = get_feature_fps(info, cam_high_key) or float(info.get("fps", 10))
    features = info.get("features", {})
    if cam_high_key not in features:
        available = ", ".join(sorted(features))
        raise KeyError(f"{cam_high_key!r} not found in {info_path}; available: {available}")

    jobs = [
        {
            "dataset_name": dataset_name,
            "episode": episode,
            "dataset_dir": str(dataset_dir),
            "output_root": str(output_root),
            "info": info,
            "cam_high_key": cam_high_key,
            "fps": fps,
            "width": width,
            "height": height,
            "ffmpeg_encoder": ffmpeg_encoder,
            "overwrite": overwrite,
            "threads": threads_per_worker,
            "timeout": timeout,
            "preserve_source_resolution": preserve_source_resolution,
        }
        for episode in episodes
    ]

    if num_workers <= 1:
        return [
            _process_episode(job)
            for job in tqdm(jobs, desc=dataset_name)
        ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        return list(
            tqdm(
                pool.imap_unordered(_process_episode, jobs),
                total=len(jobs),
                desc=dataset_name,
            )
        )


def write_csv(records: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the Piper real-robot LeRobot dataset for teacher SFT by "
            "extracting only observation.images.cam_high."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        required=True,
        help="Path to an extracted Piper LeRobot dataset. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where single-view teacher-SFT videos and metadata are written.",
    )
    parser.add_argument(
        "--metadata-path",
        default=None,
        help="Output CSV path. Defaults to <output-root>/piper_videos.csv.",
    )
    parser.add_argument("--cam-high-key", default=DEFAULT_CAM_HIGH_KEY)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument(
        "--preserve-source-resolution",
        action="store_true",
        help=(
            "Copy the source cam_high MP4 without transcoding, resizing, or padding. "
            "The CSV records the source video's native dimensions."
        ),
    )
    parser.add_argument(
        "--ffmpeg-encoder",
        default="libx264",
        help="Encoder used by ffmpeg.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (mp.cpu_count() or 1) // 4),
        help="Number of worker processes for per-episode ffmpeg conversion.",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=None,
        help=(
            "ffmpeg -threads per worker process. Defaults to "
            "max(1, cpu_count // num_workers) so num_workers * threads_per_worker "
            "does not oversubscribe the machine's CPU cores."
        ),
    )
    parser.add_argument(
        "--encode-timeout",
        type=float,
        default=60.0,
        help=(
            "Per-attempt ffmpeg timeout in seconds. If the primary encoder "
            "(e.g. h264_nvenc under NVENC session contention) hangs past this, "
            "the attempt is killed and treated as a failure so the libx264 "
            "fallback can run instead of stalling indefinitely."
        ),
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
        else output_root / "piper_videos.csv"
    )

    all_records = []
    threads_per_worker = args.threads_per_worker
    if threads_per_worker is None:
        threads_per_worker = max(1, (mp.cpu_count() or 1) // max(1, args.num_workers))
    print(
        f"num_workers={args.num_workers} threads_per_worker={threads_per_worker} "
        f"(cpu_count={mp.cpu_count()}) encoder={args.ffmpeg_encoder} timeout={args.encode_timeout}s",
        file=sys.stderr,
    )
    for dataset_dir in dataset_dirs:
        all_records.extend(
            build_records_for_dataset(
                dataset_dir=dataset_dir,
                output_root=output_root,
                cam_high_key=args.cam_high_key,
                width=args.width,
                height=args.height,
                ffmpeg_encoder=args.ffmpeg_encoder,
                overwrite=args.overwrite,
                limit=args.limit,
                num_workers=args.num_workers,
                threads_per_worker=threads_per_worker,
                timeout=args.encode_timeout,
                preserve_source_resolution=args.preserve_source_resolution,
            )
        )

    if not all_records:
        raise RuntimeError("No Piper episodes were converted.")

    successes = [r for r in all_records if r.get("ok")]
    failures = [r for r in all_records if not r.get("ok")]
    for record in successes:
        record.pop("ok", None)

    if not successes:
        raise RuntimeError(
            f"All {len(failures)} episodes failed to convert. "
            f"See [FAILED] lines above for details."
        )

    write_csv(successes, output_csv)
    print(f"Successfully wrote {len(successes)} records to {output_csv}")

    if failures:
        failed_indices = sorted(f["episode_index"] for f in failures)
        print(
            f"WARNING: {len(failures)} episode(s) FAILED and were excluded from the CSV: "
            f"{failed_indices}",
            file=sys.stderr,
        )
        print(
            "Re-run the exact same command (without --overwrite) to retry only the "
            "failed/missing episodes -- already-converted episodes are skipped.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
