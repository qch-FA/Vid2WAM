import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "datasets" / "prepare_libero_sft_data.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_libero_sft_data", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_video(path: Path, color_bgr, num_frames: int = 4, size=(12, 10), fps: int = 20):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    frame[:, :] = np.asarray(color_bgr, dtype=np.uint8)
    for _ in range(num_frames):
        writer.write(frame)
    writer.release()


def write_lerobot_meta(dataset_dir: Path):
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "fps": 20,
        "chunks_size": 1000,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.image": {"info": {"video.fps": 20}},
            "observation.images.wrist_image": {"info": {"video.fps": 20}},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta_dir / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "tasks": ["put the bowl on the plate"],
                "length": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )


class PrepareLiberoSftDataTests(unittest.TestCase):
    def test_resizes_and_concats_image_then_wrist_image(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_dir = tmp / "libero_goal_no_noops_lerobot"
            output_root = tmp / "teacher_sft"
            output_csv = output_root / "libero_videos.csv"

            write_lerobot_meta(dataset_dir)
            write_video(
                dataset_dir
                / "videos/chunk-000/observation.images.image/episode_000000.mp4",
                color_bgr=(10, 20, 230),
            )
            write_video(
                dataset_dir
                / "videos/chunk-000/observation.images.wrist_image/episode_000000.mp4",
                color_bgr=(20, 220, 30),
            )

            module.main(
                [
                    "--dataset-dir",
                    str(dataset_dir),
                    "--output-root",
                    str(output_root),
                    "--metadata-path",
                    str(output_csv),
                ]
            )

            with output_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["video_path"], "videos/libero_goal_no_noops_lerobot/episode_000000.mp4")
            self.assertEqual(row["prompt_embed_path"], "videos/libero_goal_no_noops_lerobot/episode_000000.pt")
            self.assertEqual(row["width"], "448")
            self.assertEqual(row["height"], "224")
            self.assertEqual(row["n_frames"], "4")
            self.assertEqual(row["caption"], "put the bowl on the plate")

            cap = cv2.VideoCapture(str(output_root / row["video_path"]))
            ok, frame = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape[:2], (224, 448))

            left_mean = frame[:, :224].mean(axis=(0, 1))
            right_mean = frame[:, 224:].mean(axis=(0, 1))
            self.assertGreater(left_mean[2], 150)
            self.assertGreater(right_mean[1], 150)


if __name__ == "__main__":
    unittest.main()
