import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "datasets" / "generate_robotwin_csv.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_robotwin_csv", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_test_video(path: Path, num_frames: int = 3, width: int = 8, height: int = 6, fps: int = 12):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open test video writer for {path}")
    for frame_id in range(num_frames):
        frame = np.full((height, width, 3), frame_id * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class GenerateRobotwinCsvTests(unittest.TestCase):
    def test_main_scans_demo_clean_and_writes_expected_columns(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "RoboTwin" / "data"
            video_path = root / "click_bell" / "demo_clean" / "video" / "episode0.mp4"
            instruction_path = root / "click_bell" / "demo_clean" / "instructions" / "episode0.json"
            output_csv = root / "robotwin_videos.csv"

            write_test_video(video_path, num_frames=4, width=10, height=8, fps=15)
            instruction_path.parent.mkdir(parents=True, exist_ok=True)
            instruction_path.write_text(
                json.dumps({"seen": ["Press the bell"], "unseen": ["Unused fallback"]}),
                encoding="utf-8",
            )

            module.main(
                [
                    "--root-dir",
                    str(root),
                    "--output-csv",
                    str(output_csv),
                ]
            )

            with output_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["video_path"], str(video_path))
            self.assertEqual(row["caption"], "Press the bell")
            self.assertEqual(
                row["prompt_embed_path"],
                "click_bell/demo_clean/video/episode0.pt",
            )
            self.assertEqual(row["n_frames"], "4")
            self.assertEqual(row["width"], "10")
            self.assertEqual(row["height"], "8")

    def test_main_supports_non_vidar_variant_names_and_unseen_fallback(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "RoboTwin" / "data"
            video_path = root / "move_stapler_pad" / "demo_clean_v2" / "video" / "episode3.mp4"
            instruction_path = root / "move_stapler_pad" / "demo_clean_v2" / "instructions" / "episode3.json"
            output_csv = root / "robotwin_videos.csv"

            write_test_video(video_path)
            instruction_path.parent.mkdir(parents=True, exist_ok=True)
            instruction_path.write_text(
                json.dumps({"seen": [], "unseen": ["Fallback prompt"]}),
                encoding="utf-8",
            )

            module.main(
                [
                    "--root-dir",
                    str(root),
                    "--output-csv",
                    str(output_csv),
                ]
            )

            with output_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["caption"], "Fallback prompt")
            self.assertEqual(
                rows[0]["prompt_embed_path"],
                "move_stapler_pad/demo_clean_v2/video/episode3.pt",
            )


if __name__ == "__main__":
    unittest.main()
