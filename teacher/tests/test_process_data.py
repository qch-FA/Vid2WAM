import csv
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configurations"


def load_module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    module = importlib.import_module("experiments.process_data")
    return importlib.reload(module)


def compose_cfg(data_root: Path, new_data_root: Path):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=process_data",
                "dataset=robotwin",
                "algorithm=wan_i2v",
                f"dataset.data_root={data_root}",
                "dataset.metadata_path=robotwin_videos.csv",
                "wandb.mode=disabled",
                f"experiment.new_data_root={new_data_root}",
                "experiment.cache_prompt_embed.batch_size=2",
            ],
        )
    with open_dict(cfg):
        cfg.experiment._name = "process_data"
        cfg.dataset._name = "robotwin"
        cfg.algorithm._name = "wan_i2v"
    return cfg


def write_metadata_csv(path: Path):
    rows = [
        {
            "video_path": str(path.parent / "click_bell" / "demo_clean" / "video" / "episode0.mp4"),
            "fps": 30.0,
            "n_frames": 12,
            "width": 320,
            "height": 240,
            "caption": "Press the bell",
        },
        {
            "video_path": str(path.parent / "move_stapler_pad" / "demo_clean" / "video" / "episode1.mp4"),
            "fps": 30.0,
            "n_frames": 14,
            "width": 320,
            "height": 240,
            "caption": "Move the stapler",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video_path", "fps", "n_frames", "width", "height", "caption"],
        )
        writer.writeheader()
        writer.writerows(rows)


class FakePromptEncoder:
    def __init__(self):
        self.calls = []

    def predict(self, prompts):
        self.calls.append(list(prompts))
        outputs = []
        for idx, _ in enumerate(prompts):
            outputs.append(torch.full((2, 4), float(idx + 1)))
        return outputs


class ProcessDataTests(unittest.TestCase):
    def test_cache_prompt_embed_updates_robotwin_csv_in_place(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "RoboTwin" / "data"
            data_root.mkdir(parents=True, exist_ok=True)
            metadata_path = data_root / "robotwin_videos.csv"
            write_metadata_csv(metadata_path)

            cfg = compose_cfg(data_root, data_root)
            encoder = FakePromptEncoder()

            with mock.patch.object(module, "build_prompt_encoder", return_value=encoder):
                experiment = module.ProcessDataExperiment(cfg, data_root / "outputs")
                experiment.cache_prompt_embed()

            pt_path = data_root / "click_bell" / "demo_clean" / "video" / "episode0.pt"
            self.assertTrue(pt_path.exists())
            self.assertTrue(torch.equal(torch.load(pt_path), torch.full((2, 4), 1.0)))

            with metadata_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["prompt_embed_path"], "click_bell/demo_clean/video/episode0.pt")
            self.assertEqual(rows[1]["prompt_embed_path"], "move_stapler_pad/demo_clean/video/episode1.pt")
            self.assertEqual(encoder.calls, [["Press the bell", "Move the stapler"]])

    def test_cache_prompt_embed_preserves_relative_layout_under_new_data_root(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_root = tmp / "RoboTwin" / "data"
            new_root = tmp / "processed"
            data_root.mkdir(parents=True, exist_ok=True)
            metadata_path = data_root / "robotwin_videos.csv"
            write_metadata_csv(metadata_path)

            cfg = compose_cfg(data_root, new_root)
            encoder = FakePromptEncoder()

            with mock.patch.object(module, "build_prompt_encoder", return_value=encoder):
                experiment = module.ProcessDataExperiment(cfg, tmp / "outputs")
                experiment.cache_prompt_embed()

            new_metadata = new_root / "robotwin_videos.csv"
            new_pt = new_root / "click_bell" / "demo_clean" / "video" / "episode0.pt"
            self.assertTrue(new_metadata.exists())
            self.assertTrue(new_pt.exists())

            with new_metadata.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["prompt_embed_path"], "click_bell/demo_clean/video/episode0.pt")
            self.assertEqual(rows[1]["prompt_embed_path"], "move_stapler_pad/demo_clean/video/episode1.pt")


if __name__ == "__main__":
    unittest.main()
