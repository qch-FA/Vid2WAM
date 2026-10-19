"""Synchronized three-camera RoboTwin LeRobot videos for teacher training."""
import torch
import torch.nn.functional as F

from .video_base import VideoDataset, decord
import av


class _VideoReader:
    """Use PyAV for codecs such as AV1 unsupported by Decord's bundled FFmpeg."""

    def __init__(self, path):
        self.path = str(path)
        try:
            self.reader = decord.VideoReader(self.path)
            self.length = len(self.reader)
        except (RuntimeError, decord.DECORDError):
            self.reader = None
            with av.open(self.path) as container:
                self.length = container.streams.video[0].frames
                if not self.length:
                    self.length = sum(1 for _ in container.decode(video=0))

    def __len__(self):
        return self.length

    def get_batch(self, indices):
        if self.reader is not None:
            try:
                frames = self.reader.get_batch(indices)
                return frames if isinstance(frames, torch.Tensor) else torch.from_numpy(frames.asnumpy())
            except (RuntimeError, decord.DECORDError):
                self.reader = None
        wanted = set(indices)
        frames = {}
        with av.open(self.path) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    frames[index] = torch.from_numpy(frame.to_ndarray(format="rgb24"))
                if index >= max(wanted):
                    break
        missing = wanted - frames.keys()
        if missing:
            raise ValueError(f"Missing frames {sorted(missing)} in {self.path}")
        return torch.stack([frames[index] for index in indices])


class RobotwinLerobotDataset(VideoDataset):
    camera_columns = ("video_path", "video_path_left_wrist", "video_path_right_wrist")

    def _check_record(self, record):
        for column in self.camera_columns:
            path = self.data_root / record[column]
            if not path.is_file():
                raise FileNotFoundError(f"Missing {column}: {path}")

    def _load_video(self, record):
        readers = [_VideoReader(self.data_root / record[key]) for key in self.camera_columns]
        lengths = [len(reader) for reader in readers]
        if len(set(lengths)) != 1:
            raise ValueError(f"Camera videos must have matching frame counts, got {lengths}")
        start = int(record.get("trim_start", 0))
        end = int(record.get("trim_end", lengths[0]))
        if not 0 <= start < end <= lengths[0]:
            raise ValueError(f"Invalid trim interval [{start}, {end}) for {lengths[0]} frames")
        indices = (start + self._temporal_sample(end - start, record["fps"])).tolist()
        views = []
        for reader, size in zip(readers, ((256, 320), (128, 160), (128, 160))):
            frames = reader.get_batch(indices)
            if not isinstance(frames, torch.Tensor):
                frames = torch.from_numpy(frames.asnumpy())
            frames = frames.permute(0, 3, 1, 2).float() / 255.0
            views.append(F.interpolate(frames, size=size, mode="bilinear", align_corners=False))
        frames = torch.cat((views[0], torch.cat(views[1:], dim=3)), dim=2)
        transform = self.augment_transforms if self.split == "training" else self.no_augment_transforms
        return self.img_normalize(transform(frames))
