import torch
import torch.nn as nn

from .resnet import ResNet, Resnet_plus
from .unet import UNet


class IDM(nn.Module):
    def __init__(self, model_name, train_mean=None, train_std=None, *args, **kwargs):
        super().__init__()
        self.output_dim = int(kwargs.get("output_dim", 14))
        if model_name == "mask":
            self.model = Mask(*args, **kwargs)
        elif model_name == "unet":
            self.model = Unet(*args, **kwargs)
        elif model_name == "resnet":
            self.model = ResNet(*args, **kwargs)
        elif model_name == "resnet_plus":
            self.model = Resnet_plus(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        self.register_buffer(
            "train_mean",
            self._make_stats_tensor(train_mean, default_value=0.0),
        )
        self.register_buffer(
            "train_std",
            self._make_stats_tensor(train_std, default_value=1.0).clamp_min(1e-8),
        )

    def _make_stats_tensor(self, values, default_value: float) -> torch.Tensor:
        if values is None or (isinstance(values, str) and values == "auto"):
            return torch.full((self.output_dim,), default_value, dtype=torch.float32)
        if isinstance(values, torch.Tensor):
            tensor = values.detach().clone().float()
        else:
            tensor = torch.tensor(list(values), dtype=torch.float32)
        tensor = tensor.reshape(-1)
        if tensor.numel() != self.output_dim:
            raise ValueError(
                f"IDM normalization stats must have length {self.output_dim}, "
                f"got {tensor.numel()}."
            )
        return tensor

    def set_normalization_stats(self, train_mean, train_std) -> None:
        device = self.train_mean.device
        self.train_mean = self._make_stats_tensor(
            train_mean,
            default_value=0.0,
        ).to(device)
        self.train_std = (
            self._make_stats_tensor(train_std, default_value=1.0)
            .clamp_min(1e-8)
            .to(device)
        )

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.train_mean) / (self.train_std + 1e-8)

    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.train_std + self.train_mean

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if isinstance(output, tuple):
            return self.denormalize(output[0]), *output[1:]
        return self.denormalize(output)


class Unet(nn.Module):
    def __init__(self, output_dim: int = 14, *args, **kwargs):
        super().__init__()
        self.mask_model = UNet(3, 3)
        self.resnet_model = ResNet(14, 3)

    def forward(self, images, *args, **kwargs):
        return self.resnet_model(torch.tanh(self.mask_model(images)))


class Mask(nn.Module):
    def __init__(self, output_dim: int = 14, *args, **kwargs):
        super().__init__()
        self.mask_model = UNet(3, 1)
        self.resnet_model = ResNet(14, *args, **kwargs)

    def forward(self, images, return_mask=False, *args, **kwargs):
        mask = (1 + torch.tanh(self.mask_model(images))) / 2
        mask_hard = torch.where(mask < 0.5, 0.0, 1.0)
        inputs = images * ((mask_hard - mask).detach() + mask)
        outputs = self.resnet_model(inputs)
        if return_mask:
            return outputs, mask
        return outputs
