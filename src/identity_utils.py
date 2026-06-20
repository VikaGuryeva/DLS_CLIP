from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


class ArcFaceEncoder(torch.nn.Module):
    """
    Дифференцируемая обёртка над официальным ArcFace backbone.

    Параметры ArcFace заморожены, но encode_images() не использует
    torch.no_grad(), поэтому gradient может пройти от identity loss
    через ArcFace и StyleGAN2 к редактируемому W+.
    """

    def __init__(
        self,
        model_path: str | Path,
        arcface_torch_dir: str | Path,
        device: torch.device | str,
        backbone_name: str = "r50",
        input_size: int = 112,
        embedding_size: int = 512,
        crop_box: tuple[float, float, float, float] | None = None,
        strict: bool = True,
    ) -> None:
        super().__init__()

        self.model_path = Path(model_path)
        self.arcface_torch_dir = Path(arcface_torch_dir)
        self.device_name = torch.device(device)
        self.backbone_name = backbone_name
        self.input_size = int(input_size)
        self.embedding_size = int(embedding_size)
        self.crop_box = crop_box

        self._validate_init_arguments()

        get_model = self._load_backbone_factory()

        self.model = get_model(
            self.backbone_name,
            fp16=False,
            num_features=self.embedding_size,
        )

        state_dict = self._load_state_dict(
            self.model_path
        )

        self.model.load_state_dict(
            state_dict,
            strict=strict,
        )

        self.model.eval()
        self.model.requires_grad_(False)
        self.to(self.device_name)

    def _validate_init_arguments(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ArcFace checkpoint was not found: {self.model_path}"
            )

        backbones_init = (
            self.arcface_torch_dir
            / "backbones"
            / "__init__.py"
        )

        if not backbones_init.exists():
            raise FileNotFoundError(
                "Official ArcFace PyTorch backbones were not found: "
                f"{backbones_init}"
            )

        if self.input_size < 1:
            raise ValueError("input_size must be positive.")

        if self.embedding_size < 1:
            raise ValueError("embedding_size must be positive.")

        if self.crop_box is not None:
            if len(self.crop_box) != 4:
                raise ValueError(
                    "crop_box must contain four values: "
                    "(left, top, right, bottom)."
                )

            left, top, right, bottom = self.crop_box

            if not (
                0.0 <= left < right <= 1.0
                and 0.0 <= top < bottom <= 1.0
            ):
                raise ValueError(
                    "crop_box coordinates must satisfy "
                    "0 <= left < right <= 1 and "
                    "0 <= top < bottom <= 1."
                )

    def _load_backbone_factory(self):
        arcface_path = str(self.arcface_torch_dir)

        if arcface_path not in sys.path:
            sys.path.insert(0, arcface_path)

        backbones_module = importlib.import_module(
            "backbones"
        )

        module_path = Path(
            backbones_module.__file__
        ).resolve()

        expected_root = self.arcface_torch_dir.resolve()

        if expected_root not in module_path.parents:
            raise ImportError(
                "Imported an unexpected 'backbones' package. "
                f"Expected it inside {expected_root}, "
                f"but loaded {module_path}."
            )

        if not hasattr(backbones_module, "get_model"):
            raise AttributeError(
                "Official ArcFace backbones package "
                "does not contain get_model()."
            )

        return backbones_module.get_model

    @staticmethod
    def _load_state_dict(
        model_path: Path,
    ) -> dict[str, torch.Tensor]:
        try:
            checkpoint: Any = torch.load(
                model_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(
                model_path,
                map_location="cpu",
            )

        if not isinstance(checkpoint, dict):
            raise TypeError(
                "ArcFace checkpoint must be a dictionary."
            )

        state_dict = checkpoint

        for key in ("state_dict", "model", "backbone"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                state_dict = candidate
                break

        cleaned_state_dict: dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue

            clean_key = key

            if clean_key.startswith("module."):
                clean_key = clean_key[len("module."):]

            cleaned_state_dict[clean_key] = value

        if not cleaned_state_dict:
            raise ValueError(
                "The ArcFace checkpoint does not contain tensor weights."
            )

        return cleaned_state_dict

    def _apply_fixed_crop(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        if self.crop_box is None:
            return images

        _, _, height, width = images.shape
        left, top, right, bottom = self.crop_box

        x1 = int(round(left * width))
        y1 = int(round(top * height))
        x2 = int(round(right * width))
        y2 = int(round(bottom * height))

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        return images[:, :, y1:y2, x1:x2]

    def preprocess_images(
        self,
        stylegan_images: torch.Tensor,
    ) -> torch.Tensor:
        if stylegan_images.ndim != 4:
            raise ValueError(
                "Expected images with shape [B, C, H, W], "
                f"got {tuple(stylegan_images.shape)}."
            )

        if stylegan_images.shape[1] != 3:
            raise ValueError(
                "ArcFace expects RGB images with three channels."
            )

        if not torch.is_floating_point(stylegan_images):
            raise TypeError(
                "Images must be floating-point tensors."
            )

        images = stylegan_images.to(self.device_name)
        images = images.clamp(min=-1.0, max=1.0)
        images = self._apply_fixed_crop(images)

        images = F.interpolate(
            images,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        return images

    def encode_images(
        self,
        stylegan_images: torch.Tensor,
    ) -> torch.Tensor:
        model_input = self.preprocess_images(
            stylegan_images
        )

        identity_features = self.model(
            model_input
        )

        return F.normalize(
            identity_features,
            p=2,
            dim=-1,
        )

    def forward(
        self,
        stylegan_images: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode_images(
            stylegan_images
        )

    def count_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
