# freeze_model() Универсальная функция заморозки модели.
# Важно: замораживаются параметры генератора, но gradient по его входу W+ сохраняется.

# configure_reference_operations()- скрывает техническое исправление CUDA fallback. 
#В notebook больше не нужно вручную менять:upfirdn2d._plugin, bias_act._plugin

# load_stylegan_generator()
# чтение .pkl;
# извлечение G_ema;
# перенос на GPU;
# заморозку

# validate_generator_info() - проверяет, что latent bank был создан тем же типом генератора.

# synthesize_from_w() - единая функция генерации из W+.

# stylegan_tensor_to_uint8() - преобразование для визуализации больше не будет дублироваться между ноутбуками.

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def freeze_model(
    model: torch.nn.Module,
) -> torch.nn.Module:
    """
    Переводит модель в inference-режим и отключает
    вычисление градиентов для всех её параметров.

    Градиент по входу модели при этом сохраняется.
    Например, через замороженный StyleGAN2 всё равно
    можно вычислять dL/dw.
    """

    model.eval()
    model.requires_grad_(False)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def configure_reference_operations() -> None:
    """
    Принудительно включает reference PyTorch-реализации
    операций StyleGAN2.

    В текущем Colab-окружении custom CUDA extensions
    bias_act и upfirdn2d не собираются. Reference-версии
    медленнее, но остаются дифференцируемыми.
    """

    from torch_utils.ops import bias_act
    from torch_utils.ops import upfirdn2d

    upfirdn2d._plugin = None
    upfirdn2d._inited = True

    bias_act._plugin = None
    bias_act._inited = True


def load_stylegan_generator(
    model_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """
    Загружает G_ema из официального StyleGAN2 pickle,
    переносит генератор на устройство и замораживает его.
    """

    import legacy

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"StyleGAN2 model was not found: {model_path}"
        )

    with model_path.open("rb") as file:
        network_data: dict[str, Any] = (
            legacy.load_network_pkl(file)
        )

    if "G_ema" not in network_data:
        raise KeyError(
            "The model file does not contain G_ema."
        )

    generator = network_data["G_ema"].to(device)

    generator = freeze_model(
        generator
    )

    return generator


def validate_generator_info(
    generator: torch.nn.Module,
    generator_info: dict,
) -> None:
    """
    Проверяет совместимость загруженного генератора
    с метаданными сохранённого latent bank.
    """

    expected_values = {
        "z_dim": generator.z_dim,
        "w_dim": generator.w_dim,
        "num_ws": generator.num_ws,
        "resolution": generator.img_resolution,
        "channels": generator.img_channels,
    }

    mismatches = {}

    for key, actual_value in expected_values.items():
        stored_value = generator_info.get(key)

        if stored_value != actual_value:
            mismatches[key] = {
                "stored": stored_value,
                "loaded": actual_value,
            }

    if mismatches:
        raise ValueError(
            "Generator is incompatible with latent bank: "
            f"{mismatches}"
        )


def synthesize_from_w(
    generator: torch.nn.Module,
    w: torch.Tensor,
    noise_mode: str = "const",
) -> torch.Tensor:
    """
    Генерирует изображение из W+.

    Функция намеренно не использует torch.no_grad().
    Режим вычисления градиентов контролирует вызывающий код:

        with torch.no_grad():
            source = synthesize_from_w(...)

    или:

        edited = synthesize_from_w(...)
    """

    if w.ndim != 3:
        raise ValueError(
            "Expected W+ with shape [B, num_ws, w_dim], "
            f"got {tuple(w.shape)}."
        )

    expected_shape = (
        generator.num_ws,
        generator.w_dim,
    )

    if tuple(w.shape[1:]) != expected_shape:
        raise ValueError(
            "W+ is incompatible with generator. "
            f"Expected [B, {generator.num_ws}, "
            f"{generator.w_dim}], got {tuple(w.shape)}."
        )

    image = generator.synthesis(
        w,
        noise_mode=noise_mode,
        force_fp32=True,
    )

    return image


def stylegan_tensor_to_uint8(
    image: torch.Tensor,
) -> np.ndarray:
    """
    Преобразует одно изображение StyleGAN2:

        [1, 3, H, W], float, [-1, 1]

    в:

        [H, W, 3], uint8, [0, 255].

    Функция используется только для визуализации
    и сохранения результатов.
    """

    if image.ndim != 4:
        raise ValueError(
            "Expected image with shape [B, C, H, W], "
            f"got {tuple(image.shape)}."
        )

    if image.shape[0] != 1:
        raise ValueError(
            "This function supports one image at a time."
        )

    if image.shape[1] != 3:
        raise ValueError(
            "Expected an RGB image with three channels."
        )

    image_uint8 = (
        image
        .detach()
        .permute(0, 2, 3, 1)
        .mul(127.5)
        .add(128)
        .clamp(0, 255)
        .to(torch.uint8)
    )

    return image_uint8[0].cpu().numpy()
