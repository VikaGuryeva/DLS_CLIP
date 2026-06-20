from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import torch

from .losses import (
    clip_alignment_loss,
    identity_preservation_loss,
    latent_l2_loss,
)
from .stylegan_utils import synthesize_from_w


@dataclass(frozen=True)
class OptimizationConfig:
    """
    Конфигурация одного запуска latent optimization.
    """

    experiment_name: str
    target_prompt: str
    num_steps: int
    learning_rate: float
    lambda_l2: float = 0.0
    lambda_id: float = 0.0
    save_steps: tuple[int, ...] = (0,)
    noise_mode: str = "const"

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError(
                "experiment_name must not be empty."
            )

        if not self.target_prompt.strip():
            raise ValueError(
                "target_prompt must not be empty."
            )

        if self.num_steps < 1:
            raise ValueError(
                "num_steps must be at least 1."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.lambda_l2 < 0:
            raise ValueError(
                "lambda_l2 must be non-negative."
            )

        if self.lambda_id < 0:
            raise ValueError(
                "lambda_id must be non-negative."
            )

        invalid_save_steps = [
            step
            for step in self.save_steps
            if step < 0 or step > self.num_steps
        ]

        if invalid_save_steps:
            raise ValueError(
                "save_steps must belong to the interval "
                f"[0, {self.num_steps}]. "
                f"Invalid values: {invalid_save_steps}"
            )

        if self.noise_mode not in {
            "const",
            "random",
            "none",
        }:
            raise ValueError(
                "noise_mode must be one of: "
                "'const', 'random', 'none'."
            )


@dataclass
class OptimizationResult:
    """
    Результат одного запуска latent optimization.
    """

    config: OptimizationConfig
    source_w: torch.Tensor
    edited_w: torch.Tensor
    best_w: torch.Tensor
    best_step: int
    best_clip_similarity: float
    history: pd.DataFrame
    saved_images: dict[int, torch.Tensor]
    elapsed_time_seconds: float


def _validate_frozen_model(
    model: torch.nn.Module,
    model_name: str,
) -> None:
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if trainable_parameters != 0:
        raise ValueError(
            f"{model_name} must be frozen. "
            f"Trainable parameters: {trainable_parameters}."
        )


def _synchronize_device(
    device: torch.device,
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def optimize_latent(
    generator: torch.nn.Module,
    clip_encoder: torch.nn.Module,
    source_w: torch.Tensor,
    config: OptimizationConfig,
    identity_encoder: torch.nn.Module | None = None,
) -> OptimizationResult:
    """
    Оптимизирует W+ по:

        total_loss =
            clip_loss
            + lambda_l2 * l2_loss
            + lambda_id * identity_loss

    При lambda_id = 0 identity encoder не используется.
    """

    if source_w.ndim != 3:
        raise ValueError(
            "source_w must have shape "
            "[B, num_ws, w_dim], "
            f"got {tuple(source_w.shape)}."
        )

    if not torch.is_floating_point(source_w):
        raise TypeError(
            "source_w must be a floating-point tensor."
        )

    expected_latent_shape = (
        generator.num_ws,
        generator.w_dim,
    )

    if tuple(source_w.shape[1:]) != expected_latent_shape:
        raise ValueError(
            "source_w is incompatible with generator. "
            f"Expected [B, {generator.num_ws}, "
            f"{generator.w_dim}], "
            f"got {tuple(source_w.shape)}."
        )

    if not hasattr(clip_encoder, "encode_text"):
        raise TypeError(
            "clip_encoder must implement encode_text()."
        )

    if not hasattr(clip_encoder, "encode_images"):
        raise TypeError(
            "clip_encoder must implement encode_images()."
        )

    use_identity_loss = config.lambda_id > 0.0

    if use_identity_loss:
        if identity_encoder is None:
            raise ValueError(
                "identity_encoder is required "
                "when lambda_id > 0."
            )

        if not hasattr(identity_encoder, "encode_images"):
            raise TypeError(
                "identity_encoder must implement encode_images()."
            )

        _validate_frozen_model(
            identity_encoder,
            "Identity encoder",
        )

    _validate_frozen_model(
        generator,
        "StyleGAN2 generator",
    )

    _validate_frozen_model(
        clip_encoder,
        "CLIP encoder",
    )

    generator.eval()
    clip_encoder.eval()

    if use_identity_loss:
        identity_encoder.eval()

    source_w_fixed = (
        source_w
        .clone()
        .detach()
    )

    edited_w = (
        source_w_fixed
        .clone()
        .detach()
        .requires_grad_(True)
    )

    optimizer = torch.optim.Adam(
        [edited_w],
        lr=config.learning_rate,
    )

    target_text_features = clip_encoder.encode_text(
        [config.target_prompt]
    )

    source_identity_features = None

    if use_identity_loss:
        with torch.no_grad():
            source_image = synthesize_from_w(
                generator=generator,
                w=source_w_fixed,
                noise_mode=config.noise_mode,
            )

            source_identity_features = (
                identity_encoder
                .encode_images(source_image)
                .detach()
            )

    required_save_steps = set(
        config.save_steps
    )

    required_save_steps.add(0)
    required_save_steps.add(
        config.num_steps
    )

    history_records: list[dict] = []
    saved_images: dict[int, torch.Tensor] = {}

    best_similarity = float("-inf")
    best_step = 0
    best_w = (
        source_w_fixed
        .detach()
        .cpu()
        .clone()
    )

    generator.zero_grad(set_to_none=True)
    clip_encoder.zero_grad(set_to_none=True)

    if use_identity_loss:
        identity_encoder.zero_grad(
            set_to_none=True
        )

    _synchronize_device(source_w.device)
    start_time = time.perf_counter()

    for step in range(config.num_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        edited_image = synthesize_from_w(
            generator=generator,
            w=edited_w,
            noise_mode=config.noise_mode,
        )

        image_features = clip_encoder.encode_images(
            edited_image
        )

        clip_loss, clip_similarity = (
            clip_alignment_loss(
                image_features=image_features,
                text_features=target_text_features,
            )
        )

        l2_loss = latent_l2_loss(
            edited_w=edited_w,
            source_w=source_w_fixed,
        )

        weighted_l2_loss = (
            config.lambda_l2
            * l2_loss
        )

        if use_identity_loss:
            edited_identity_features = (
                identity_encoder.encode_images(
                    edited_image
                )
            )

            id_loss, identity_similarity = (
                identity_preservation_loss(
                    source_features=source_identity_features,
                    edited_features=edited_identity_features,
                )
            )

            weighted_id_loss = (
                config.lambda_id
                * id_loss
            )

            id_loss_for_history = float(
                id_loss.detach().cpu()
            )

            weighted_id_loss_for_history = float(
                weighted_id_loss.detach().cpu()
            )

            identity_similarity_for_history = float(
                identity_similarity.detach().cpu()
            )
        else:
            weighted_id_loss = torch.zeros(
                (),
                device=edited_w.device,
                dtype=edited_w.dtype,
            )

            id_loss_for_history = float("nan")
            weighted_id_loss_for_history = 0.0
            identity_similarity_for_history = float("nan")

        total_loss = (
            clip_loss
            + weighted_l2_loss
            + weighted_id_loss
        )

        current_similarity = float(
            clip_similarity.detach().cpu()
        )

        if current_similarity > best_similarity:
            best_similarity = current_similarity
            best_step = int(step)
            best_w = (
                edited_w
                .detach()
                .cpu()
                .clone()
            )

        if step in required_save_steps:
            saved_images[step] = (
                edited_image
                .detach()
                .cpu()
                .clone()
            )

        gradient_norm: float | None = None

        if step < config.num_steps:
            total_loss.backward()

            if edited_w.grad is None:
                raise RuntimeError(
                    "Gradient did not reach W+ "
                    f"at step {step}."
                )

            if not torch.isfinite(
                edited_w.grad
            ).all():
                raise RuntimeError(
                    "Non-finite gradient detected "
                    f"at step {step}."
                )

            gradient_norm = float(
                edited_w.grad
                .detach()
                .norm()
                .cpu()
            )

            optimizer.step()

        history_records.append(
            {
                "step": int(step),
                "total_loss": float(
                    total_loss.detach().cpu()
                ),
                "clip_loss": float(
                    clip_loss.detach().cpu()
                ),
                "clip_similarity": current_similarity,
                "l2_loss": float(
                    l2_loss.detach().cpu()
                ),
                "weighted_l2_loss": float(
                    weighted_l2_loss.detach().cpu()
                ),
                "id_loss": id_loss_for_history,
                "weighted_id_loss": (
                    weighted_id_loss_for_history
                ),
                "identity_similarity": (
                    identity_similarity_for_history
                ),
                "latent_distance": float(
                    l2_loss.detach().cpu()
                ),
                "gradient_norm": gradient_norm,
            }
        )

    _synchronize_device(source_w.device)

    elapsed_time_seconds = (
        time.perf_counter()
        - start_time
    )

    history = pd.DataFrame(
        history_records
    )

    if not torch.equal(
        source_w_fixed,
        source_w.detach(),
    ):
        raise RuntimeError(
            "The source W+ was modified."
        )

    generator_gradients = sum(
        parameter.grad is not None
        for parameter in generator.parameters()
    )

    clip_gradients = sum(
        parameter.grad is not None
        for parameter in clip_encoder.parameters()
    )

    if generator_gradients != 0:
        raise RuntimeError(
            "StyleGAN2 parameters received gradients."
        )

    if clip_gradients != 0:
        raise RuntimeError(
            "CLIP parameters received gradients."
        )

    if use_identity_loss:
        identity_gradients = sum(
            parameter.grad is not None
            for parameter in identity_encoder.parameters()
        )

        if identity_gradients != 0:
            raise RuntimeError(
                "Identity encoder parameters "
                "received gradients."
            )

    return OptimizationResult(
        config=config,
        source_w=(
            source_w_fixed
            .detach()
            .cpu()
        ),
        edited_w=(
            edited_w
            .detach()
            .cpu()
        ),
        best_w=best_w,
        best_step=best_step,
        best_clip_similarity=best_similarity,
        history=history,
        saved_images=saved_images,
        elapsed_time_seconds=float(
            elapsed_time_seconds
        ),
    )
