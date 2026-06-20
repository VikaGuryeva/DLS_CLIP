from __future__ import annotations

import math
from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .optimization import OptimizationResult
from .stylegan_utils import stylegan_tensor_to_uint8


def plot_optimization_grid(
    result: OptimizationResult,
    columns: int = 3,
    title: str | None = None,
) -> Figure:
    if columns < 1:
        raise ValueError(
            "columns must be at least 1."
        )

    if not result.saved_images:
        raise ValueError(
            "OptimizationResult does not contain saved images."
        )

    steps = sorted(result.saved_images.keys())
    rows = math.ceil(len(steps) / columns)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 4 * rows),
        squeeze=False,
    )

    axes_flat = axes.flatten()
    history_by_step = result.history.set_index(
        "step"
    )

    for axis, step in zip(
        axes_flat,
        steps,
    ):
        image_uint8 = stylegan_tensor_to_uint8(
            result.saved_images[step]
        )

        axis.imshow(image_uint8)

        if step in history_by_step.index:
            similarity = history_by_step.loc[
                step,
                "clip_similarity",
            ]

            axis.set_title(
                f"Step {step}\n"
                f"similarity = {similarity:.4f}"
            )
        else:
            axis.set_title(
                f"Step {step}"
            )

        axis.axis("off")

    for axis in axes_flat[len(steps):]:
        axis.axis("off")

    figure.suptitle(
        title or result.config.target_prompt,
        fontsize=14,
    )

    figure.tight_layout()

    return figure


def plot_metric_history(
    history: pd.DataFrame,
    metric: str,
    title: str | None = None,
    ylabel: str | None = None,
) -> Figure:
    if "step" not in history.columns:
        raise KeyError(
            "History must contain the 'step' column."
        )

    if metric not in history.columns:
        raise KeyError(
            f"Metric '{metric}' was not found. "
            f"Available columns: {list(history.columns)}"
        )

    if history.empty:
        raise ValueError(
            "History must not be empty."
        )

    figure = plt.figure(
        figsize=(8, 4)
    )

    plt.plot(
        history["step"],
        history[metric],
    )

    plt.xlabel("Optimization step")
    plt.ylabel(ylabel or metric)
    plt.title(
        title or f"{metric} during optimization"
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()

    return figure


def _has_metric_data(
    history: pd.DataFrame,
    metric: str,
) -> bool:
    return (
        metric in history.columns
        and history[metric].notna().any()
    )


def create_standard_figures(
    result: OptimizationResult,
) -> dict[str, Figure]:
    figures = {
        "optimization_grid": plot_optimization_grid(
            result=result,
        ),
        "total_loss": plot_metric_history(
            history=result.history,
            metric="total_loss",
            title="Total loss during latent optimization",
            ylabel="Total loss",
        ),
        "clip_loss": plot_metric_history(
            history=result.history,
            metric="clip_loss",
            title="CLIP loss during latent optimization",
            ylabel="CLIP loss",
        ),
        "clip_similarity": plot_metric_history(
            history=result.history,
            metric="clip_similarity",
            title=(
                "Image–text similarity during "
                "latent optimization"
            ),
            ylabel="Cosine similarity",
        ),
        "latent_distance": plot_metric_history(
            history=result.history,
            metric="latent_distance",
            title="Latent displacement during optimization",
            ylabel="L2 distance from source W+",
        ),
        "l2_loss": plot_metric_history(
            history=result.history,
            metric="l2_loss",
            title="Latent L2 loss during optimization",
            ylabel="L2 loss",
        ),
        "weighted_l2_loss": plot_metric_history(
            history=result.history,
            metric="weighted_l2_loss",
            title="Weighted latent L2 loss",
            ylabel="Weighted L2 loss",
        ),
    }

    if _has_metric_data(
        result.history,
        "id_loss",
    ):
        figures["id_loss"] = plot_metric_history(
            history=result.history,
            metric="id_loss",
            title=(
                "Identity loss during "
                "latent optimization"
            ),
            ylabel="Identity loss",
        )

    if _has_metric_data(
        result.history,
        "weighted_id_loss",
    ):
        figures["weighted_id_loss"] = (
            plot_metric_history(
                history=result.history,
                metric="weighted_id_loss",
                title="Weighted identity loss",
                ylabel="Weighted identity loss",
            )
        )

    if _has_metric_data(
        result.history,
        "identity_similarity",
    ):
        figures["identity_similarity"] = (
            plot_metric_history(
                history=result.history,
                metric="identity_similarity",
                title=(
                    "Identity similarity during "
                    "latent optimization"
                ),
                ylabel="Cosine similarity",
            )
        )

    return figures


def close_figures(
    figures: Mapping[str, Figure],
) -> None:
    for figure in figures.values():
        plt.close(figure)
