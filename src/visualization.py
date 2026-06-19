
from __future__ import annotations

import math
from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.figure import Figure

from .optimization import OptimizationResult
from .stylegan_utils import stylegan_tensor_to_uint8


def plot_optimization_grid(
    result: OptimizationResult,
    columns: int = 3,
    title: str | None = None,
) -> Figure:
    """
    Строит сетку промежуточных изображений,
    сохранённых во время latent optimization.

    Parameters
    ----------
    result:
        Результат optimize_latent().

    columns:
        Количество изображений в одной строке.

    title:
        Общий заголовок. Если не указан,
        используется target prompt.

    Returns
    -------
    matplotlib.figure.Figure
        Созданная фигура.

    Notes
    -----
    Функция не вызывает plt.show() и не сохраняет файл.
    Это остаётся ответственностью notebook
    или experiment_io.py.
    """

    if columns < 1:
        raise ValueError(
            "columns must be at least 1."
        )

    if not result.saved_images:
        raise ValueError(
            "OptimizationResult does not contain "
            "saved images."
        )

    steps = sorted(
        result.saved_images.keys()
    )

    rows = math.ceil(
        len(steps) / columns
    )

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            4 * columns,
            4 * rows,
        ),
        squeeze=False,
    )

    axes_flat = axes.flatten()

    history_by_step = (
        result.history
        .set_index("step")
    )

    for axis, step in zip(
        axes_flat,
        steps,
    ):
        image_tensor = (
            result.saved_images[step]
        )

        image_uint8 = (
            stylegan_tensor_to_uint8(
                image_tensor
            )
        )

        axis.imshow(
            image_uint8
        )

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

    # Если количество изображений не кратно columns,
    # скрываем оставшиеся пустые области.
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
    """
    Строит график одной метрики по шагам оптимизации.

    Parameters
    ----------
    history:
        Таблица, полученная optimize_latent().

    metric:
        Название колонки, например:
        clip_loss, clip_similarity, l2_loss,
        weighted_l2_loss, total_loss,
        latent_distance или gradient_norm.

    title:
        Заголовок графика.

    ylabel:
        Подпись оси Y.

    Returns
    -------
    matplotlib.figure.Figure
        Созданная фигура.
    """

    if "step" not in history.columns:
        raise KeyError(
            "History must contain the 'step' column."
        )

    if metric not in history.columns:
        raise KeyError(
            f"Metric '{metric}' was not found. "
            f"Available columns: "
            f"{list(history.columns)}"
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

    plt.xlabel(
        "Optimization step"
    )

    plt.ylabel(
        ylabel or metric
    )

    plt.title(
        title or f"{metric} during optimization"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    return figure


def create_standard_figures(
    result: OptimizationResult,
) -> dict[str, Figure]:
    """
    Создаёт стандартный набор визуализаций
    для одного эксперимента.

    Returns
    -------
    dict[str, Figure]
        Словарь:
        {
            "optimization_grid": ...,
            "total_loss": ...,
            "clip_loss": ...,
            "clip_similarity": ...,
            "latent_distance": ...,
            "l2_loss": ...,
            "weighted_l2_loss": ...
        }
    """

    figures = {
        "optimization_grid": (
            plot_optimization_grid(
                result=result,
            )
        ),

        "total_loss": (
            plot_metric_history(
                history=result.history,
                metric="total_loss",
                title=(
                    "Total loss during "
                    "latent optimization"
                ),
                ylabel="Total loss",
            )
        ),

        "clip_loss": (
            plot_metric_history(
                history=result.history,
                metric="clip_loss",
                title=(
                    "CLIP loss during "
                    "latent optimization"
                ),
                ylabel="CLIP loss",
            )
        ),

        "clip_similarity": (
            plot_metric_history(
                history=result.history,
                metric="clip_similarity",
                title=(
                    "Image–text similarity during "
                    "latent optimization"
                ),
                ylabel="Cosine similarity",
            )
        ),

        "latent_distance": (
            plot_metric_history(
                history=result.history,
                metric="latent_distance",
                title=(
                    "Latent displacement during "
                    "optimization"
                ),
                ylabel="L2 distance from source W+",
            )
        ),

        "l2_loss": (
            plot_metric_history(
                history=result.history,
                metric="l2_loss",
                title=(
                    "Latent L2 loss during "
                    "optimization"
                ),
                ylabel="L2 loss",
            )
        ),

        "weighted_l2_loss": (
            plot_metric_history(
                history=result.history,
                metric="weighted_l2_loss",
                title=(
                    "Weighted latent L2 loss"
                ),
                ylabel="Weighted L2 loss",
            )
        ),
    }

    return figures


def close_figures(
    figures: Mapping[str, Figure],
) -> None:
    """
    Закрывает набор matplotlib figures.

    Полезно после сохранения большого числа графиков,
    чтобы не накапливать память.
    """

    for figure in figures.values():
        plt.close(figure)
