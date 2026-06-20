from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from matplotlib.figure import Figure
from PIL import Image

from .data_utils import SourceSample
from .optimization import OptimizationResult
from .stylegan_utils import stylegan_tensor_to_uint8
from .visualization import (
    close_figures,
    create_standard_figures,
)


@dataclass(frozen=True)
class ExperimentArtifacts:
    root_dir: Path
    config_path: Path
    metrics_path: Path
    environment_path: Path
    history_path: Path
    latent_path: Path
    notes_path: Path
    manifest_path: Path
    images_dir: Path
    figures_dir: Path


def _get_git_commit(
    repo_dir: str | Path | None,
) -> str | None:
    if repo_dir is None:
        return None

    repo_dir = Path(repo_dir)

    if not (repo_dir / ".git").exists():
        return None

    try:
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_dir),
                "rev-parse",
                "HEAD",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return None


def _prepare_output_directory(
    output_dir: str | Path,
    overwrite: bool,
) -> Path:
    output_dir = Path(output_dir)

    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not overwrite
    ):
        raise FileExistsError(
            "Experiment directory already exists and is not empty: "
            f"{output_dir}. "
            "Use overwrite=True or choose another directory."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def _write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def _build_config(
    result: OptimizationResult,
    source_sample: SourceSample,
    clip_model_id: str | None,
    generator_info: Mapping[str, Any] | None,
    identity_model_info: Mapping[str, Any] | None,
    git_commit: str | None,
) -> dict[str, Any]:
    config = asdict(result.config)

    config.update(
        {
            "image_id": source_sample.image_id,
            "source_seed": int(source_sample.seed),
            "source_split": source_sample.split,
            "source_image_path": str(
                source_sample.image_path
            ),
            "clip_model_id": clip_model_id,
            "git_commit": git_commit,
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }
    )

    if generator_info is not None:
        config["generator"] = dict(
            generator_info
        )

    if identity_model_info is not None:
        config["identity_model"] = dict(
            identity_model_info
        )

    return config


def _build_metrics_summary(
    result: OptimizationResult,
) -> dict[str, Any]:
    history = result.history

    if history.empty:
        raise ValueError(
            "Optimization history must not be empty."
        )

    initial_row = history.iloc[0]
    final_row = history.iloc[-1]

    identity_enabled = (
        "identity_similarity" in history.columns
        and history["identity_similarity"].notna().any()
    )

    summary: dict[str, Any] = {
        "initial_step": int(initial_row["step"]),
        "final_step": int(final_row["step"]),
        "initial_total_loss": float(
            initial_row["total_loss"]
        ),
        "final_total_loss": float(
            final_row["total_loss"]
        ),
        "total_loss_change": float(
            final_row["total_loss"]
            - initial_row["total_loss"]
        ),
        "initial_clip_loss": float(
            initial_row["clip_loss"]
        ),
        "final_clip_loss": float(
            final_row["clip_loss"]
        ),
        "clip_loss_change": float(
            final_row["clip_loss"]
            - initial_row["clip_loss"]
        ),
        "initial_clip_similarity": float(
            initial_row["clip_similarity"]
        ),
        "final_clip_similarity": float(
            final_row["clip_similarity"]
        ),
        "clip_similarity_change": float(
            final_row["clip_similarity"]
            - initial_row["clip_similarity"]
        ),
        "initial_l2_loss": float(
            initial_row["l2_loss"]
        ),
        "final_l2_loss": float(
            final_row["l2_loss"]
        ),
        "final_weighted_l2_loss": float(
            final_row["weighted_l2_loss"]
        ),
        "final_latent_distance": float(
            final_row["latent_distance"]
        ),
        "best_step": int(result.best_step),
        "best_clip_similarity": float(
            result.best_clip_similarity
        ),
        "elapsed_time_seconds": float(
            result.elapsed_time_seconds
        ),
        "average_seconds_per_update": float(
            result.elapsed_time_seconds
            / result.config.num_steps
        ),
        "identity_enabled": bool(
            identity_enabled
        ),
    }

    if identity_enabled:
        identity_history = history.dropna(
            subset=["identity_similarity"]
        )

        initial_identity_row = (
            identity_history.iloc[0]
        )
        final_identity_row = (
            identity_history.iloc[-1]
        )

        summary.update(
            {
                "initial_identity_similarity": float(
                    initial_identity_row[
                        "identity_similarity"
                    ]
                ),
                "final_identity_similarity": float(
                    final_identity_row[
                        "identity_similarity"
                    ]
                ),
                "identity_similarity_change": float(
                    final_identity_row[
                        "identity_similarity"
                    ]
                    - initial_identity_row[
                        "identity_similarity"
                    ]
                ),
                "initial_id_loss": float(
                    initial_identity_row["id_loss"]
                ),
                "final_id_loss": float(
                    final_identity_row["id_loss"]
                ),
                "final_weighted_id_loss": float(
                    final_identity_row[
                        "weighted_id_loss"
                    ]
                ),
                "minimum_identity_similarity": float(
                    identity_history[
                        "identity_similarity"
                    ].min()
                ),
            }
        )
    else:
        summary.update(
            {
                "initial_identity_similarity": None,
                "final_identity_similarity": None,
                "identity_similarity_change": None,
                "initial_id_loss": None,
                "final_id_loss": None,
                "final_weighted_id_loss": None,
                "minimum_identity_similarity": None,
            }
        )

    return summary


def _build_environment_info(
    repo_dir: str | Path | None,
    git_commit: str | None,
) -> dict[str, Any]:
    gpu_name = None

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
        ),
        "cuda_available": bool(
            torch.cuda.is_available()
        ),
        "gpu": gpu_name,
        "git_commit": git_commit,
        "repo_dir": (
            str(Path(repo_dir))
            if repo_dir is not None
            else None
        ),
    }


def _save_latents(
    result: OptimizationResult,
    path: Path,
    source_sample: SourceSample,
) -> None:
    torch.save(
        {
            "image_id": source_sample.image_id,
            "source_seed": int(source_sample.seed),
            "target_prompt": (
                result.config.target_prompt
            ),
            "source_w": result.source_w.detach().cpu(),
            "edited_w": result.edited_w.detach().cpu(),
            "best_w": result.best_w.detach().cpu(),
            "best_step": int(result.best_step),
            "best_clip_similarity": float(
                result.best_clip_similarity
            ),
        },
        path,
    )


def _save_step_images(
    result: OptimizationResult,
    images_dir: Path,
) -> dict[str, Path]:
    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_paths: dict[str, Path] = {}

    for step, image_tensor in sorted(
        result.saved_images.items()
    ):
        image_array = stylegan_tensor_to_uint8(
            image_tensor
        )

        image_path = (
            images_dir
            / f"step_{step:03d}.png"
        )

        Image.fromarray(image_array).save(
            image_path
        )

        saved_paths[
            f"step_{step:03d}"
        ] = image_path

    if 0 in result.saved_images:
        source_path = images_dir / "source.png"

        Image.fromarray(
            stylegan_tensor_to_uint8(
                result.saved_images[0]
            )
        ).save(source_path)

        saved_paths["source"] = source_path

    final_step = result.config.num_steps

    if final_step in result.saved_images:
        final_path = images_dir / "final.png"

        Image.fromarray(
            stylegan_tensor_to_uint8(
                result.saved_images[
                    final_step
                ]
            )
        ).save(final_path)

        saved_paths["final"] = final_path

    if result.best_step in result.saved_images:
        best_path = images_dir / "best.png"

        Image.fromarray(
            stylegan_tensor_to_uint8(
                result.saved_images[
                    result.best_step
                ]
            )
        ).save(best_path)

        saved_paths["best"] = best_path

    return saved_paths


def _save_figures(
    figures: Mapping[str, Figure],
    figures_dir: Path,
    dpi: int,
) -> dict[str, Path]:
    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_paths: dict[str, Path] = {}

    for name, figure in figures.items():
        figure_path = (
            figures_dir
            / f"{name}.png"
        )

        figure.savefig(
            figure_path,
            dpi=dpi,
            bbox_inches="tight",
        )

        saved_paths[name] = figure_path

    return saved_paths


def build_notes_template(
    result: OptimizationResult,
    source_sample: SourceSample,
    metrics_summary: Mapping[str, Any],
) -> str:
    config = result.config
    identity_enabled = bool(
        metrics_summary["identity_enabled"]
    )

    if identity_enabled:
        identity_metrics_text = f"""
## 3. Метрики сохранения identity

- Начальная identity similarity: `{metrics_summary["initial_identity_similarity"]:.6f}`
- Финальная identity similarity: `{metrics_summary["final_identity_similarity"]:.6f}`
- Изменение identity similarity: `{metrics_summary["identity_similarity_change"]:+.6f}`
- Минимальная identity similarity: `{metrics_summary["minimum_identity_similarity"]:.6f}`
- Начальный ID loss: `{metrics_summary["initial_id_loss"]:.6f}`
- Финальный ID loss: `{metrics_summary["final_id_loss"]:.6f}`
- Финальный взвешенный ID loss: `{metrics_summary["final_weighted_id_loss"]:.6f}`
"""
    else:
        identity_metrics_text = """
## 3. Метрики сохранения identity

Identity loss в этом эксперименте не использовался.
"""

    return f"""# Эксперимент: {config.experiment_name}

## 1. Конфигурация

- Исходное изображение: `{source_sample.image_id}`
- Seed: `{source_sample.seed}`
- Split: `{source_sample.split}`
- Target prompt: `{config.target_prompt}`
- Количество шагов: `{config.num_steps}`
- Learning rate: `{config.learning_rate}`
- Optimizer: Adam
- Коэффициент L2: `{config.lambda_l2}`
- Коэффициент identity loss: `{config.lambda_id}`
- Noise mode: `{config.noise_mode}`

## 2. Основные количественные результаты

- Начальная CLIP similarity: `{metrics_summary["initial_clip_similarity"]:.6f}`
- Финальная CLIP similarity: `{metrics_summary["final_clip_similarity"]:.6f}`
- Изменение CLIP similarity: `{metrics_summary["clip_similarity_change"]:+.6f}`
- Начальный CLIP loss: `{metrics_summary["initial_clip_loss"]:.6f}`
- Финальный CLIP loss: `{metrics_summary["final_clip_loss"]:.6f}`
- Финальный L2 loss: `{metrics_summary["final_l2_loss"]:.6f}`
- Финальный взвешенный L2 loss: `{metrics_summary["final_weighted_l2_loss"]:.6f}`
- Финальное расстояние в W+: `{metrics_summary["final_latent_distance"]:.6f}`
- Лучший шаг по CLIP similarity: `{metrics_summary["best_step"]}`
- Лучшая CLIP similarity: `{metrics_summary["best_clip_similarity"]:.6f}`
- Время выполнения: `{metrics_summary["elapsed_time_seconds"]:.4f}` секунд
- Среднее время одного обновления: `{metrics_summary["average_seconds_per_update"]:.6f}` секунд

{identity_metrics_text}

## 4. Визуальные наблюдения

<!-- Опиши появление целевого атрибута и сохранение личности. -->

## 5. Интерпретация динамики loss

<!-- Опиши компромисс между CLIP, L2 и ID-компонентами. -->

## 6. Вывод

<!-- Сформулируй итог и следующие параметры для проверки. -->
"""


def _build_manifest(
    root_dir: Path,
) -> dict[str, Any]:
    files = []

    for path in sorted(
        root_dir.rglob("*")
    ):
        if path.is_file():
            files.append(
                {
                    "path": str(
                        path.relative_to(root_dir)
                    ),
                    "size_bytes": int(
                        path.stat().st_size
                    ),
                }
            )

    return {
        "root_dir": str(root_dir),
        "file_count": len(files),
        "files": files,
    }


def save_experiment_package(
    result: OptimizationResult,
    source_sample: SourceSample,
    output_dir: str | Path,
    *,
    clip_model_id: str | None = None,
    generator_info: Mapping[str, Any] | None = None,
    identity_model_info: Mapping[str, Any] | None = None,
    repo_dir: str | Path | None = None,
    notes_text: str | None = None,
    figures: Mapping[str, Figure] | None = None,
    create_figures: bool = True,
    close_created_figures: bool = True,
    dpi: int = 200,
    overwrite: bool = False,
) -> ExperimentArtifacts:
    if dpi < 1:
        raise ValueError(
            "dpi must be positive."
        )

    output_dir = _prepare_output_directory(
        output_dir=output_dir,
        overwrite=overwrite,
    )

    images_dir = output_dir / "images"
    figures_dir = output_dir / "figures"
    config_path = output_dir / "config.json"
    metrics_path = (
        output_dir / "metrics_summary.json"
    )
    environment_path = (
        output_dir / "environment.json"
    )
    history_path = output_dir / "history.csv"
    latent_path = output_dir / "latents.pt"
    notes_path = (
        output_dir / "experiment_notes.md"
    )
    manifest_path = output_dir / "manifest.json"

    git_commit = _get_git_commit(
        repo_dir
    )

    config_data = _build_config(
        result=result,
        source_sample=source_sample,
        clip_model_id=clip_model_id,
        generator_info=generator_info,
        identity_model_info=identity_model_info,
        git_commit=git_commit,
    )

    metrics_data = _build_metrics_summary(
        result
    )

    environment_data = _build_environment_info(
        repo_dir=repo_dir,
        git_commit=git_commit,
    )

    _write_json(config_path, config_data)
    _write_json(metrics_path, metrics_data)
    _write_json(
        environment_path,
        environment_data,
    )

    result.history.to_csv(
        history_path,
        index=False,
    )

    _save_latents(
        result=result,
        path=latent_path,
        source_sample=source_sample,
    )

    _save_step_images(
        result=result,
        images_dir=images_dir,
    )

    created_figures = False

    if figures is None and create_figures:
        figures = create_standard_figures(
            result
        )
        created_figures = True

    if figures is not None:
        _save_figures(
            figures=figures,
            figures_dir=figures_dir,
            dpi=dpi,
        )

    if notes_text is None:
        notes_text = build_notes_template(
            result=result,
            source_sample=source_sample,
            metrics_summary=metrics_data,
        )

    notes_path.write_text(
        notes_text,
        encoding="utf-8",
    )

    if (
        created_figures
        and figures is not None
        and close_created_figures
    ):
        close_figures(figures)

    manifest_data = _build_manifest(
        output_dir
    )

    _write_json(
        manifest_path,
        manifest_data,
    )

    return ExperimentArtifacts(
        root_dir=output_dir,
        config_path=config_path,
        metrics_path=metrics_path,
        environment_path=environment_path,
        history_path=history_path,
        latent_path=latent_path,
        notes_path=notes_path,
        manifest_path=manifest_path,
        images_dir=images_dir,
        figures_dir=figures_dir,
    )
