
from __future__ import annotations

import torch


def clip_alignment_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Вычисляет direct CLIP loss.

    Предполагается, что image_features и text_features
    уже L2-нормализованы.

    Формула:
        similarity = cosine(image_features, text_features)
        loss = 1 - similarity

    Parameters
    ----------
    image_features:
        CLIP image embeddings формы [B, D].

    text_features:
        CLIP text embeddings формы [1, D] или [B, D].

    Returns
    -------
    loss:
        Средний CLIP loss по batch.

    similarity:
        Средняя cosine similarity по batch.
    """

    if image_features.ndim != 2:
        raise ValueError(
            "Expected image features with shape [B, D], "
            f"got {tuple(image_features.shape)}."
        )

    if text_features.ndim != 2:
        raise ValueError(
            "Expected text features with shape [B, D], "
            f"got {tuple(text_features.shape)}."
        )

    if image_features.shape[1] != text_features.shape[1]:
        raise ValueError(
            "Image and text embeddings must have "
            "the same feature dimension. "
            f"Got {image_features.shape[1]} and "
            f"{text_features.shape[1]}."
        )

    image_batch_size = image_features.shape[0]
    text_batch_size = text_features.shape[0]

    # Один prompt можно применить ко всему batch изображений.
    if text_batch_size == 1 and image_batch_size > 1:
        text_features = text_features.expand(
            image_batch_size,
            -1,
        )

    elif text_batch_size != image_batch_size:
        raise ValueError(
            "Text batch size must be 1 or match "
            "the image batch size. "
            f"Got image batch {image_batch_size} and "
            f"text batch {text_batch_size}."
        )

    cosine_similarity = (
        image_features
        * text_features
    ).sum(dim=-1)

    mean_similarity = cosine_similarity.mean()
    loss = 1.0 - mean_similarity

    return loss, mean_similarity


def latent_l2_loss(
    edited_w: torch.Tensor,
    source_w: torch.Tensor,
) -> torch.Tensor:
    """
    Вычисляет L2-расстояние между редактируемым
    и исходным latent-кодами W+.

    Для каждого объекта batch вычисляется:

        ||edited_w - source_w||_2

    Затем значения усредняются по batch.

    Parameters
    ----------
    edited_w:
        Редактируемый latent-код [B, num_ws, w_dim].

    source_w:
        Исходный latent-код [B, num_ws, w_dim].

    Returns
    -------
    torch.Tensor:
        Скалярный L2 loss.
    """

    if edited_w.shape != source_w.shape:
        raise ValueError(
            "Edited and source latent codes must have "
            "the same shape. "
            f"Got {tuple(edited_w.shape)} and "
            f"{tuple(source_w.shape)}."
        )

    if edited_w.ndim != 3:
        raise ValueError(
            "Expected W+ tensors with shape "
            "[B, num_ws, w_dim], "
            f"got {tuple(edited_w.shape)}."
        )

    latent_difference = edited_w - source_w

    # Объединяем num_ws и w_dim,
    # сохраняя отдельное значение для каждого объекта batch.
    per_sample_distance = torch.linalg.vector_norm(
        latent_difference.flatten(start_dim=1),
        ord=2,
        dim=1,
    )

    return per_sample_distance.mean()
