from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_alignment_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Вычисляет CLIP loss между изображением и текстом.

    Предполагается, что embeddings имеют форму [B, D].
    Один text embedding формы [1, D] можно применять
    ко всему batch изображений.
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
            "the same feature dimension."
        )

    image_batch_size = image_features.shape[0]
    text_batch_size = text_features.shape[0]

    if text_batch_size == 1 and image_batch_size > 1:
        text_features = text_features.expand(
            image_batch_size,
            -1,
        )
    elif text_batch_size != image_batch_size:
        raise ValueError(
            "Text batch size must be 1 or match "
            "the image batch size."
        )

    image_features = F.normalize(
        image_features,
        p=2,
        dim=-1,
    )

    text_features = F.normalize(
        text_features,
        p=2,
        dim=-1,
    )

    per_sample_similarity = (
        image_features
        * text_features
    ).sum(dim=-1)

    similarity = per_sample_similarity.mean()
    loss = 1.0 - similarity

    return loss, similarity


def latent_l2_loss(
    edited_w: torch.Tensor,
    source_w: torch.Tensor,
) -> torch.Tensor:
    """
    Вычисляет L2-расстояние между отредактированным
    и исходным W+ и усредняет его по batch.
    """

    if edited_w.ndim != 3:
        raise ValueError(
            "Expected edited W+ with shape "
            "[B, num_ws, w_dim], "
            f"got {tuple(edited_w.shape)}."
        )

    if source_w.ndim != 3:
        raise ValueError(
            "Expected source W+ with shape "
            "[B, num_ws, w_dim], "
            f"got {tuple(source_w.shape)}."
        )

    if edited_w.shape != source_w.shape:
        raise ValueError(
            "Edited and source latent codes must have "
            "the same shape. "
            f"Got {tuple(edited_w.shape)} and "
            f"{tuple(source_w.shape)}."
        )

    latent_difference = edited_w - source_w

    per_sample_distance = torch.linalg.vector_norm(
        latent_difference.flatten(start_dim=1),
        ord=2,
        dim=1,
    )

    return per_sample_distance.mean()


def identity_preservation_loss(
    source_features: torch.Tensor,
    edited_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Вычисляет identity loss:

        L_ID = 1 - cosine(source_features, edited_features)

    Исходные признаки отсоединяются от графа.
    Edited features остаются в графе, чтобы gradient
    прошёл через ArcFace и StyleGAN2 к W+.
    """

    if source_features.ndim != 2:
        raise ValueError(
            "Expected source identity features "
            "with shape [B, D], "
            f"got {tuple(source_features.shape)}."
        )

    if edited_features.ndim != 2:
        raise ValueError(
            "Expected edited identity features "
            "with shape [B, D], "
            f"got {tuple(edited_features.shape)}."
        )

    if source_features.shape != edited_features.shape:
        raise ValueError(
            "Source and edited identity features "
            "must have the same shape. "
            f"Got {tuple(source_features.shape)} and "
            f"{tuple(edited_features.shape)}."
        )

    if source_features.device != edited_features.device:
        raise ValueError(
            "Source and edited identity features "
            "must be on the same device."
        )

    if not torch.is_floating_point(source_features):
        raise TypeError(
            "source_features must be floating-point."
        )

    if not torch.is_floating_point(edited_features):
        raise TypeError(
            "edited_features must be floating-point."
        )

    source_features = F.normalize(
        source_features.detach(),
        p=2,
        dim=-1,
    )

    edited_features = F.normalize(
        edited_features,
        p=2,
        dim=-1,
    )

    per_sample_similarity = (
        source_features
        * edited_features
    ).sum(dim=-1)

    identity_similarity = per_sample_similarity.mean()
    identity_loss = 1.0 - identity_similarity

    return identity_loss, identity_similarity
