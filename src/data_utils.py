
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch


REQUIRED_METADATA_COLUMNS = {
    "image_id",
    "split",
    "seed",
    "image_path",
    "latent_key",
    "z_shape",
    "w_plus_shape",
    "resolution",
    "truncation_psi",
    "noise_mode",
}


@dataclass
class SourceSample:
    """
    Один объект из source latent bank.

    Attributes
    ----------
    image_id:
        Уникальный идентификатор объекта.

    seed:
        Seed, использованный для получения исходного z.

    split:
        Часть выборки: debug, dev или test.

    z:
        Исходный latent-вектор пространства Z.

    w_plus:
        Исходный latent-код пространства W+.

    image_path:
        Абсолютный путь к сохранённому PNG.
    """

    image_id: str
    seed: int
    split: str
    z: torch.Tensor
    w_plus: torch.Tensor
    image_path: Path

    def to(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "SourceSample":
        """
        Возвращает новый SourceSample,
        в котором z и W+ перенесены на device.
        """

        return SourceSample(
            image_id=self.image_id,
            seed=self.seed,
            split=self.split,
            z=self.z.to(
                device=device,
                dtype=dtype,
            ),
            w_plus=self.w_plus.to(
                device=device,
                dtype=dtype,
            ),
            image_path=self.image_path,
        )


def load_source_bank(
    latent_bank_path: str | Path,
    metadata_path: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Загружает latent_bank.pt и metadata.csv.
    """

    latent_bank_path = Path(
        latent_bank_path
    )

    metadata_path = Path(
        metadata_path
    )

    if not latent_bank_path.exists():
        raise FileNotFoundError(
            "Latent bank was not found: "
            f"{latent_bank_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Metadata file was not found: "
            f"{metadata_path}"
        )

    latent_bank = torch.load(
        latent_bank_path,
        map_location="cpu",
        weights_only=False,
    )

    metadata = pd.read_csv(
        metadata_path
    )

    validate_source_bank(
        latent_bank=latent_bank,
        metadata=metadata,
    )

    return latent_bank, metadata


def load_model_info(
    model_info_path: str | Path,
) -> dict[str, Any]:
    """
    Загружает JSON с информацией о генераторе.
    """

    model_info_path = Path(
        model_info_path
    )

    if not model_info_path.exists():
        raise FileNotFoundError(
            "Model info file was not found: "
            f"{model_info_path}"
        )

    with model_info_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        model_info = json.load(file)

    if not isinstance(model_info, dict):
        raise TypeError(
            "Model info must be a dictionary."
        )

    return model_info


def validate_source_bank(
    latent_bank: dict[str, Any],
    metadata: pd.DataFrame,
) -> None:
    """
    Проверяет внутреннюю согласованность
    latent bank и metadata.
    """

    required_bank_keys = {
        "format_version",
        "generator",
        "splits",
        "items",
    }

    missing_bank_keys = (
        required_bank_keys
        - set(latent_bank.keys())
    )

    if missing_bank_keys:
        raise KeyError(
            "Latent bank is missing keys: "
            f"{sorted(missing_bank_keys)}"
        )

    missing_metadata_columns = (
        REQUIRED_METADATA_COLUMNS
        - set(metadata.columns)
    )

    if missing_metadata_columns:
        raise KeyError(
            "Metadata is missing columns: "
            f"{sorted(missing_metadata_columns)}"
        )

    if metadata["image_id"].duplicated().any():
        duplicated_ids = (
            metadata.loc[
                metadata["image_id"].duplicated(),
                "image_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Metadata contains duplicated image IDs: "
            f"{duplicated_ids}"
        )

    if metadata["seed"].duplicated().any():
        duplicated_seeds = (
            metadata.loc[
                metadata["seed"].duplicated(),
                "seed",
            ]
            .tolist()
        )

        raise ValueError(
            "Metadata contains duplicated seeds: "
            f"{duplicated_seeds}"
        )

    bank_items = latent_bank["items"]

    if not isinstance(bank_items, dict):
        raise TypeError(
            "latent_bank['items'] must be a dictionary."
        )

    metadata_ids = set(
        metadata["image_id"].tolist()
    )

    bank_ids = set(
        bank_items.keys()
    )

    if metadata_ids != bank_ids:
        only_in_metadata = sorted(
            metadata_ids - bank_ids
        )

        only_in_bank = sorted(
            bank_ids - metadata_ids
        )

        raise ValueError(
            "Metadata and latent bank contain "
            "different object IDs. "
            f"Only in metadata: {only_in_metadata}; "
            f"only in bank: {only_in_bank}."
        )

    for image_id, item in bank_items.items():
        required_item_keys = {
            "seed",
            "split",
            "z",
            "w_plus",
        }

        missing_item_keys = (
            required_item_keys
            - set(item.keys())
        )

        if missing_item_keys:
            raise KeyError(
                f"{image_id} is missing keys: "
                f"{sorted(missing_item_keys)}"
            )

        if not isinstance(
            item["z"],
            torch.Tensor,
        ):
            raise TypeError(
                f"{image_id}: z must be a tensor."
            )

        if not isinstance(
            item["w_plus"],
            torch.Tensor,
        ):
            raise TypeError(
                f"{image_id}: w_plus must be a tensor."
            )

        if item["z"].ndim != 2:
            raise ValueError(
                f"{image_id}: expected z with shape "
                "[B, z_dim], got "
                f"{tuple(item['z'].shape)}."
            )

        if item["w_plus"].ndim != 3:
            raise ValueError(
                f"{image_id}: expected W+ with shape "
                "[B, num_ws, w_dim], got "
                f"{tuple(item['w_plus'].shape)}."
            )


def get_source_sample(
    image_id: str,
    latent_bank: dict[str, Any],
    metadata: pd.DataFrame,
    storage_dir: str | Path,
) -> SourceSample:
    """
    Возвращает один объект по image_id.
    """

    storage_dir = Path(
        storage_dir
    )

    bank_items = latent_bank["items"]

    if image_id not in bank_items:
        raise KeyError(
            f"Object was not found in latent bank: "
            f"{image_id}"
        )

    metadata_rows = metadata[
        metadata["image_id"] == image_id
    ]

    if len(metadata_rows) == 0:
        raise KeyError(
            f"Object was not found in metadata: "
            f"{image_id}"
        )

    if len(metadata_rows) > 1:
        raise ValueError(
            f"Metadata contains multiple rows for "
            f"{image_id}."
        )

    metadata_row = metadata_rows.iloc[0]
    bank_item = bank_items[image_id]

    image_path = (
        storage_dir
        / metadata_row["image_path"]
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Source PNG was not found: {image_path}"
        )

    metadata_seed = int(
        metadata_row["seed"]
    )

    bank_seed = int(
        bank_item["seed"]
    )

    if metadata_seed != bank_seed:
        raise ValueError(
            f"Seed mismatch for {image_id}: "
            f"metadata={metadata_seed}, "
            f"bank={bank_seed}."
        )

    metadata_split = str(
        metadata_row["split"]
    )

    bank_split = str(
        bank_item["split"]
    )

    if metadata_split != bank_split:
        raise ValueError(
            f"Split mismatch for {image_id}: "
            f"metadata={metadata_split}, "
            f"bank={bank_split}."
        )

    return SourceSample(
        image_id=image_id,
        seed=bank_seed,
        split=bank_split,
        z=bank_item["z"].clone(),
        w_plus=bank_item["w_plus"].clone(),
        image_path=image_path,
    )


def get_split_metadata(
    metadata: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """
    Возвращает строки указанного split,
    отсортированные по seed.
    """

    split_metadata = (
        metadata[
            metadata["split"] == split
        ]
        .sort_values("seed")
        .reset_index(drop=True)
    )

    if split_metadata.empty:
        raise ValueError(
            f"Split is empty or does not exist: {split}"
        )

    return split_metadata
