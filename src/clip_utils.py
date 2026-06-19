
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPModel,
)


class CLIPEncoder(torch.nn.Module):
    """
    Обёртка над предобученным CLIP.

    Класс отвечает за:

    - загрузку CLIP и tokenizer;
    - заморозку параметров CLIP;
    - подготовку изображений StyleGAN2;
    - кодирование текста;
    - кодирование изображений.

    Text и image embeddings возвращаются
    L2-нормализованными.
    """

    def __init__(
        self,
        model_id: str,
        device: torch.device | str,
    ) -> None:
        super().__init__()

        self.model_id = model_id
        self.device_name = torch.device(device)

        self.model = CLIPModel.from_pretrained(
            model_id
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id
        )

        image_processor = (
            CLIPImageProcessor.from_pretrained(
                model_id
            )
        )

        self.image_size = int(
            self.model.config.vision_config.image_size
        )

        self.projection_dim = int(
            self.model.config.projection_dim
        )

        image_mean = torch.tensor(
            image_processor.image_mean,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        image_std = torch.tensor(
            image_processor.image_std,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        # Буферы переносятся на GPU вместе с классом,
        # но не считаются обучаемыми параметрами.
        self.register_buffer(
            "image_mean",
            image_mean,
            persistent=False,
        )

        self.register_buffer(
            "image_std",
            image_std,
            persistent=False,
        )

        self.model.eval()
        self.model.requires_grad_(False)

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.to(self.device_name)

    def preprocess_images(
        self,
        stylegan_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Подготавливает изображения StyleGAN2 для CLIP.

        Parameters
        ----------
        stylegan_images:
            Tensor формы [B, 3, H, W],
            диапазон значений [-1, 1].

        Returns
        -------
        torch.Tensor
            Tensor формы [B, 3, image_size, image_size],
            нормализованный для CLIP.

        В функции нет detach(), NumPy или PIL,
        поэтому gradient может пройти обратно к W+.
        """

        if stylegan_images.ndim != 4:
            raise ValueError(
                "Expected images with shape "
                "[B, C, H, W], got "
                f"{tuple(stylegan_images.shape)}."
            )

        if stylegan_images.shape[1] != 3:
            raise ValueError(
                "CLIP expects three-channel RGB images."
            )

        if not torch.is_floating_point(
            stylegan_images
        ):
            raise TypeError(
                "StyleGAN images must be floating-point tensors."
            )

        # StyleGAN2: [-1, 1] → [0, 1].
        images = (
            stylegan_images + 1.0
        ) / 2.0

        images = images.clamp(
            min=0.0,
            max=1.0,
        )

        images = F.interpolate(
            images,
            size=(
                self.image_size,
                self.image_size,
            ),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        images = (
            images - self.image_mean
        ) / self.image_std

        return images

    @torch.no_grad()
    def encode_text(
        self,
        prompts: Sequence[str],
    ) -> torch.Tensor:
        """
        Кодирует список prompts.

        Gradient для текста не нужен, потому что
        текстовый prompt во время оптимизации фиксирован.
        """

        prompts = list(prompts)

        if not prompts:
            raise ValueError(
                "The prompt list must not be empty."
            )

        if not all(
            isinstance(prompt, str)
            and prompt.strip()
            for prompt in prompts
        ):
            raise ValueError(
                "Every prompt must be a non-empty string."
            )

        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        tokens = {
            key: value.to(self.device_name)
            for key, value in tokens.items()
        }

        # Вызываем внутренние компоненты явно,
        # чтобы не зависеть от формата возврата
        # get_text_features() в разных версиях Transformers.
        text_outputs = self.model.text_model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens.get(
                "attention_mask"
            ),
        )

        pooled_text = text_outputs.pooler_output

        text_features = (
            self.model.text_projection(
                pooled_text
            )
        )

        text_features = F.normalize(
            text_features,
            p=2,
            dim=-1,
        )

        return text_features

    def encode_images(
        self,
        stylegan_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Кодирует изображения StyleGAN2.

        Здесь нет torch.no_grad(), так как gradient
        должен пройти:

        CLIP loss → CLIP image encoder → image → W+.
        """

        pixel_values = self.preprocess_images(
            stylegan_images
        )

        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values
        )

        pooled_image = (
            vision_outputs.pooler_output
        )

        image_features = (
            self.model.visual_projection(
                pooled_image
            )
        )

        image_features = F.normalize(
            image_features,
            p=2,
            dim=-1,
        )

        return image_features

    def count_trainable_parameters(
        self,
    ) -> int:
        """
        Возвращает число обучаемых параметров CLIP.
        Для замороженной модели должно быть 0.
        """

        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
