import random
from abc import ABCMeta, abstractclassmethod
from typing import Any

import numpy as np
import torch
from app.config.settings import PATCH_SIZE
from app.domain.transform import Transform
from PIL import Image
from torch import Tensor


def resize(img: Image) -> Image:
    if img.size[1] == PATCH_SIZE:
        return img
    y_ratio = PATCH_SIZE / img.size[1]
    width = int(img.size[0] * y_ratio)
    if width < PATCH_SIZE:
        return img.resize((PATCH_SIZE, PATCH_SIZE))
    return img.resize((width, PATCH_SIZE))


class Preprocessor(metaclass=ABCMeta):
    @abstractclassmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError("Method not implemented")


class FontImagePreprocessor:
    def __init__(self, transform: Transform, num_patchs: int) -> None:
        self.transform = transform
        self.num_patches = num_patchs

    def __call__(self, img: np.array) -> Tensor:
        resize_img = resize(img)
        img_transformed = self.transform(np.array(resize_img), "val")
        patchs = []
        for _ in range(self.num_patches):
            patch_x = random.randint(0, img_transformed.shape[-1] - PATCH_SIZE)
            patch = img_transformed[:, :, patch_x : patch_x + PATCH_SIZE]
            patchs.append(patch)

        return torch.stack(patchs)
