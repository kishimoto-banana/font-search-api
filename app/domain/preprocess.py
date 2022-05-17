from abc import ABCMeta, abstractclassmethod
from typing import Any, List, Tuple

import numpy as np
import torch
from app.config.settings import PATCH_MARGIN, PATCH_SIZE
from app.domain.entity import BoundingBox, ResizeRatio
from app.domain.transform import Transform
from PIL.Image import Image
from torch import Tensor


def resize(img: Image) -> Tuple[Image, ResizeRatio]:
    if img.size[1] == PATCH_SIZE:
        return img, ResizeRatio(x=1.0, y=1.0)
    y_ratio = PATCH_SIZE / img.size[1]
    width = int(img.size[0] * y_ratio)
    if width < PATCH_SIZE:
        x_ratio = PATCH_SIZE / img.size[0]
        return img.resize((PATCH_SIZE, PATCH_SIZE)), ResizeRatio(x=x_ratio, y=y_ratio)
    return img.resize((width, PATCH_SIZE)), ResizeRatio(x=y_ratio, y=y_ratio)


class Preprocessor(metaclass=ABCMeta):
    @abstractclassmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError("Method not implemented")


class FontImagePreprocessor(Preprocessor):
    def __init__(self, transform: Transform) -> None:
        self.transform = transform

    def __call__(self, img: Image, bounding_boxes: List[BoundingBox]) -> Tensor:
        resize_img, resize_ratio = resize(img)

        patchs = []
        image_w, image_h = resize_img.size
        for bbox in bounding_boxes:
            bbox.resize(resize_ratio)
            bbox.add_margin(
                margin=PATCH_MARGIN, image_width=image_w, image_height=image_h
            )

            cropped = resize_img.crop(
                (bbox.left, bbox.upper, bbox.right, bbox.lower)
            ).resize((PATCH_SIZE, PATCH_SIZE))
            img_transformed = self.transform(np.array(cropped), "val")
            patchs.append(img_transformed)

        return torch.stack(patchs)
