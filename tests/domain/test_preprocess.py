import numpy as np
import pytest
import torch
from PIL import Image
from app.domain.transform import Transform
from app.domain.preprocess import resize, FontImagePreprocessor
from app.config.settings import PATCH_SIZE
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from app.domain.entity import BoundingBox


class MockTransform(Transform):
    def __init__(self) -> None:
        self.transform = {"val": A.Compose([ToTensorV2()])}

    def __call__(self, img: np.array, phase: str = "val") -> Tensor:
        return self.transform[phase](image=img)["image"]


def test_resize():
    # heightがpatch_sizeならリサイズしない
    img = Image.fromarray(np.random.random_sample((PATCH_SIZE, PATCH_SIZE)))
    resized_img, resize_ratio = resize(img)

    assert img.size == resized_img.size
    assert resize_ratio.x == pytest.approx(1.0) and resize_ratio.y == pytest.approx(1.0)

    # heightがpatch_sizeでないならheightをpatch_sizeに、アスペクト比を保つためwidthを変換
    height = 200
    width = 800
    img = Image.fromarray(np.random.random_sample((height, width)))

    ratio = PATCH_SIZE / height
    resize_width = int(width * ratio)
    resized_img, resize_ratio = resize(img)

    assert (resize_width, PATCH_SIZE) == resized_img.size
    assert resize_ratio.x == pytest.approx(ratio) and resize_ratio.y == pytest.approx(
        ratio
    )

    # heightがpatch_sizeでないならheightをpatch_sizeに、widthがPATCH_SIZE以下なら(PATCH_SIZE, PATCH_SIZE)
    height = 200
    width = 100
    img = Image.fromarray(np.random.random_sample((height, width)))

    resized_img, resize_ratio = resize(img)

    assert (PATCH_SIZE, PATCH_SIZE) == resized_img.size
    assert resize_ratio.x == pytest.approx(PATCH_SIZE / width)
    assert resize_ratio.y == pytest.approx(PATCH_SIZE / height)


def test_font_image_preprocessor():
    np_img = np.random.random_sample((PATCH_SIZE, PATCH_SIZE)).astype(np.float32)
    bouding_boxes = [
        BoundingBox(left=0, upper=0, right=PATCH_SIZE, lower=PATCH_SIZE),
        BoundingBox(left=0, upper=0, right=PATCH_SIZE, lower=PATCH_SIZE),
    ]
    img = Image.fromarray(np_img)

    transform = MockTransform()
    preprocessor = FontImagePreprocessor(transform)

    preprocess_img = preprocessor(img, bouding_boxes)

    expected_img = torch.from_numpy(
        np.stack([np.expand_dims(np_img, 0), np.expand_dims(np_img, 0)])
    )

    assert expected_img.size() == preprocess_img.size()
    assert torch.allclose(expected_img, preprocess_img)
