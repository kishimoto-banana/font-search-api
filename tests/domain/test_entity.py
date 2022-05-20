from app.domain.entity import BoundingBox
from app.domain.entity import ResizeRatio
import pytest

from app.domain.preprocess import resize


def test_bounding_box():
    left = 4
    upper = 4
    right = 100
    lower = 100
    bbox = BoundingBox(left=left, upper=upper, right=right, lower=lower)
    resize_ratio = ResizeRatio(x=0.9, y=0.9)

    # resize
    bbox.resize(resize_ratio)
    assert bbox.left == int(left * resize_ratio.x)
    assert bbox.upper == int(upper * resize_ratio.y)
    assert bbox.right == int(right * resize_ratio.x)
    assert bbox.lower == int(lower * resize_ratio.y)

    # maring
    image_w = int(101 * resize_ratio.x)
    image_h = int(101 * resize_ratio.y)

    bbox.add_margin(5, image_height=image_h, image_width=image_w)
    assert bbox.left == 0
    assert bbox.upper == 0
    assert bbox.right == image_w
    assert bbox.lower == image_h
