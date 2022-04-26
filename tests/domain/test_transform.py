import numpy as np
import torch
from app.domain.transform import FontImageTranform


def test_transform():
    img = np.array([[1, 2], [3, 4]], dtype=np.float32)
    expected_img = torch.from_numpy(img / 255)

    transform = FontImageTranform()
    trans_img = transform(img, phase="val")

    assert torch.allclose(expected_img, trans_img)
