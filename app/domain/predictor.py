from abc import ABCMeta, abstractmethod
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from app.config.settings import FONT_LABEL_TO_META, NUM_TOP_K
from app.domain.entity import BoundingBox, PredictFont
from app.domain.preprocess import Preprocessor
from PIL.Image import Image
from torchvision import models


def fetch_vgg16() -> nn.Module:
    net = models.vgg16_bn(pretrained=False)
    net.features[0] = nn.Conv2d(1, 64, 3, stride=1, padding=1)
    net.classifier[6] = nn.Linear(4096, 365)

    return net


class Predictor(metaclass=ABCMeta):
    @abstractmethod
    def predict(
        self, image: Image, bounding_boxes: List[BoundingBox]
    ) -> List[PredictFont]:
        raise NotImplementedError("Method not implemented")


class MockPredictor(Predictor):
    def predict(
        self, image: Image, bounding_boxes: List[BoundingBox]
    ) -> List[PredictFont]:
        return [
            PredictFont(
                fontName="a",
                fontNameJa="a",
                fontNameEn="a",
                fontWeight=100,
                type="adobe",
                adobeId="asssa",
                score=0.1,
            )
        ]


class FontPredictor(Predictor):
    def __init__(self, preprocessor: Preprocessor, model: nn.Module) -> None:
        self.preprocessor = preprocessor
        self.model = model

    def predict(
        self, image: Image, bounding_boxes: List[BoundingBox]
    ) -> List[PredictFont]:
        patches = self.preprocessor(image, bounding_boxes)
        outputs = self.model(patches)
        agg_outputs = torch.mean(outputs, dim=0)
        top_fonts = torch.argsort(agg_outputs, descending=True)[:NUM_TOP_K].numpy()
        scores = F.softmax(agg_outputs, dim=0)[top_fonts].detach().numpy()
        return [
            PredictFont(
                fontName=FONT_LABEL_TO_META[f]["fontName"],
                fontNameJa=FONT_LABEL_TO_META[f]["fontNameJa"],
                fontNameEn=FONT_LABEL_TO_META[f]["fontNameEn"],
                fontWeight=FONT_LABEL_TO_META[f]["fontWeight"],
                type=FONT_LABEL_TO_META[f]["type"],
                adobeId=FONT_LABEL_TO_META[f]["adobeId"],
                score=round(s, 3),
            )
            for f, s in zip(top_fonts, scores)
        ]
