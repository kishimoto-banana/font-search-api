from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from app.config.settings import NUM_TOP_K
from app.domain.entity import PredictFont, Response
from app.domain.preprocess import Preprocessor
from PIL import Image
from torchvision import models


def fetch_vgg16() -> nn.Module:
    net = models.vgg16_bn()
    net.features[0] = nn.Conv2d(1, 64, 3, stride=1, padding=1)
    net.classifier[6] = nn.Linear(4096, 77)

    return net


class Predictor(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, image: Image) -> Response:
        raise NotImplementedError("Method not implemented")


class FontPredictor(Predictor):
    def __init__(self, preprocessor: Preprocessor, model: nn.Module) -> None:
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, image: Image) -> Response:
        patches = self.preprocessor(image)
        outputs = self.model(patches)
        top_fonts = torch.argsort(torch.mean(outputs, dim=0), descending=True)[
            :NUM_TOP_K
        ].numpy()

        # TODO: softmax
        return Response(fonts=[PredictFont(label=f, score=0.9) for f in top_fonts])
