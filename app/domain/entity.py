from dataclasses import dataclass
from typing import List

from pydantic import BaseModel


class PredictFont(BaseModel):
    fontName: str
    fontNameJa: str
    fontNameEn: str
    fontWeight: int
    type: str
    adobeId: str
    score: float


class Request(BaseModel):
    content: str


class Response(BaseModel):
    text: str
    fonts: List[PredictFont]


@dataclass
class ResizeRatio:
    x: float
    y: float


@dataclass
class BoundingBox:
    left: int
    upper: int
    right: int
    lower: int

    def resize(self, ratio: ResizeRatio):
        self.left = int(self.left * ratio.x)
        self.upper = int(self.upper * ratio.y)
        self.right = int(self.right * ratio.x)
        self.lower = int(self.lower * ratio.y)

    def add_margin(self, margin: int, image_width: int, image_height: int):
        self.left = max(0, self.left - margin)
        self.upper = max(0, self.upper - margin)
        self.right = min(image_width, self.right + margin)
        self.lower = min(image_height, self.lower + margin)
