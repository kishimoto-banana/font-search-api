from dataclasses import dataclass
from typing import List

from pydantic import BaseModel


class PredictFont(BaseModel):
    fontName: str
    fontNameJa: str
    fontNameEn: str
    fontWeight: int
    score: float


class Request(BaseModel):
    content: str


class Response(BaseModel):
    fonts: List[PredictFont]
