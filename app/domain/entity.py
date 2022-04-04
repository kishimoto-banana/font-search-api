from dataclasses import dataclass
from typing import List

from pydantic import BaseModel


@dataclass
class PredictFont:
    label: int
    score: float


class Request(BaseModel):
    content: str


class Response(BaseModel):
    fonts: List[PredictFont]
