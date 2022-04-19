import base64
import random
from io import BytesIO

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from mangum import Mangum

from app.config.settings import MODEL_PATH, NUM_PATCHES
from app.config.constant import OCR_RESPONSE_BODY
from app.domain.entity import Request, Response
from app.domain.predictor import FontPredictor, fetch_vgg16
from app.domain.preprocess import FontImagePreprocessor
from app.domain.transform import FontImageTranform

torch.manual_seed(5)
np.random.seed(5)
random.seed(5)


def base64_to_pil(img_str: str, gray: bool = False) -> Image:
    img_raw = base64.b64decode(img_str)
    if gray:
        return Image.open(BytesIO(img_raw)).convert("L")

    return Image.open(BytesIO(img_raw))


app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://font-search.vercel.app/",
    "https://font-search.vercel.app",
    "https://fontpint.com",
    "https://fontpint.com/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://font-search.*kishimoto-banana.vercel.app.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    net = fetch_vgg16()
    net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    net.eval()

    transform = FontImageTranform()
    preprocessor = FontImagePreprocessor(transform=transform, num_patchs=NUM_PATCHES)
    predictor = FontPredictor(preprocessor=preprocessor, model=net)
    app.state.predictor = predictor


@app.post("/v1/fonts/", response_model=Response)
async def predict_fonts(req: Request):
    content = req.content
    image = base64_to_pil(content, gray=True)

    return app.state.predictor.predict(image)


@app.post("/mock/ocr/")
async def mock_ocr():
    return OCR_RESPONSE_BODY


handler = Mangum(app)
