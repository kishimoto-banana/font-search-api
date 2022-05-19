import base64
import random
from io import BytesIO
from logging import getLogger

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from PIL import Image

from app.config.constant import OCR_RESPONSE_BODY
from app.config.settings import MODEL_PATH
from app.domain.entity import Request, Response
from app.domain.ocr import TextDetectorAzure, TextDetectorGcp
from app.domain.predictor import FontPredictor, fetch_vgg16
from app.domain.preprocess import FontImagePreprocessor
from app.domain.transform import FontImageTranform

load_dotenv()

logger = getLogger("uvicorn")

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
    logger.info("fetch vgg16")
    net = fetch_vgg16()
    logger.info("load weight")
    net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    logger.info("eval")
    net.eval()

    transform = FontImageTranform()
    preprocessor = FontImagePreprocessor(transform=transform)
    predictor = FontPredictor(preprocessor=preprocessor, model=net)
    app.state.predictor = predictor

    text_detector = TextDetectorGcp()
    app.state.text_detector = text_detector
    text_detector_azure = TextDetectorAzure()
    app.state.text_detector_azure = text_detector_azure


@app.post("/v1/fonts/", response_model=Response)
async def predict_fonts(req: Request):
    content = req.content

    # OCR
    # bounding_boxes, text = app.state.text_detector.detect(content)
    bounding_boxes, text = app.state.text_detector.detect_http(content)
    # bounding_boxes, text = app.state.text_detector_azure.detect(content)

    # フォント認識
    image = base64_to_pil(content, gray=True)
    fonts = app.state.predictor.predict(image, bounding_boxes)
    logger.info(fonts)
    return Response(text=text, fonts=fonts)


@app.post("/mock/ocr/")
async def mock_ocr():
    return OCR_RESPONSE_BODY


@app.get("/health")
async def health():
    return {"status": "OK"}


handler = Mangum(app)
