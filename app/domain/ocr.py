import base64
import json
import os
import tempfile
from abc import ABCMeta, abstractmethod
from io import BytesIO
from logging import getLogger
from typing import List, Tuple

import boto3
import requests
from app.domain.entity import BoundingBox
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from fastapi import HTTPException
from google.cloud import vision
from google.oauth2 import service_account
from msrest.authentication import CognitiveServicesCredentials

logger = getLogger("uvicorn")


class TextDetector(metaclass=ABCMeta):
    @abstractmethod
    def detect(self, content: str) -> Tuple[List[BoundingBox], str]:
        raise NotImplementedError("Method not implemented")


class TextDetectorAzure(TextDetector):
    def __init__(self) -> None:
        subscription_key = os.environ["AZURE_SUBSCRIPTION_KEY"]
        endpoint = os.environ["AZURE_OCR_ENDPOINT"]
        self.client = ComputerVisionClient(
            endpoint, CognitiveServicesCredentials(subscription_key)
        )

    def detect(self, content: str) -> Tuple[List[BoundingBox], str]:
        image = BytesIO(base64.b64decode(content))
        response = self.client.recognize_printed_text_in_stream(
            image=image, language="ja"
        )
        if not response.regions:
            raise HTTPException(status_code=400, detail="Text not found")

        text = ""
        bounding_boxes = []
        for region in response.regions:
            for line in region.lines:
                # print("Bounding box: {}".format(line.bounding_box))
                for word in line.words:
                    print(word.text)
                    print(word.bounding_box)
                    left, upper, width, height = map(int, word.bounding_box.split(","))
                    bounding_boxes.append(
                        BoundingBox(
                            left=left,
                            upper=upper,
                            right=left + width,
                            lower=upper + height,
                        )
                    )
                    text += word.text

        print(bounding_boxes, text)
        return bounding_boxes, text


class TextDetectorGcp(TextDetector):
    def __init__(self) -> None:
        ssm = boto3.client("ssm")
        param_name = os.environ["SSM_PARAM_NAME"]
        json_credentials = ssm.get_parameter(Name=param_name, WithDecryption=True)[
            "Parameter"
        ]["Value"]
        fp = tempfile.NamedTemporaryFile()
        fp.write(json_credentials.encode("utf-8"))
        fp.flush()
        cred = service_account.Credentials.from_service_account_file(fp.name)
        self.client = vision.ImageAnnotatorClient(credentials=cred)
        fp.close()

    def detect(self, content: str) -> Tuple[List[BoundingBox], str]:
        image = vision.Image(content=content)

        response = self.client.text_detection(image=image, timeout=10)
        response_json = type(response.full_text_annotation).to_json(
            response.full_text_annotation
        )
        response_dict = json.loads(response_json)

        if not response_dict["pages"]:
            raise HTTPException(status_code=400, detail="Text not found")

        text = response_dict["text"].replace("\n", "")
        bounding_boxes = []
        for paragraph in response_dict["pages"][0]["blocks"][0]["paragraphs"]:
            for words in paragraph["words"]:
                for symbol in words["symbols"]:
                    char = symbol["text"]
                    left = int(symbol["boundingBox"]["vertices"][0]["x"])
                    upper = int(symbol["boundingBox"]["vertices"][0]["y"])
                    right = int(symbol["boundingBox"]["vertices"][2]["x"])
                    lower = int(symbol["boundingBox"]["vertices"][2]["y"])

                    bounding_boxes.append(
                        BoundingBox(left=left, upper=upper, right=right, lower=lower)
                    )
        print(bounding_boxes)
        return bounding_boxes, text

    def detect_http(self, content: str):
        res = requests.post(
            "https://vision.googleapis.com/v1/images:annotate?key=AIzaSyAYz8Xz5Vai2ixq-XHs2lUAyxIElURd1nk",
            json={
                "requests": [
                    {
                        "image": {"content": content},
                        "features": [
                            {"type": "TEXT_DETECTION", "model": "builtin/latest"}
                        ],
                    }
                ]
            },
        )

        response = res.json()
        logger.info(response)

        if not response["responses"][0]:
            raise HTTPException(status_code=400, detail="Text not found")

        text = response["responses"][0]["fullTextAnnotation"]["text"].replace("\n", "")
        bounding_boxes = []
        for paragraph in response["responses"][0]["fullTextAnnotation"]["pages"][0][
            "blocks"
        ][0]["paragraphs"]:
            for words in paragraph["words"]:
                for symbol in words["symbols"]:
                    char = symbol["text"]
                    try:
                        left = (
                            int(symbol["boundingBox"]["vertices"][0]["x"])
                            if "x" in symbol["boundingBox"]["vertices"][0]
                            else int(symbol["boundingBox"]["vertices"][3]["x"])
                        )
                        upper = (
                            int(symbol["boundingBox"]["vertices"][0]["y"])
                            if "y" in symbol["boundingBox"]["vertices"][0]
                            else int(symbol["boundingBox"]["vertices"][1]["y"])
                        )
                        right = (
                            int(symbol["boundingBox"]["vertices"][2]["x"])
                            if "x" in symbol["boundingBox"]["vertices"][2]
                            else int(symbol["boundingBox"]["vertices"][1]["x"])
                        )
                        lower = (
                            int(symbol["boundingBox"]["vertices"][2]["y"])
                            if "y" in symbol["boundingBox"]["vertices"][2]
                            else int(symbol["boundingBox"]["vertices"][3]["y"])
                        )
                    except KeyError:
                        raise HTTPException(status_code=400, detail="Text not found")

                    bounding_boxes.append(
                        BoundingBox(left=left, upper=upper, right=right, lower=lower)
                    )
        print(bounding_boxes)
        return bounding_boxes, text
