import json
import os
import tempfile
from dataclasses import dataclass

import boto3
from google.cloud import vision
from google.oauth2 import service_account


@dataclass
class BoundingBox:
    left: int
    upper: int
    right: int
    lower: int


class TextDetector:
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

    def detect(self, content):
        image = vision.Image(content=content)

        response = self.client.text_detection(image=image, timeout=10)
        response_json = type(response.full_text_annotation).to_json(
            response.full_text_annotation
        )
        response_dict = json.loads(response_json)

        char_bbs = []
        for paragraph in response_dict["pages"][0]["blocks"][0]["paragraphs"]:
            for words in paragraph["words"]:
                for symbol in words["symbols"]:
                    char = symbol["text"]
                    left = int(symbol["boundingBox"]["vertices"][0]["x"])
                    upper = int(symbol["boundingBox"]["vertices"][0]["y"])
                    right = int(symbol["boundingBox"]["vertices"][2]["x"])
                    lower = int(symbol["boundingBox"]["vertices"][2]["y"])

                    char_bb = {
                        "char": char,
                        "bb": BoundingBox(
                            left=left, upper=upper, right=right, lower=lower
                        ),
                    }
                    char_bbs.append(char_bb)

        print(char_bbs)
