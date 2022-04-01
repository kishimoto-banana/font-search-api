from lib2to3.pytree import Base
from fastapi import FastAPI
from pydantic import BaseModel

class Image(BaseModel):
    name: str
    content: str

app = FastAPI()

@app.post("/v1/fonts/")
async def predict_fonts(image: Image):
    return image.dict()

