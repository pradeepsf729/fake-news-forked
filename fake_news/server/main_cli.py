import sys
sys.path.append('/root/fake-news') # replace with project root dir
import logging
from typing import List

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic.v1 import BaseSettings
from pydantic.main import BaseModel
from fake_news.model.tree_based import RandomForestModel
from fake_news.utils.features import construct_datapoint

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG
)
LOGGER = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_dir: str


app = FastAPI()
settings = Settings()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
config = {
    "model_output_path": "/root/fake-news/model_checkpoints/random_forest",
    "featurizer_output_path": "/root/fake-news/model_checkpoints/random_forest"
}
model = RandomForestModel(config)


class Statement(BaseModel):
    text: str


class Prediction(BaseModel):
    label: float
    probs: List[float]


#@app.post("/api/predict-fakeness", response_model=Prediction)

def predict_fakeness(text):
    datapoint = construct_datapoint(text)
    probs = model.predict([datapoint])
    #LOGGER.info(probs)
    label = np.argmax(probs, axis=1)
    LOGGER.info("Label - predicted : %s " ,  label)
    prediction = Prediction(label=label[0], probs=list(probs[0]))
    LOGGER.info(prediction)
    return prediction

if __name__ == '__main__':
    text = sys.argv[1]
    #stmt = Statement(text)
    pred = predict_fakeness(text)