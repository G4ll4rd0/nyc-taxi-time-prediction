'''
Main App File
'''
import pickle
from dataclasses import dataclass

import dagshub
import mlflow
import uvicorn
from fastapi import FastAPI
from mlflow import MlflowClient
from pydantic import BaseModel

##### MLflow settings #####
DAGSHUB_REPO = "https://dagshub.com/G4ll4rd0/nyc-taxi-time-prediction"

dagshub.init(url=DAGSHUB_REPO, mlflow=True) # type: ignore

MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

run_ = mlflow.search_runs(order_by=['metrics.rmse ASC'],
                          output_format="list",
                          experiment_names=["nyc-taxi-experiment-prefect"]
                          )[0]
run_id = run_.info.run_id
run_id = '51fdab30ac6d4e9297625835169f2004'

run_uri = f"runs:/{run_id}/preprocessor"

client.download_artifacts(
    run_id=run_id,
    path='preprocessor',
    dst_path='.'
)

with open("preprocessor/preprocessor.b", "rb") as f_in:
    dv = pickle.load(f_in)

MODEL_NAME = "nyc-taxi-model"
ALIAS = "champion"

model_uri = f"models:/{MODEL_NAME}@{ALIAS}"

champion_model = mlflow.pyfunc.load_model(
    model_uri=model_uri
)

##### FUNCS #####
def preprocess(input_data):
    '''Preprocess data'''
    input_dict = {
        'PU_DO': input_data.PULocationID + "_" + input_data.DOLocationID,
        'trip_distance': input_data.trip_distance,
    }

    return dv.transform(input_dict)

def predict(input_data):
    '''Predicts data'''
    x_val = preprocess(input_data)

    return champion_model.predict(x_val)

##### APP #####
app = FastAPI()

@dataclass
class InputData(BaseModel):
    '''Class used to send prediction data in API'''
    PULocationID: str # pylint: disable=invalid-name
    DOLocationID: str # pylint: disable=invalid-name
    trip_distance: float

@app.post("/predict")
def predict_endpoint(input_data: InputData):
    '''Endpoint used for predictions'''
    result = predict(input_data)[0]
    return {"prediction": float(result)}
