import uvicorn
from fastapi import FastAPI, File, UploadFile
from modeling import get_modeling
import io
import pandas as pd

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome from the AutoML API"}

@app.post("/receive_df")
async def receive_df(file: UploadFile = File(...)):
    global df
    try:
        # read contents
        contents = await file.read()

        # buffer
        parquet_buffer = io.BytesIO(contents)

        # read data into df
        df = pd.read_parquet(parquet_buffer, engine='pyarrow')
        return {"message":"DataFrame received successfully"}
    except Exception as e:
        return {"error":str(e)}

@app.post("/target/{target}/epochs/{epochs}/features/{features}/test_split/{test_split}/batch_size/{batch_size}")
def modeling(
        target: Any, epochs: int, features: Any|None, test_split: float, batch_size: int,
):
    try:
        results, setup = get_modeling(target, epochs, df, features, test_split, batch_size)
        return {"results": results,"setup": setup}      #  returns jsons of dataframes
    except Exception as e:
        return {"error":str(e)}