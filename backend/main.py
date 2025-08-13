import uvicorn
from fastapi import FastAPI, File, UploadFile
from modeling import get_modeling #model_training
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
        return {"message":"DataFrame received successfully",
                "error":"0"}
    except Exception as e:
        return {"error":str(e)}

@app.post("/target/{target}/epochs/{epochs}/features/{features}/test_split/{test_split}")
def modeling(
        target: str, epochs: int, features: str, test_split: float,
):
    try:
        results, setup, y_test = get_modeling(target, epochs, df, features, test_split)
        return {"results": results,
                "setup": setup,  # returns jsons of dataframes
                "y_test": y_test,
                "error": "0",
                }
    except Exception as e:
        return {"error":str(e)}

if __name__=="__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080)