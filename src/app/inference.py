import pandas as pd
import os
import mlflow
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException

# Load the environment variables from the .env file into the application
load_dotenv()
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

# Initialize the FastAPI application
app = FastAPI()


# Create a class to store the deployed model & use it for prediction
class Model:
    def __init__(self, model_name, model_stage):
        """
        To initialize the model
        model_name: Name of the model in registry
        model_stage: Stage of the model
        """
        # Load the model from Registry
        self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")

    def predict(self, data):
        """
        To use the loaded model to make predictions on the data
        data: Pandas DataFrame to perform predictions
        """
        predictions = self.model.predict(data)
        return predictions


# Create model
model = Model("real_estate_lgbm", "Staging")

# Create the POST endpoint with path '/invocations'
@app.post("/invocations")
async def create_upload_file(file: UploadFile = File(...)):
    # Handle the file only if it is a CSV
    if file.filename.endswith(".csv"):
        # Create a temporary file with the same name as the uploaded
        # CSV file to load the data into a pandas Dataframe
        with open(file.filename, "wb")as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)
        os.remove(file.filename)        # Return a JSON object containing the model predictions
        return list(model.predict(data))

    else:
        # Raise a HTTP 400 Exception, indicating Bad Request
        # (you can learn more about HTTP response status codes here)
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")


# Check if the environment variables for AWS access are available.
# If not, exit the program
if os.getenv("AWS_ACCESS_KEY_ID") is None or os.getenv("AWS_SECRET_ACCESS_KEY") is None:
    exit(1)