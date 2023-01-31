FROM python:3.10

WORKDIR /code

RUN pip install --upgrade pip

COPY ./Docker/model_service/requrements.txt /code/

RUN pip install -r code/requirements.txt

COPY ../../src/app /code/app
COPY ./Docker/model_service/.env /code/app/.env

CMD ["uvicorn", "app.inference:app", "--host", "0.0.0.0", "--port", "80"]