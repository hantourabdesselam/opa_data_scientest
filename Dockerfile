FROM python:3.8-slim

COPY ./requirements.txt ./requirements.txt

COPY ./main.py ./main.py
COPY ./models.py ./models.py

RUN apt-get update && apt-get install -y default-libmysqlclient-dev \
    build-essential \
    pkg-config

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN rm requirements.txt

EXPOSE 8000

CMD uvicorn main:api --host 0.0.0.0
