FROM ubuntu:16.04

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev python3-tk

RUN pip install --upgrade pip
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app
