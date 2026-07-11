FROM python:3.11-slim AS app-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY app ./app
COPY main.py pipeline.py ./

FROM app-base AS api

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM ghcr.io/open-webui/pipelines:main AS pipelines

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/mtbank-requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/mtbank-requirements.txt

COPY app /app/pipelines/app
COPY app /app/app
COPY pipeline.py /app/pipelines/mtbank_ai_transcription.py

ENV PIPELINES_DIR=/app/pipelines \
    PYTHONPATH=/app:/app/pipelines
EXPOSE 9099
