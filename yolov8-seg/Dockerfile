FROM python:3.8-slim

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    gcc \
    python3-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0

ENV PYTHONUNBUFFERED=True \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD exec gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
