FROM python

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /models
COPY rift-coder-v0-7b-q4_0.gguf /models

COPY pdfs /pdfs

COPY main.py .