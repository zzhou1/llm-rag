FROM python

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /root/.cache/gpt4all
COPY rift-coder-v0-7b-q4_0.gguf /root/.cache/gpt4all
COPY main.py .