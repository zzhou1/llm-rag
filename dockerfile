FROM python

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /models
COPY mistral-7b-openorca.gguf2.Q4_0.gguf /models
RUN mkdir -p /root/.cache/gpt4all
COPY all-MiniLM-L6-v2-f16.gguf /root/.cache/gpt4all
COPY pdfs /pdfs

RUN mkdir /templates
COPY templates/* /templates
COPY main.py .

ENTRYPOINT [ "python", "main.py" ]