FROM python as builder

COPY build-requirements.txt /build/
WORKDIR /build
RUN pip install --no-cache-dir -r build-requirements.txt

COPY pdfs /build/pdfs
RUN mkdir -p /root/.cache/gpt4all
COPY all-MiniLM-L6-v2-f16.gguf /root/.cache/gpt4all/

COPY generate_index.py /build/
RUN python generate_index.py

FROM python

COPY run-requirements.txt .
RUN pip install --no-cache-dir -r run-requirements.txt

COPY --from=builder /build/faiss .

COPY --from=builder /root/.cache/gpt4all/ /root/.cache/gpt4all/

RUN mkdir -p /models
COPY mistral-7b-openorca.gguf2.Q4_0.gguf /models/

COPY templates /templates
COPY main.py .

ENTRYPOINT [ "python", "main.py" ]
