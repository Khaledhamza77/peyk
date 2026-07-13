# No CUDA/GPU needed: every backend is a remote API client (Bedrock, Vertex AI) — same
# lightweight pattern as peyk-dcr.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY prompts.py .
COPY backends/ backends/
COPY run.py .

ENTRYPOINT ["python3", "run.py"]
