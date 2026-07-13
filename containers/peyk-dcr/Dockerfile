# No model, no GPU — direct PDF text-layer extraction only (pypdfium2).
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run.py .

ENTRYPOINT ["python3", "run.py"]
