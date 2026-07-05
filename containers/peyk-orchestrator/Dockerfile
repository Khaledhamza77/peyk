# Lightweight, no GPU model of its own — dispatches the other five containers via
# `docker run` rather than loading any model itself (architecture_proposal.md 4.2/4.7).
# Needs the host's docker socket mounted at runtime so it can invoke sibling containers:
#   docker run -v /var/run/docker.sock:/var/run/docker.sock ...
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    docker-cli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py pipeline.py stages.py run.py ./
COPY config/ config/

ENTRYPOINT ["python3", "run.py"]
