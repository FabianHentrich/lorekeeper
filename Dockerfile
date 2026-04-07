FROM python:3.11-slim

# curl is needed for the HEALTHCHECK; build without it would leave docker ps
# status permanently "starting".
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config/ config/

# Ensure the HF cache dir is writable by the non-root user
RUN mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app /app
ENV HF_HOME=/home/app/.cache/huggingface

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
