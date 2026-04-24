# DocuBot — see README "Deployment" → Docker
# Build:  docker build -t docubot .
# Run:    docker run --name docubot --rm -p 8501:8501 --env-file .env -e DOCUBOT_DATA_DIR=/data -v docubot_data:/data docubot
# Persist user DB + all per-user Chroma indexes by mounting a volume at DOCUBOT_DATA_DIR (/data).
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV DOCUBOT_DATA_DIR=/data

EXPOSE 8501

# README: use PORT (PaaS) or STREAMLIT_SERVER_PORT, else 8501; listen on 0.0.0.0
CMD sh -c 'streamlit run app.py \
  --server.port=${PORT:-${STREAMLIT_SERVER_PORT:-8501}} \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false'
