# DocuBot — see README "Deployment" → Docker
# Build:  docker build -t docubot .
# Run:    docker run --rm -p 8501:8501 --env-file .env -v docubot_chroma:/data/chroma docubot
# Chroma: persist by mounting a volume at /data/chroma. The app uses DOCUBOT_CHROMA_PATH below;
#         if your .env sets DOCUBOT_CHROMA_PATH=./vector_db, that overrides this — set /data/chroma
#         in .env for Docker, or add -e DOCUBOT_CHROMA_PATH=/data/chroma after --env-file.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV DOCUBOT_CHROMA_PATH=/data/chroma

EXPOSE 8501

# README: use PORT (PaaS) or STREAMLIT_SERVER_PORT, else 8501; listen on 0.0.0.0
CMD sh -c 'streamlit run app.py \
  --server.port=${PORT:-${STREAMLIT_SERVER_PORT:-8501}} \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false'
