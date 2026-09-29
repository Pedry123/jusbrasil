# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/models/hf

WORKDIR /app

# dependências: torch CPU (índice pytorch) + demais pacotes
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# código e dados (base canônica + citações mescladas já geradas)
COPY src/ src/
COPY app/ app/
COPY data/ data/
COPY kaggle_metric.py json_to_submission.py ./

# treina o classificador e monta o índice FAISS no build (baixa o MiniLM)
RUN python -m src.train
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
