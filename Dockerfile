# Imagem para rodar o frontend Streamlit do projeto (frontend/app.py).
# Build:  docker compose up --build
# Acesso: http://localhost:8501

FROM python:3.11-slim

WORKDIR /app

# libgl1/libglib2.0-0: exigidos pelo OpenCV/MediaPipe para processar vídeo.
# ffmpeg: usado pelo librosa/soundfile para ler alguns formatos de áudio.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
