# Použijeme oficiálny PyTorch obraz, ktorý už obsahuje CUDA a cuDNN
# Pre kompatibilitu s tvojím prostredím použijeme verziu s podporou CUDA 11.8
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Nastavíme pracovný adresár v kontajneri
WORKDIR /app

# Aktualizácia systému a inštalácia systémových závislostí
# OpenCV vyžaduje libgl1 a libglib2.0-0
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Nainštalujeme Python knižnice, ktoré si mal v environment.yml
RUN pip install --no-cache-dir \
    opencv-python \
    pandas \
    matplotlib \
    scikit-learn \
    tqdm \
    facenet-pytorch \
    insightface \
    onnxruntime

# Tento adresár sa pripojí z hostovského systému cez docker-compose,
# takže zatiaľ ho netreba celý kopírovať počas buildu (kvôli rýchlosti).
# COPY . /app/
