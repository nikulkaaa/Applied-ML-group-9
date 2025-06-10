# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=continuumio/miniconda3:24.11.1-0
FROM --platform=$BUILDPLATFORM ${BASE_IMAGE} AS base

ARG TARGETPLATFORM TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools needed for dlib etc.
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy application files
COPY requirements_preproc.txt  requirements_preproc.txt
COPY requirements_predict.txt  requirements_predict.txt
COPY dlib-19.22.0.tar.gz       dlib-19.22.0.tar.gz
COPY app/          ./app
COPY checkpoints/  ./checkpoints
COPY DECA/         ./DECA
COPY app.py        ./app.py

# Copy entrypoint script (for api service)
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

SHELL ["/bin/bash", "-c"]

# Create conda environments with arch-aware installs
RUN conda create -y -n deca_env python=3.8 pip && \
    conda run -n deca_env bash -c '\
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 && \
        if [[ "$TARGETARCH" == "amd64" ]]; then \
            pip install pytorch3d==0.7.8+pt2.1.0cpu --extra-index-url https://miropsota.github.io/torch_packages_builder ; \
        fi && \
        pip install numpy==1.23 scipy "scikit-image>=0.15" opencv-python PyYAML==5.1.1 face-alignment==1.3.4 yacs==0.1.8 ninja fvcore chumpy kornia tqdm'

RUN conda create -y -n preproc_env python=3.8 pip && \
    conda run -n preproc_env bash -c '\
        pip install /workspace/dlib-19.22.0.tar.gz && \
        pip install -r /workspace/requirements_preproc.txt'

RUN conda create -y -n predict_env python=3.10 pip && \
    conda run -n predict_env bash -c '\
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 && \
        pip install -r /workspace/requirements_predict.txt && \
        pip install fastapi uvicorn streamlit python-multipart requests pillow'

EXPOSE 8000 8501

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
