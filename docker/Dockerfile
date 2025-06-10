FROM continuumio/miniconda3:latest

SHELL ["/bin/bash", "-c"]

WORKDIR /workspace

# --- Install build tools for dlib ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake libgl1 && \
    rm -rf /var/lib/apt/lists/*


# Copy requirements and dlib files
COPY requirements_preproc.txt requirements_preproc.txt
COPY requirements_predict.txt requirements_predict.txt
COPY dlib-19.22.0.tar.gz dlib-19.22.0.tar.gz
COPY app ./app
COPY checkpoints ./checkpoints
COPY DECA ./DECA
COPY app.py ./app.py

# Set up conda environments

# DECA environment
RUN conda create -y -n deca_env python=3.8 pip && \
    conda run -n deca_env pip install torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu \
      --index-url https://download.pytorch.org/whl/cpu && \
    conda run -n deca_env pip install pytorch3d==0.7.8+pt2.1.0cpu --extra-index-url https://miropsota.github.io/torch_packages_builder && \
    conda run -n deca_env pip install numpy==1.23 scipy "scikit-image>=0.15" opencv-python PyYAML==5.1.1 face-alignment==1.3.4 yacs==0.1.8 ninja fvcore chumpy kornia tqdm

# Preproc environment
RUN conda create -y -n preproc_env python=3.8 pip && \
    conda run -n preproc_env pip install ./dlib-19.22.0.tar.gz && \
    conda run -n preproc_env pip install -r requirements_preproc.txt

# Predict environment
RUN conda create -y -n predict_env python=3.10 pip && \
    conda run -n predict_env pip install torch==2.7.0 \
      -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    conda run -n predict_env pip install -r requirements_predict.txt && \
    conda run -n predict_env pip install fastapi uvicorn streamlit python-multipart requests pillow

EXPOSE 8000 8501
