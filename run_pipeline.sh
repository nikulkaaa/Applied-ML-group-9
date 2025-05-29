#!/usr/bin/env bash
set -euo pipefail

# 1) Install Miniconda locally if not avaliable since it is a prereq.
MINICONDA_DIR="${HOME}/miniconda3"
INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
URL="https://repo.anaconda.com/miniconda/${INSTALLER}"

if [ ! -d "$MINICONDA_DIR" ]; then
  echo "Installing Miniconda into $MINICONDA_DIR…"
  if command -v wget >/dev/null; then
    wget -q "$URL" -O /tmp/mcl.sh
  elif command -v curl >/dev/null; then
    curl -sL "$URL" -o /tmp/mcl.sh
  else
    echo "Error: neither wget nor curl is installed." >&2
    exit 1
  fi
  bash /tmp/mcl.sh -b -p "$MINICONDA_DIR"
  rm /tmp/mcl.sh
else
  echo "Miniconda already installed; skipping."
fi

# 2) Locate conda executable
if [ -x "${MINICONDA_DIR}/bin/conda" ]; then
  CONDA_BIN="${MINICONDA_DIR}/bin/conda"
elif [ -x "${MINICONDA_DIR}/condabin/conda" ]; then
  CONDA_BIN="${MINICONDA_DIR}/condabin/conda"
elif [ -x "${MINICONDA_DIR}/Scripts/conda.exe" ]; then
  CONDA_BIN="${MINICONDA_DIR}/Scripts/conda.exe"
else
  echo "Error: could not find conda under $MINICONDA_DIR" >&2
  exit 1
fi
echo "Using conda at: $CONDA_BIN"
export CONDA_BIN

# 3a) Preprocessing env (Python 3.8)
if ! "$CONDA_BIN" env list | grep -qE "^preproc_env"; then
  echo "[1/2] Creating preproc_env (Python 3.8)…"
  "$CONDA_BIN" create -y -n preproc_env python=3.8 pip

  # Install dlib from the root of the repo
  echo " - Installing dlib 19.22.99 from wheel…"
  "$CONDA_BIN" run -n preproc_env --no-capture-output \
    python -m pip install "./dlib-19.22.99-cp38-cp38-win_amd64.whl"

  echo " - Installing other preprocessing requirements…"
  "$CONDA_BIN" run -n preproc_env --no-capture-output \
    pip install -r requirements_preproc.txt
else
  echo "preproc_env exists; skipping."
fi

# 3b) Prediction env (Python 3.10)
if ! "$CONDA_BIN" env list | grep -qE "^predict_env"; then
  echo "[2/2] Creating predict_env (Python 3.10)…"
  "$CONDA_BIN" create -y -n predict_env python=3.10 pip

  echo " - Installing CPU-only PyTorch…"
  "$CONDA_BIN" run -n predict_env --no-capture-output \
    pip install torch==2.7.0 \
      -f https://download.pytorch.org/whl/cpu/torch_stable.html

  echo " - Installing other prediction requirements…"
  "$CONDA_BIN" run -n predict_env --no-capture-output \
    pip install -r requirements_predict.txt

  echo " - Installing FastAPI/Streamlit stack…"
  "$CONDA_BIN" run -n predict_env --no-capture-output \
    pip install fastapi uvicorn streamlit python-multipart requests pillow
else
  echo "predict_env exists; skipping."
fi

# 4) Launch FastAPI in background (via predict_env)
echo "Starting FastAPI on http://localhost:8000"
"$CONDA_BIN" run -n predict_env --no-capture-output \
  uvicorn app:app --reload --port 8000 &
UVICORN_PID=$!

sleep 2

# 5) Launch Streamlit in foreground (via predict_env)
echo "Starting Streamlit on http://localhost:8501"
"$CONDA_BIN" run -n predict_env --no-capture-output \
  streamlit run streamlit_app.py

# 6) Teardown: kill FastAPI when Streamlit exits
echo "Stopping FastAPI (pid=$UVICORN_PID)"
kill $UVICORN_PID
