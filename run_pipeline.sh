#!/usr/bin/env bash
set -euo pipefail

# 1) Install Miniconda locally if not available since it is a prereq.
# Miniconda must be installed at C:\Users\[insert_your_user_here]\miniconda3
MINICONDA_DIR="${HOME}/miniconda3"
# Default to Linux; we’ll override on Windows/MSYS
INSTALLER_NAME="Miniconda3-latest-Linux-x86_64.sh"
INSTALLER_FLAGS=(-b)
INSTALLER_URL_BASE="https://repo.anaconda.com/miniconda"

case "$(uname -s)" in
  Linux)
    INSTALLER_NAME="Miniconda3-latest-Linux-x86_64.sh"
    ;;
  Darwin)
    INSTALLER_NAME="Miniconda3-latest-MacOSX-x86_64.sh"
    ;;
  MINGW*|MSYS*|CYGWIN*)
    # Windows (MSYS / Git-bash / Cygwin)
    INSTALLER_NAME="Miniconda3-latest-Windows-x86_64.exe"
    INSTALLER_FLAGS=(/S /D="${MINICONDA_DIR}")
    ;;
  *)
    echo "Unsupported OS: $(uname -s)" >&2
    exit 1
    ;;
esac

INSTALLER_URL="${INSTALLER_URL_BASE}/${INSTALLER_NAME}"

if [ ! -d "$MINICONDA_DIR" ]; then
  echo "Installing Miniconda into $MINICONDA_DIR…"
  tmpfile=$(mktemp)
  if command -v wget >/dev/null; then
    wget -q "$INSTALLER_URL" -O "$tmpfile"
  elif command -v curl >/dev/null; then
    curl -sL "$INSTALLER_URL" -o "$tmpfile"
  else
    echo "Error: neither wget nor curl is installed." >&2
    exit 1
  fi

  if [[ "$INSTALLER_NAME" =~ \.sh$ ]]; then
    bash "$tmpfile" "${INSTALLER_FLAGS[@]}" -p "$MINICONDA_DIR"
  else
    # .exe installer on Windows
    chmod +x "$tmpfile"
    # run the Windows installer silently
    "$tmpfile" "${INSTALLER_FLAGS[@]}"
  fi

  rm -f "$tmpfile"
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

  # Install dlib from the root of the repo.
  # Use the source tar.gz on Linux/macOS, or Windows wheel if on Windows.
  OS_NAME="$(uname -s)"
  if [[ "$OS_NAME" == "Linux" || "$OS_NAME" == "Darwin" ]]; then
    echo " - Installing dlib 19.22.0 from source tarball…"
    "$CONDA_BIN" run -n preproc_env --no-capture-output \
      python -m pip install "./dlib-19.22.0.tar.gz"
  else
    echo " - Installing dlib 19.22.99 from Windows wheel…"
    "$CONDA_BIN" run -n preproc_env --no-capture-output \
      python -m pip install "./dlib-19.22.99-cp38-cp38-win_amd64.whl"
  fi

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
  streamlit run app/streamlit_app.py

# 6) Teardown: kill FastAPI when Streamlit exits
echo "Stopping FastAPI (pid=$UVICORN_PID)"
kill $UVICORN_PID
