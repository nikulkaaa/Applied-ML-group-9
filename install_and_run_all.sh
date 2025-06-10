#!/usr/bin/env bash
set -euo pipefail

DECA_ENV=${1:-deca_env}

# 1. Miniconda bootstrap (trust existing folder first)
MINICONDA_DIR="${HOME}/miniconda3"
INSTALLER_URL_BASE="https://repo.anaconda.com/miniconda"

case "$(uname -s)" in
  Linux)                    INSTALLER_NAME="Miniconda3-latest-Linux-x86_64.sh" ;;
  Darwin)                   INSTALLER_NAME="Miniconda3-latest-MacOSX-x86_64.sh" ;;
  MINGW*|MSYS*|CYGWIN*)     INSTALLER_NAME="Miniconda3-latest-Windows-x86_64.exe" ;;
  *) echo "Unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac
INSTALLER_URL="${INSTALLER_URL_BASE}/${INSTALLER_NAME}"

if [ -d "${MINICONDA_DIR}" ]; then
  echo "Miniconda already present at ${MINICONDA_DIR} – skipping installer."
else
  if ! command -v conda &>/dev/null; then
    echo ">>> Installing Miniconda into ${MINICONDA_DIR}…"
    tmpfile=$(mktemp)
    if command -v wget &>/dev/null; then
      wget -q "${INSTALLER_URL}" -O "${tmpfile}"
    else
      curl -sL "${INSTALLER_URL}" -o "${tmpfile}"
    fi
    if [[ "${INSTALLER_NAME}" == *.sh ]]; then
      bash "${tmpfile}" -b -p "${MINICONDA_DIR}"
    else
      chmod +x "${tmpfile}"
      "${tmpfile}" /S /D="${MINICONDA_DIR}"
    fi
    rm -f "${tmpfile}"
  else
    echo "conda already on PATH – installer skipped."
  fi
fi

# 2. Locate conda
if command -v conda &>/dev/null; then
  CONDA_BIN="$(command -v conda)"
elif [ -x "${MINICONDA_DIR}/bin/conda" ]; then
  CONDA_BIN="${MINICONDA_DIR}/bin/conda"
elif [ -x "${MINICONDA_DIR}/condabin/conda" ]; then
  CONDA_BIN="${MINICONDA_DIR}/condabin/conda"
elif [ -x "${MINICONDA_DIR}/Scripts/conda.exe" ]; then
  CONDA_BIN="${MINICONDA_DIR}/Scripts/conda.exe"
else
  echo "❌  Conda executable not found." >&2
  exit 1
fi
echo ">>> Using conda at: ${CONDA_BIN}"

conda_env_exists() { "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq "$1"; }
conda_run()       { local env="$1"; shift; "${CONDA_BIN}" run -n "${env}" --no-capture-output "$@"; }

# 3. Environment: DECA
if ! conda_env_exists "${DECA_ENV}"; then
  echo ">>> [1/3] Creating ${DECA_ENV} (Python 3.8)…"
  "${CONDA_BIN}" create -y -n "${DECA_ENV}" python=3.8 pip

  TORCH_VER=2.1.0
  P3D_VER=0.7.8
  echo ">>> Installing PyTorch ${TORCH_VER} & PyTorch3D ${P3D_VER} (CPU-only)…"
  conda_run "${DECA_ENV}" pip install \
    torch==${TORCH_VER}+cpu torchvision==0.16.0+cpu torchaudio==${TORCH_VER}+cpu \
    --index-url https://download.pytorch.org/whl/cpu
  conda_run "${DECA_ENV}" pip install \
    pytorch3d==${P3D_VER}+pt${TORCH_VER}cpu \
    --extra-index-url https://miropsota.github.io/torch_packages_builder

  echo ">>> Installing DECA runtime dependencies…"
  conda_run "${DECA_ENV}" pip install -q \
    numpy==1.23 scipy "scikit-image>=0.15" opencv-python \
    PyYAML==5.1.1 face-alignment==1.3.4 yacs==0.1.8 \
    ninja fvcore chumpy kornia tqdm
else
  echo ">>> deca_env already exists - skipping creation."
fi

# 4. Create environment: preproc_env
if ! conda_env_exists preproc_env; then
  echo ">>> [2/3] Creating preproc_env (Python 3.8)…"
  "${CONDA_BIN}" create -y -n preproc_env python=3.8 pip

  echo ">>> Installing dlib…"
  conda_run preproc_env conda install -c conda-forge dlib=19.22.0 -y

  echo ">>> Installing preprocessing requirements…"
  conda_run preproc_env pip install -r requirements_preproc.txt
else
  echo ">>> preproc_env already exists - skipping creation."
fi

# 5. Create environment: predict_env
if ! conda_env_exists predict_env; then
  echo ">>> [3/3] Creating predict_env (Python 3.10)…"
  "${CONDA_BIN}" create -y -n predict_env python=3.10 pip

  echo ">>> Installing CPU-only PyTorch 2.7.0…"
    conda_run predict_env pip install torch==2.7.0 \
      -f https://download.pytorch.org/whl/cpu/torch_stable.html

  echo ">>> Installing prediction stack & web back-ends…"
  conda_run predict_env pip install -r requirements_predict.txt
  conda_run predict_env pip install fastapi uvicorn streamlit python-multipart requests pillow
else
  echo ">>> predict_env already exists - skipping creation."
fi


# 6. Launch FastAPI + Streamlit
echo ">>> Starting FastAPI on http://localhost:8000"
conda_run predict_env python -m uvicorn app:app --reload --port 8000 &
UVICORN_PID=$!

sleep 2

echo ">>> Starting Streamlit on http://localhost:8501 (Ctrl-C to quit)…"
conda_run predict_env python -m streamlit run app/streamlit_app.py

echo ">>> Shutting down FastAPI (pid=${UVICORN_PID})"
kill "${UVICORN_PID}"
