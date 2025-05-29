#!/usr/bin/env bash
# Conda-only DECA installer + runner
# 0) MUST HAVE CONDA IN THE SYSTEM PATH OR LOCALLY IN CMD PROMPT!
# also make sure you have enough disk space to install all the packages + render images

# 0.5) STEPS TO ACTIVATE CONDA IN LOCAL PROMPT INSTEAD OF GLOBAL PATH:
  # CONDA_ROOT="/c/Users/hunte/Miniconda3"
    # replace with your own directory.
  # source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  # conda activate
  # conda --version
  # should return a verison, otherwise conda not activated: conda 25.3.1

# 1) go to root directory 
  # Usage: ./install_and_run_deca.sh [env_name] [input_path] [output_path]

# 2) this was successfully run in Git Bash on GPU AND CPU:
  # bash install_and_run_deca.sh deca_test project_name/data/preprocessed_dataset/preprocessed_eye_align project_name/temp_test

  # proposed to run in automated pipeline (also ran good, see the files):
  # bash install_and_run_deca.sh deca_test project_name/data/Uploaded_by_User project_name/data/3DRecon_by_User

# 3) check your output folder after running the script (or during!)

set -euo pipefail

ENV_NAME=${1:-deca}
INPUT_PATH=${2:-preprocessed_dataset/preprocessed_eye_align}
OUTPUT_PATH=${3:-results}
PYTHON_VER=3.8
TORCH_VER=2.1.0
CUDA_TAG=cu121
P3D_VER=0.7.8
# ──────────────────────────────────────────────────────────────────────────

# 0. Ensure Conda is available
if ! command -v conda &>/dev/null; then
  cat <<EOF >&2

❌  ERROR: 'conda' command not found.

This installer requires Conda. Please:

  1. Install Miniconda (https://docs.conda.io/en/latest/miniconda.html) or Anaconda.
  2. Restart your terminal so 'conda' is on your PATH. Check whether it is avaliable using "conda --version".
     If this does not return a verison, please ensure your PATH is set correctly.
  3. Re-run this script.

EOF
  exit 1
fi

# 1. Create & activate the environment
echo ">>> [1/4] Creating Conda env '$ENV_NAME' (Python $PYTHON_VER)…"
conda create -y -n "$ENV_NAME" python=$PYTHON_VER

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

PIP_CMD="conda run -n $ENV_NAME pip"
PY_CMD="conda run -n $ENV_NAME python"

# 2. Detect NVIDIA GPU
echo ">>> [2/4] Checking for NVIDIA GPU…"
if command -v nvidia-smi &>/dev/null; then
#  echo "    NVIDIA GPU found → installing CUDA wheels."
#  GPU=true
#else
  echo "    No NVIDIA GPU → installing CPU-only wheels."
  GPU=false
fi

# 3. Install PyTorch & PyTorch3D
echo ">>> [3/4] Installing PyTorch + PyTorch3D…"
if $GPU; then
  $PIP_CMD install \
    torch==${TORCH_VER}+${CUDA_TAG} \
    torchvision==0.16.0+${CUDA_TAG} \
    torchaudio==${TORCH_VER}+${CUDA_TAG} \
    --index-url https://download.pytorch.org/whl/${CUDA_TAG}

  $PIP_CMD install \
    pytorch3d==${P3D_VER}+pt${TORCH_VER}${CUDA_TAG} \
    --extra-index-url https://miropsota.github.io/torch_packages_builder
else
  $PIP_CMD install \
    torch==${TORCH_VER}+cpu \
    torchvision==0.16.0+cpu \
    torchaudio==${TORCH_VER}+cpu \
    --index-url https://download.pytorch.org/whl/cpu

  $PIP_CMD install \
    pytorch3d==${P3D_VER}+pt${TORCH_VER}cpu \
    --extra-index-url https://miropsota.github.io/torch_packages_builder
fi

# 4. Install DECA runtime deps
echo ">>> [4/4] Installing DECA runtime dependencies…"
$PIP_CMD install -q \
  numpy==1.23 scipy "scikit-image>=0.15" opencv-python \
  PyYAML==5.1.1 face-alignment==1.3.4 yacs==0.1.8 \
  ninja fvcore chumpy kornia tqdm


# Run DECA
echo
echo ">>> Running DECA reconstruction…"
if $GPU; then
  $PY_CMD DECA/demos/demo_reconstruct.py \
    -i "$INPUT_PATH" \
    --saveDepth True \
    --saveObj   True \
    --useTex    True \
    --rasterizer_type pytorch3d \
    -s    "$OUTPUT_PATH"
else
  export CUDA_VISIBLE_DEVICES=""
  $PY_CMD DECA/demos/demo_reconstruct.py \
    -i "$INPUT_PATH" \
    --saveDepth True \
    --saveObj   True \
    --useTex    True \
    --device    cpu \
    --rasterizer_type pytorch3d \
    -s    "$OUTPUT_PATH"
fi

echo
echo "All done! Results are in '$OUTPUT_PATH'."
