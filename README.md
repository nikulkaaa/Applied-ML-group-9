# Applied Machine Learning Project (Group 9)🛠️

**Deepfake Detection Model** 

## Prerequisites
Before you begin, ensure you have the following:
    - A Bash-compatible shell on Windows (Git Bash, MSYS2, WSL) or a standard terminal on macOS/Linux.
    - One of wget or curl installed and available in your PATH.
    - Write permissions to install Miniconda into:
    - Windows: C:\Users\<your_user>\miniconda3
    - macOS/Linux: ~/miniconda3

### Installation
1. Clone the repository
    git clone https://github.com/nikulkaaa/Applied-ML-group-9.git

2. Make the installer script executable
    chmod +x run_pipeline.sh

3. Run the setup script
    ./run_pipeline.sh

This script will:
1. Detect your OS (Windows, macOS, or Linux) and download the matching Miniconda installer.
2. Install Miniconda into ~/miniconda3 (or C:\Users\<your_user>\miniconda3).
3. Create two Conda environments:
        preproc_env (Python 3.8) for data preprocessing, including dlib and other requirements.
        predict_env (Python 3.10) with CPU-only PyTorch, FastAPI, Streamlit, and prediction dependencies.

Once installation completes, the script will automatically start the FastAPI server in the background:
URL: http://localhost:8000
