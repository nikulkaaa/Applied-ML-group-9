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

2. Run the setup script
    ./run_pipeline.sh

This script will:
1. Detect your OS (Windows, macOS, or Linux) and download the matching Miniconda installer.
2. Install Miniconda into ~/miniconda3 (or C:\Users\<your_user>\miniconda3).
3. Create two Conda environments:
        preproc_env (Python 3.8) for data preprocessing, including dlib and other requirements.
        predict_env (Python 3.10) with CPU-only PyTorch, FastAPI, Streamlit, and prediction dependencies.

Once installation completes, the script will automatically start the FastAPI server in the background:
URL: http://localhost:8000

### Populating the repository 
If you'd like to train the models on your own or just explore the data used in this project, you can populate the repository by running the following command:

```bash
python project_name/data/populate_repo.py
```
The original full dataset can be found at https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images?resource=download

The DECA model was taken from: https://github.com/yfeng95/DECA \
Furthermore, it was edited for our specific use case. 