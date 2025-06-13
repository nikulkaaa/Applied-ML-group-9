# Applied Machine Learning Project (Group 9)🛠️

**Deepfake Detection Model** 

## 1. Prerequisites
Before you begin, ensure you have the following:
  - A Bash-compatible shell on Windows (Git Bash, MSYS2, WSL) or a standard terminal on macOS/Linux.
  - One of wget or curl installed and available in your PATH.
  - [Miniconda3](https://docs.conda.io/en/latest/miniconda.html)
  - Make sure to repopulate the repository (this is described in detail below)
  - Follow installation guides (tested extensively on Windows 11)

## 2. Installation with Docker (multi-platform and most recommended)

2.1 Open a Git Bash terminal and clone the repository
   ```bash     
   git clone https://github.com/nikulkaaa/Applied-ML-group-9.git
   ```
2.2 Download and install Docker Desktop, make sure to install the one compatible with your system:  
   https://www.docker.com/products/docker-desktop/

2.3 Open Docker Desktop and make sure that it is running in the background before following any of the next instructions.

2.4 Navigate to the root directory that was just cloned
   ```bash   
   cd Applied-ML-group-9
   ```
   
2.5 Make a temporary virtual environment using requirements.txt, this will be used for populating the repository with the necessary files and folders to run our model. 
   Make the environment named .populate_env that will contain the requirements
   ```bash
   python -m venv .populate_env
   ```
   
   Activate the environment
   ```bash
   # For Linux/macOS:
   source .populate_env/bin/activate

   # For Windows (using Git Bash):
   source .populate_env/Scripts/activate
   ```
   
   Install the requirements in your newly activated environment
   ```bash
   pip install -r requirements.txt
   ```

2.6 Populate the repository with the DECA folder as it is crucial for 3D reconstruction (use the environment you just made)
   ```bash
   python project_name/data/populate_repo.py
   ```

   Thereafter, deactivate the environment like so:
   ```bash   
   deactivate
   ```
   Once completed, you may now remove the temporary virtual environment made in step four for population (optional)
   ```bash   
   rm -rf .populate_env
   ```
2.7 After populating the folders successfully, navigate to the docker folder (make sure you are still in the root or the repository):  
   ```bash   
   cd docker
   ```
2.8 Build the Docker images and start the services:  
   ```bash   
   docker compose up --build
   ```
   Wait for the build to complete and containers to start. Once running, you can access:  
   - FastAPI backend docs at: http://localhost:8000/docs  
   - Streamlit UI at: http://localhost:8501

2.9 To stop the containers, press Ctrl+C or, alternatively, run:  
   ```bash      
   docker compose down
   ```

2.10 If you want to start up the web services again without rebuilding, please run:  
   ```bash      
   docker compose up
   ```

2.11 (Optional) View logs for the services:  
   ```bash      
   docker compose logs -f api  
   docker compose logs -f ui
   ```

This script will:

- Clone the project repository containing all source code, configurations, and dependencies.  
- Guide you in installing Docker Desktop, the platform that enables containerization on your operating system.  
- Ensure Docker Desktop is running in the background before proceeding with further commands.  
- Navigate into the 'docker' folder within the project, where the Dockerfile and Docker Compose configurations are located.  
- Build Docker images that create isolated environments, including:  
  - Three Separate conda environments for DECA 3D Reconstruction ('deca_env'), preprocessing ('preproc_env') and prediction ('predict_env') with their respective dependencies, including dlib, FastAPI, Streamlit, and other utilities
  - Ensures that if you build the image, it will let you package your environment and share it so that “it just works” everywhere that Docker is supported (like other host machines)
- Launch two services inside Docker containers:  
  - A FastAPI backend server accessible at 'http://localhost:8000/docs' for API interactions and documentation  
  - A Streamlit frontend UI accessible at 'http://localhost:8501' for interactive visualization and user interaction  
- Enable you to stop the running containers either by pressing 'Ctrl+C' in the terminal or running 'docker compose down'.  
- Allow restarting the services without rebuilding the images to save time by running 'docker compose up'.  

## 3. Installation with Shell Script (only tested on Windows; may be unreliable)

3.1 Open a Git Bash terminal and clone the repository: 
   ```bash    
   git clone https://github.com/nikulkaaa/Applied-ML-group-9.git
   ```
3.2 Download and install Miniconda if it is not already installed.  
   https://docs.conda.io/en/latest/miniconda.html

3.3 Navigate into the project root folder:
   ```bash      
   cd Applied-ML-group-9
   ```

3.4 Make a temporary virtual environment using requirements.txt, this will be used for populating the repository with the necessary files and folders to run our model. 
   Make the environment named .populate_env that will contain the requirements
   ```bash
   python -m venv .populate_env
   ```
   
   Activate the environment
   ```bash
   # For Linux/macOS:
   source .populate_env/bin/activate

   # For Windows (using Git Bash or similar):
   source .populate_env/Scripts/activate
   ```
   
   Install the requirements in your newly activated environment
   ```bash
   pip install -r requirements.txt
   ```

3.5 Populate the repository with the DECA folder, as it is crucial for 3D reconstruction (use the environment you just made)
   ```bash
   python project_name/data/populate_repo.py
   ```

   Thereafter, deactivate the environment like so:
   ```bash   
   deactivate
   ```
   Once completed, you may now remove the temporary virtual environment made in step four for population (optional)
   ```bash   
   rm -rf .populate_env
   ```
3.6 Make the following script executable:
   ```bash    
   chmod +x ./install_and_run_all.sh
   ```
3.7 Run the provided shell script to set up environments and launch the services:
   ```bash  
   ./install_and_run_all.sh
   ```
   To stop the services, press Ctrl+C in the terminal.

The script will:  
   - Install Miniconda if missing  
   - Create three conda environments: deca_env, preproc_env, and predict_env with Python 3.8 and 3.10 as needed  
   - Install all necessary Python packages (PyTorch CPU versions, dlib, FastAPI, Streamlit, and others) in each environment  
   - Start the FastAPI server accessible at http://localhost:8000/docs 
   - Start the Streamlit UI accessible at http://localhost:8501  
   - Manage the startup and shutdown of these services for the user

## 4. Overview of Training
The full model achieves higher performance than the baseline model. The full model makes use of 3D reconstruction of all the sample data and uses it to compute error maps in comparison to the 2D image to make accurate predictions. The baseline model is only trained on the 2D images, and is weaker in detecting depth and angle discrepancies in the faces (which 3D reconstruction is strong for). If you would like to retrain the models, please make sure to have all the data by completing the populate step (make sure to populate the environment before retraining, see steps 3.4-3.5 or 2.5-2.6). Arguments to run or tune the models can be found in detail in the code. For the full model, we have tuned hyperparameters available to use; these were obtained with Optuna and provided the best results for us. 

### 4.1 Baseline
- The baseline model can be retrained by running:
    ```bash
    python project_name/models/baseline_cnn.py
    ```
- This includes running k-fold cross-validation with k=5 to monitor the performance across different folds to check for overfitting
- The average performance over all five folds when we ran testing resulted in the following metrics:
    - Mean Accuracy: 0.894
    - Mean F1 Score: 0.894
    - Mean ROC-AUC: 0.971
    - Mean Equal Error Rate: 0.0873

### 4.2 Full Model
- Can make use of the predict_env to run this model, requirements found in 'requirements_predict.txt' (ignore name-mismatch, it still works)
- The full model can be retrained by running:
    ```bash
    python project_name/models/two_stream_model.py --preproc-root data/preprocessed_no_background --recon-root data/3DRecon
    ```
- These arguments can be changed to any directory if you would like to retrain your model using your own data. It will load the best hyperparameters we found through hyperparameter tuning automatically. These can be overridden by parsing the hyperparameter arguments of your choosing.
- This includes running k-fold cross-validation with k=5 in order to monitor the performance across different folds to check for overfitting and if our model was 'lucky' at any point
- The average performance over all five folds when we ran testing resulted in the following metrics:
    - Mean Accuracy: 0.948
    - Mean F1 Score: 0.949
    - Mean ROC-AUC: 0.988
    - Mean Equal Error Rate: 0.0483
    

## 5. Justifications and Design Choices
- Even though DECA was GPU compatible we chose to only allow running it on the CPU. This was because of lots of dependency and compatibility issues that we ran into,
  especially when running on different hardware that utilized different GPUs. Therefore, the CPU-only route was chosen to focus on user-friendly interaction and
  setup with our deepfake detection model. As a note, all of the 3D reconstructed data used for training was performed on a GPU; however, since we only reconstruct
  one image per upload by the user for detection, running on a CPU provides an adequate amount of waiting time. Furthermore, we justified this by providing the baseline
  model to the user, so that if they want a quick but less reliable prediction, they can use the fast-performing model to get results fast.  

- We chose to implement Grad-CAM saliency maps to add an explainable layer to the project.
  We wanted to make sure that neither one of the models (especially the full model) used noise in the images for its decision.
  With the saliency maps, we detected that the full model was only using the background of the images to classify them, and so we chose to remove the background from all images in preprocessing such that the models could not rely on the background any longer, which both removed that problem and improved the performance of our model, showing how useful the saliency maps were to designing our model.

- Detailed Expression Capture and Animation (DECA) was the best publicly available 3D reconstruction model that outperformed many other models. We chose it as we found that it would provide us with the best results in order to compute the most precise error maps to detect deepfakes (as they can have minor discrepancies). Furthermore, it had an easy setup process that was user-friendly and could be used on a majority of machines. Also, it managed to make a 3D reconstruction from one image, as we found that many other models had to use multiple angles and frames to do this. We did not want to use another model that needed many images, as we wanted our model to be novel in the sense that it could do a one-shot deepfake detection if a user uploaded their image. Therefore, we found this to be the most useful in order to use and make our app, which we developed, unique and easier to use. 


## 6. Acknowledgements
### 6.1 The Detailed Expression Capture and Animation (DECA) model
The [DECA](https://github.com/yfeng95/DECA) model was used as our primary way of reconstructing our 2D images into 3D in a single shot. All credits go to the creators Feng, Yao and Feng, Haiwen and Black, Michael J., and Bolkart, Timo. We did edit the code slightly to ensure a smoother process when running it with our specific design for deepfake detection. This included making it CPU-only for maximum user-friendly and saving only explicitly what we needed.

### 6.2 The dataset
The original full dataset can be found at https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images?resource=download
The dataset is not our own and is publicly available at the link above. It contains around 190,000 samples, of which we are using roughly 16,000. The dataset comes split into real and fake images (we used 8,000 real images vs. 8,000 deepfaked images). The dataset is already pre-split into Train/Val/Test, but in order to do k-fold cross-validation, we sourced all of our samples from the Training set (and manually checked them to clean up any images that were not useful, e.g., no faces).

### 6.3 Basel Face Model
[Basel Face Model](https://faces.dmi.unibas.ch/bfm/bfm2017.html) (BFM) was used to take the meshes in BFM and convert them into a FLAME mesh. All credit for the provided models goes to the authors Thomas Gerig, Andreas Morel-Forster, Clemens Blumer, Bernhard Egger, Marcel Lüthi, Sandro Schönborn, and Thomas Vetter on the Morphable Face Models - An Open Framework.

### 6.4 FLAME
[Flame](https://flame.is.tue.mpg.de/) was used for the DECA model. We used various models available from the site in order to carry out the 3D reconstruction as required by the DECA installation. This included FLAME2020 models and their [Basel Face Model (BFM) to Flame model converter](https://github.com/TimoBolkart/BFM_to_FLAME) for the albedo option in DECA.

### 6.5 RetinaFace & Dlib
[RetinaFace](https://github.com/serengil/retinaface) was used for preprocessing to detect the faces, crop to the correct region in the image, and use Dlib 68-landmark prediction to remove all the background from the faces. Both libraries are open-source (Apache 2.0 for RetinaFace, Boost 1.0 for dlib), and their pre-trained weights are downloaded automatically by our preproc_env.
