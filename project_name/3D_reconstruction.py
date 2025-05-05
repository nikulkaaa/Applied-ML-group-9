import threestudio
import os
import subprocess
import cv2
from preprocessing import process_images

### We have not run this properly yet, this is a very basic script that still needs to be tuned when we can run it 

def setup_threestudio(config_path, input_folder, output_folder):
    """
    Set up the threestudio environment to create 3D reconstructions from the preprocessed faces.
    
    Args:
        config_path (str): Path to the threestudio config file (e.g., control4d-static.yaml).
        input_folder (str): Path to the folder containing preprocessed images.
        output_folder (str): Path to the folder where post-3D images will be saved.
    """
    # Make sure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Run 3D reconstruction for each image in the folder
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(f"Processing {filename}...")
            
            # Call the 3D reconstruction process using threestudio
            command = [
                "python", "threestudio/scripts/launch.py", 
                "--config", config_path,  # Path to the config file
                "--train",                 # Train the model
                "--gpu", "0",              # Use GPU 0 (modify if you have more GPUs)
                "data.dataroot=" + img_path  # Path to the individual image
            ]
            
            try:
                print(f"Running 3D reconstruction on {img_path}")
                subprocess.run(command, check=True)
                
                # After the reconstruction, save the 2D rendered image to output folder
                # Assuming threestudio will save the rendered output in a specific folder, adjust if necessary
                rendered_img_path = img_path.replace("data/faces", "data/post_3D_reconstruction")
                post_3d_image = cv2.imread(rendered_img_path)  # Read the rendered 3D image

                # Save the post-3D image back to the output folder
                cv2.imwrite(os.path.join(output_folder, filename), post_3d_image)
                print(f"Saved post-3D reconstruction: {filename}")
            
            except subprocess.CalledProcessError as e:
                print(f"Error during 3D reconstruction: {e}")

def run_3d_reconstruction():
    """
    Main function to process and reconstruct the faces.
    """
    # Paths to the necessary files and directories
    dataset_folder = 'data/preprocessed/'
    config_file_path = 'threestudio/configs/control4d-static.yaml'
    output_folder = 'data/post_3D_reconstruction/' 
    
    # Make sure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each of the subdirectories (train, test, val -> real/fake)
    for split in ['Train', 'Test', 'Validation']:
        for category in ['Real', 'Fake']:
            input_folder = os.path.join(dataset_folder, split, category)
            print(f"Starting 3D reconstruction for {split}/{category} images...")
            setup_threestudio(config_file_path, input_folder, output_folder)

if __name__ == "__main__":
    run_3d_reconstruction()
