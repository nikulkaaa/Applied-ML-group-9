import os
import shutil
import random

def move_images(src_folder: str, dest_folder: str, num_images: int) -> None:
    """Move a specified number of images from one folder to another."""
    # Make sure that destination folder exists
    os.makedirs(dest_folder, exist_ok=True)

    # List all files in the source folder
    all_images = [f for f in os.listdir(src_folder) if f.endswith('.jpg')]

    # Randomly select num_images images from the list
    selected_images = random.sample(all_images, num_images)

    # Move the selected images
    for image in selected_images:
        src_path = os.path.join(src_folder, image)
        dest_path = os.path.join(dest_folder, image)

        # Move the file
        shutil.move(src_path, dest_path)

    print(f"Moved {num_images} images from {src_folder} to {dest_folder}.")

def main():
    # Use one of the following blocks depending on which type of data you want to create a subset of

    # TRAINING DATA:
    # real_src = os.path.join('project_name','data', 'Dataset', 'Train', 'Real')
    # fake_src = os.path.join('project_name', 'data', 'Dataset', 'Train', 'Fake')
    # real_dest = os.path.join('project_name', 'data', 'mini_dataset', 'Train', 'real')
    # fake_dest = os.path.join('project_name', 'data', 'mini_dataset', 'Train', 'fake')

    # VALIDATION DATA:
    real_src = os.path.join('project_name','data', 'Dataset', 'Validation', 'Real')
    fake_src = os.path.join('project_name','data', 'Dataset', 'Validation', 'Fake')
    real_dest = os.path.join('project_name', 'data', 'mini_dataset', 'Validation', 'real')
    fake_dest = os.path.join('project_name', 'data', 'mini_dataset', 'Validation', 'fake')

    # TEST DATA:
    # real_src = os.path.join('project_name','data', 'Dataset', 'Test', 'Real')
    # fake_src = os.path.join('project_name','data', 'Dataset', 'Test', 'Fake')
    # real_dest = os.path.join('project_name', 'data', 'mini_dataset', 'Test', 'real')
    # fake_dest = os.path.join('project_name', 'data', 'mini_dataset', 'Test', 'fake')

    # Check if the source directories exist
    if not os.path.exists(real_src):
        print(f"Source directory {real_src} does not exist.")
        return
    if not os.path.exists(fake_src):
        print(f"Source directory {fake_src} does not exist.")
        return
    
    # The number of real/fake images you want in your data subset
    NUMBER_OF_IMAGES = 250

    move_images(real_src, real_dest, NUMBER_OF_IMAGES)
    move_images(fake_src, fake_dest, NUMBER_OF_IMAGES)

if __name__ == "__main__":
    main()
