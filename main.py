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
    real_src = os.path.join('project_name','data', 'Dataset', 'Train', 'Real')
    fake_src = os.path.join('project_name', 'data', 'Dataset', 'Train', 'Fake')
    real_dest = os.path.join('data', 'mini_dataset', 'real')
    fake_dest = os.path.join('data', 'mini_dataset', 'fake')

    # Check if the source directories exist
    if not os.path.exists(real_src):
        print(f"Source directory {real_src} does not exist.")
        return
    if not os.path.exists(fake_src):
        print(f"Source directory {fake_src} does not exist.")
        return

    # Move 3000 images
    move_images(real_src, real_dest, 3000)
    move_images(fake_src, fake_dest, 3000)

if __name__ == "__main__":
    main()
