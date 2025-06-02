import gdown
import zipfile
import os

# Google Drive file ID for raw_dataset.zip
file_id = "1byygZ1WF5D4RwoTo0RK3aVDgNtAw-d-q"
output = "raw_dataset.zip"

# Construct the direct download URL
url = f"https://drive.google.com/uc?id={file_id}"

# Download the ZIP file
print("Downloading raw_dataset.zip...")
gdown.download(url, output, quiet=False)

# Extract the ZIP file
if output.endswith(".zip"):
    print("Extracting contents...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
         # Extract data into the data folder
        zip_ref.extractall("data/")
    print("Extraction complete ♥")

# Remove the zip file after extraction
os.remove(output)
