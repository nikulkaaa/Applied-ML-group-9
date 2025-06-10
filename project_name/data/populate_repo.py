"""
File  to populate the repository.
Downloads and extracts:
- raw_dataset.zip
- preprocessed_no_background.zip
- 3DRecon.zip
- DECA.zip (into the root of the repository)
"""

import gdown
import zipfile
import os

def download_and_extract(file_id, output_name, extract_to="data/"):
    """Download a ZIP file from Google Drive and extract it."""
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {output_name}...")
    gdown.download(url, output_name, quiet=False)

    if output_name.endswith(".zip"):
        print(f"Extracting {output_name}...")
        with zipfile.ZipFile(output_name, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extraction of {output_name} complete ♥")

    os.remove(output_name)


# Dataset 1: raw_dataset.zip
download_and_extract(
    file_id="1byygZ1WF5D4RwoTo0RK3aVDgNtAw-d-q",
    output_name="raw_dataset.zip"
)

# Dataset 2: preprocessed_no_background.zip
download_and_extract(
    file_id="1iVR0HLoRrk31jLx4pXZ1nCovHO-lpRtS",
    output_name="preprocessed_no_background.zip"
)

# Dataset 3: 3DRecon.zip
download_and_extract(
    file_id="1WNPLwmjo75i-E8b0yS_XzYn5clg2Gmx_",
    output_name="3DRecon.zip"
)

# Dataset 4: DECA.zip (extracted into root)
download_and_extract(
    file_id="1FJy731c2AnZL33jezub7Wm7Um7YW0IKs",
    output_name="DECA.zip",
    extract_to="." 
)