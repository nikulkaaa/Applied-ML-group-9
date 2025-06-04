#!/usr/bin/env python
"""
Deepfake Recognition API

This module implements a FastAPI application that exposes an endpoint to:
  1. Upload a JPEG image
  2. Run a preprocessing pipeline (dlib face detection + transforms)
  3. Run a prediction model to decide whether the image is real or deepfake
     and produce a Grad-CAM saliency image
  4. Return a structured JSON response with the label, confidence, and saliency path

Environment and dependencies are managed via Conda environments:
  - `preproc_env` for preprocessing (Python 3.8)
  - `predict_env` for prediction (Python 3.10)
"""

import os
import subprocess
import json
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

# Standard logger for Uvicorn
logger = logging.getLogger("uvicorn.error")

# Path to Conda binary (set via env or default “conda”)
CONDA_BIN = os.getenv("CONDA_BIN", "conda")

# Directory where uploads are stored
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class UploadImageResponse(BaseModel):
    """
    JSON schema for the response to an image upload.

    Attributes:
        status (str): "success" or "error".
        error (str | None): Error message if status == "error".
        image_is_real (bool | None): True if the model predicts “real”, False for “deepfake”.
        confidence (float | None): Probability score for the predicted label.
        saliency (str | None): Filepath to the saved Grad-CAM overlay image.
    """
    status: str
    error: str | None = None
    image_is_real: bool | None = None
    confidence: float | None = None
    saliency: str | None = None


# Initialize FastAPI app with metadata
app = FastAPI(
    title="Deepfake Recognition API",
    description="Upload a JPEG and run preprocessing + prediction (+ Grad-CAM)",
    version="1.0.0",
)
app.mount(
    "/uploads",
    StaticFiles(directory="uploads", html=False),
    name="uploads",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle HTTPException globally.

    Converts any raised HTTPException into a JSONResponse with:
      {
        "status": "error",
        "error": exc.detail
      }
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "error": exc.detail}
    )


@app.post("/upload-image/", response_model=UploadImageResponse)
async def upload_image(file: UploadFile = File(...)) -> UploadImageResponse:
    """
    1) Save the uploaded file to ./uploads
    2) Call `preproc_env` to run preproc_inference.py
    3) If no face is found, return 400
    4) Call `predict_env` to run predict.py (without check=True)
    5) Inspect pred.returncode:
         - If returncode != 0, try to load JSON from stdout:
             • If stdout JSON contains {"error":...}, return HTTP 400
             • Otherwise return HTTP 500 with stderr
         - If returncode == 0, parse JSON from stdout and return success
    """
    try:
        # 1) Save upload
        dest = UPLOAD_DIR / file.filename
        with open(dest, "wb") as fh:
            fh.write(await file.read())

        # 2) Preprocessing step
        pre = subprocess.run(
            [CONDA_BIN, "run", "-n", "preproc_env", "--no-capture-output",
             "python", "app/preproc_inference.py", str(dest)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if pre.returncode == 2:
            # our preproc_inference.py uses exit code 2 to signal “no face detected”
            raise HTTPException(400, "No face detected in the image.")
        elif pre.returncode != 0:
            raise HTTPException(500, f"Pre-processing error:\n{pre.stderr.strip()}")

        # 3) Prediction step (note: no check=True)
        preproc_dir = dest.parent / f"{dest.stem}_preprocessed"
        pred = subprocess.run(
            [CONDA_BIN, "run", "-n", "predict_env", "--no-capture-output",
             "python", "app/predict.py", str(preproc_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 4) If predict.py returned non-zero, examine its output
        if pred.returncode != 0:
            # First, try to parse JSON from stdout
            try:
                j = json.loads(pred.stdout.strip())
            except json.JSONDecodeError:
                # No valid JSON in stdout → we treat stderr as a 500
                raise HTTPException(500, f"Prediction failed:\n{pred.stderr.strip()}")
            else:
                # If the JSON has an "error" field, return 400 with that
                if "error" in j:
                    raise HTTPException(400, j["error"])
                # Otherwise, something unexpected happened—treat as 500:
                raise HTTPException(500, f"Unexpected prediction output: {pred.stdout.strip()}")

        # 5) At this point returncode == 0, so parse JSON from stdout
        try:
            result = json.loads(pred.stdout.strip())
        except json.JSONDecodeError:
            raise HTTPException(500, f"Prediction succeeded but returned invalid JSON.")

        # 6) If predict.py returned {"error": "..."} even with returncode==0
        if "error" in result:
            raise HTTPException(400, result["error"])

        # 7) Otherwise extract label, confidence, saliency
        label = result.get("label")
        confidence = result.get("confidence")
        saliency = result.get("saliency")

        return UploadImageResponse(
            status="success",
            image_is_real=(label == "real"),
            confidence=confidence,
            saliency=saliency
        )

    except HTTPException:
        # propagate known HTTPExceptions to be handled above
        raise
    except Exception as e:
        # Log full traceback for debugging, then return HTTP 500
        logger.exception("Unhandled error during pipeline")
        raise HTTPException(500, f"Unexpected error: {e}")



@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all exception handler for unanticipated errors.

    Prints traceback to the console and returns a 500 JSONResponse.
    """
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"status": "error", "error": str(exc)},
    )
