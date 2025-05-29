#!/usr/bin/env python
"""
Deepfake Recognition API

This module implements a FastAPI application that exposes an endpoint to:
  1. Upload a JPEG image
  2. Run a preprocessing pipeline (dlib face detection + transforms)
  3. Run a prediction model to decide whether the image is real or deepfake
  4. Return a structured JSON response with the label and confidence

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
        output (str | None): Optional textual output from preprocessing/prediction.
        error (str | None): Error message if status == "error".
        image_is_real (bool | None): True if the model predicts “real”, False for “deepfake”.
        confidence (float | None): Probability score for the predicted label.
    """
    status: str
    output: str | None = None
    error: str | None = None
    image_is_real: bool | None = None
    confidence: float | None = None


# Initialize FastAPI app with metadata
app = FastAPI(
    title="Deepfake Recognition API",
    description="Upload a JPEG and run preprocessing + prediction",
    version="1.0.0"
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

    Args:
        request (Request): The incoming HTTP request.
        exc (HTTPException): The exception raised.

    Returns:
        JSONResponse: A JSON body with error details and the appropriate status code.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "error": exc.detail}
    )


@app.post("/upload-image/", response_model=UploadImageResponse)
async def upload_image(file: UploadFile = File(...)) -> UploadImageResponse:
    """
    Uploads an image, runs preprocessing and prediction, and returns structured JSON.

    Workflow:
      1. Save the uploaded file to ./uploads
      2. Call the `preproc_env` Conda environment to run preproc_inference.py
      3. If no face is found, return 400 with a helpful message
      4. Call the `predict_env` Conda environment to run predict.py
      5. Parse its JSON stdout and return label + confidence

    Args:
        file (UploadFile): A JPEG image uploaded by the client.

    Raises:
        HTTPException(400): If no face is detected.
        HTTPException(500): If preprocessing or prediction fails unexpectedly.

    Returns:
        UploadImageResponse: Pydantic model with fields:
          - status: "success"
          - image_is_real: True/False
          - confidence: float
    """
    try:
        # 1) Save upload
        dest = UPLOAD_DIR / file.filename
        with open(dest, "wb") as fh:
            fh.write(await file.read())

        # 2) Preprocessing step
        pre = subprocess.run(
            [CONDA_BIN, "run", "-n", "preproc_env", "--no-capture-output",
             "python", "preproc_inference.py", str(dest)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if pre.returncode == 2:
            raise HTTPException(400, "No face detected in the image.")
        elif pre.returncode != 0:
            raise HTTPException(500, f"Pre-processing error:\n{pre.stderr.strip()}")

        # 3) Prediction step
        preproc_dir = dest.parent / f"{dest.stem}_preprocessed"
        pred = subprocess.run(
            [CONDA_BIN, "run", "-n", "predict_env", "--no-capture-output",
             "python", "predict.py", str(preproc_dir)],
            check=True, stdout=subprocess.PIPE, text=True
        )
        result = json.loads(pred.stdout.strip())

        # 4) Build and return response
        return UploadImageResponse(
            status="success",
            image_is_real=(result.get("label") == "real"),
            confidence=float(result.get("confidence", 0.0)),
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

    Args:
        request (Request): The incoming HTTP request.
        exc (Exception): The uncaught exception instance.

    Returns:
        JSONResponse: { "status": "error", "error": str(exc) }
    """
    # Print full traceback in server logs
    import traceback
    traceback.print_exc()

    return JSONResponse(
        status_code=500,
        content={"status": "error", "error": str(exc)},
    )
