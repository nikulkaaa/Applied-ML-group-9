import os
import sys
import subprocess
import json
import re
import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

# log for any errors to trakc
logger = logging.getLogger("uvicorn.error")

CONDA_BIN = os.getenv("CONDA_BIN", "conda")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

class UploadImageResponse(BaseModel):
    status: str
    output: str | None = None
    error: str | None = None
    image_is_real: bool | None = None
    confidence: float | None = None

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
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code,
                        content={"status": "error", "error": exc.detail})

@app.post("/upload-image/", response_model=UploadImageResponse)
async def upload_image(file: UploadFile = File(...)):
    try:

        dest = UPLOAD_DIR / file.filename
        with open(dest, "wb") as fh:
            fh.write(await file.read())

        # run the pre processing
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
            raise HTTPException(
                500, f"Pre-processing error:\n{pre.stderr.strip()}"
            )

        # run the prediction
        preproc_dir = dest.parent / f"{dest.stem}_preprocessed"
        pred = subprocess.run(
            [CONDA_BIN, "run", "-n", "predict_env", "--no-capture-output",
             "python", "predict.py", str(preproc_dir)],
            check=True, stdout=subprocess.PIPE, text=True
        )

        result = json.loads(pred.stdout.strip())

        return UploadImageResponse(
            status="success",
            image_is_real=(result.get("label") == "real"),
            confidence=float(result.get("confidence", 0.0)),
        )

    except HTTPException:
        raise
    except Exception as e:
        # log full traceback to console
        logger.exception("Unhandled error during pipeline")
        # return a 500 with the exception message
        raise HTTPException(500, f"Unexpected error: {e}")

import traceback
from fastapi import Request

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"status":"error","error":str(exc)},
    )
