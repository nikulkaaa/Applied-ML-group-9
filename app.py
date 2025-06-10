"""
Deepfake Recognition API: baseline + FULL pipeline
 + /upload-image/ -> preprocess : baseline-predict
 + /upload-image-full/  -> preprocess : DECA's demo_reconstruct.py : full-predict

Environments used
  - preproc_env   (Python 3.8)  - preprocessing
  - deca_env  (Python 3.8)  - DECA 3D reconstruction
  - predict_env   (Python 3.10) - baseline + full models
"""

import os
import uuid
import shutil
import subprocess
import json
import logging
from pathlib import Path
import time 

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

CONDA_BIN = (
    os.getenv("CONDA_BIN")
    or os.getenv("CONDA_EXE")
    or shutil.which("conda")
)
if not CONDA_BIN or not Path(CONDA_BIN).exists():
    raise RuntimeError(
        "Conda executable not found. Make sure Miniconda/Anaconda is "
        "installed and that either CONDA_BIN or CONDA_EXE is set."
    )

DECA_ENV = os.getenv("DECA_ENV", "deca_env") # the deca env we created w/ shell script

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class UploadImageResponse(BaseModel):
    status: str
    error: str | None = None
    image_is_real: bool | None = None
    confidence: float | None = None
    saliency: str | None = None


app = FastAPI(
    title="Deepfake Recognition API",
    description="Upload a JPEG and run preprocessing + prediction (+ Grad-CAM)",
    version="1.2.1",
)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

def run_subprocess_and_stream(cmd: list[str], *, error_ctx: str, face_error_ok: bool = False, status_prefix: str = ""):
    """
    Runs a subprocess and yields its output line by line.
    Handles errors and special "no face" case.
    """
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1 
        )

        stdout_lines = []
        stderr_lines = []

        for line in proc.stdout:
            stdout_lines.append(line.strip())

            logger.debug(f"Subprocess stdout: {line.strip()}")

        for line in proc.stderr:
            stderr_lines.append(line.strip())
            logger.error(f"Subprocess stderr: {line.strip()}")
            yield f"ERROR: {status_prefix} stderr: {line.strip()}\n"

        proc.wait()

        if proc.returncode == 0:
            return "\n".join(stdout_lines), "\n".join(stderr_lines)

        # Special-case: if demo_preproc signals “no face” via exit code 2
        if face_error_ok and proc.returncode == 2:
            raise HTTPException(400, "No face detected in the image. Please make sure your image is of a human face.")

        try:
            j = json.loads("".join(stdout_lines).strip() or "{}")
            if "error" in j:
                raise HTTPException(400, j["error"])
        except json.JSONDecodeError:
            pass

        raise HTTPException(500, f"{error_ctx}:\n" + '\n'.join(stderr_lines).strip())

    except FileNotFoundError as e:
        raise HTTPException(
            500,
            f"Executable not found: {cmd[0]}\nDetail: {e}"
        )


# /upload-image/ handles baseline model
@app.post("/upload-image/", response_model=UploadImageResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Baseline pipeline: preprocess → baseline model
    Returns StreamingResponse with status updates.
    """
    # Generate a unique identifier for this request
    # This ensures that each upload gets its own isolated processing space
    request_id = str(uuid.uuid4())
    
    # Create a unique directory for this specific upload and its processing
    # This will be uploads/[filename]_uuid/
    unique_upload_dir = UPLOAD_DIR / f"{file.filename.split('.')[0]}_{request_id}"
    # Create the main unique directory
    unique_upload_dir.mkdir(exist_ok=True)
    
    dest = unique_upload_dir / file.filename
    contents = await file.read()
    with open(dest, "wb") as fh:
        fh.write(contents)

    preproc_dir = unique_upload_dir / f"{dest.stem}_preprocessed"
    preproc_dir.mkdir(exist_ok=True)

    def generate_baseline_stream():
        def step(msg: str):
            yield f"STATUS: {msg}\n"
            time.sleep(0.1)

        try:
            # Preprocessing step
            yield from step("PREPROC_START")
            pre = subprocess.run(
                [
                    CONDA_BIN, "run", "-n", "preproc_env", "--no-capture-output",
                    "python", "app/preproc_inference.py", str(dest)
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if pre.returncode == 2:
                yield from step("PREPROC_NO_FACE")
                yield "ERROR: No face detected in the image.\n"
                return
            elif pre.returncode != 0:
                yield from step("PREPROC_ERROR")
                yield f"ERROR: Pre-processing failed:\n{pre.stderr}"
                return
            yield from step("PREPROC_DONE")

            # Baseline prediction
            yield from step("MODEL_START")
            pred = subprocess.run(
                [
                    CONDA_BIN, "run", "-n", "predict_env", "--no-capture-output",
                    "python", "app/predict_baseline.py", str(preproc_dir)
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if pred.returncode != 0:
                yield from step("MODEL_ERROR")
                stderr_text = pred.stderr.rstrip("\n")
                for ln in stderr_text.split("\n"):
                    yield f"ERROR: {ln}\n"
                return
            yield from step("MODEL_DONE")

            # Final JSON
            result_json_str = pred.stdout.strip()
            result = json.loads(result_json_str)

            # Adjust saliency path to be relative to UPLOADS_DIR
            if "saliency" in result and result["saliency"]:
                abs_saliency_path = Path(result["saliency"])
                relative_saliency_path = abs_saliency_path.relative_to(UPLOAD_DIR)
                result["saliency"] = str(relative_saliency_path).replace("\\", "/")
            
            yield json.dumps(result) + "\n"

        except HTTPException as e:
            yield f"ERROR: HTTP Exception - {e.status_code}: {e.detail}\n"
        except Exception as e:
            logger.exception("Unhandled error in /upload-image/ generate_baseline_stream")
            yield f"ERROR: Unexpected server error: {e}\n"

    return StreamingResponse(generate_baseline_stream(), media_type="text/plain")


# /upload-image-full/ handles full model
@app.post("/upload-image-full/")
async def upload_image_full(file: UploadFile = File(...)):
    """
    Streaming version: yields status lines as we go.
    At the very end, yields a final JSON line (with label/confidence/saliency).
    """
    # Generate a unique identifier for this request
    request_id = str(uuid.uuid4())

    # Create a unique directory for this specific upload and its processing
    unique_upload_dir = UPLOAD_DIR / f"{file.filename.split('.')[0]}_{request_id}"
    # Create the main unique directory
    unique_upload_dir.mkdir(exist_ok=True)

    # Save the file inside this unique dir
    dest = unique_upload_dir / file.filename
    contents = await file.read()
    with open(dest, "wb") as fh:
        fh.write(contents)

    preproc_dir = unique_upload_dir / f"{dest.stem}_preprocessed"
    preproc_dir.mkdir(exist_ok=True)

    deca_output_dir = preproc_dir / "deca_output"
    deca_output_dir.mkdir(exist_ok=True)

    def generate():
        def step(msg: str):
            yield f"STATUS: {msg}\n"
            import time; time.sleep(0.1)

        # Preprocessing
        yield from step("PREPROC_START")
        pre = subprocess.run(
            [
                CONDA_BIN, "run", "-n", "preproc_env", "--no-capture-output",
                "python", "app/preproc_inference.py", str(dest)
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if pre.returncode == 2:
            yield from step("PREPROC_NO_FACE")
            yield "ERROR: No face detected in the image, make sure to upload an image of a (clear) face.\n "
            return
        elif pre.returncode != 0:
            yield from step("PREPROC_ERROR")
            yield f"ERROR: Pre-processing failed:\n{pre.stderr}"
            return
        yield from step("PREPROC_DONE")

        # 3D Reconstruction
        yield from step("3D_START")
        proc3d = subprocess.run(
            [
                CONDA_BIN, "run", "-n", "deca_env", "--no-capture-output",
                "python", "DECA/demos/demo_reconstruct.py",
                "-i", str(preproc_dir),
                "--no_recursive",
                "--saveDepth", "True",
                "--useTex", "True",
                "--rasterizer_type", "pytorch3d",
                "--device", "cpu",
                "-s", str(deca_output_dir)
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if proc3d.returncode != 0:
            yield from step("3D_ERROR")
            yield f"ERROR: 3-D reconstruction failed:\n{proc3d.stderr}"
            return
        yield from step("3D_DONE")

        # Full-model Prediction
        yield from step("MODEL_START")
        pred = subprocess.run(
            [
                CONDA_BIN, "run", "-n", "predict_env", "--no-capture-output",
                "python", "app/predict_full.py",
                str(preproc_dir),
                str(deca_output_dir)
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if pred.returncode != 0:
            yield from step("MODEL_ERROR")
            stderr_text = pred.stderr.rstrip("\n")
            for ln in stderr_text.split("\n"):
                yield f"ERROR: {ln}\n"
            return
        yield from step("MODEL_DONE")

        # Final JSON
        result_json_str = pred.stdout.strip()
        result = json.loads(result_json_str)

        for key in ["saliency", "rendered_3d_image", "depth_map_image", "normals_map_image"]:
            if key in result and result[key]:
                abs_path = Path(result[key]).resolve()
                uploads_root_abs = UPLOAD_DIR.resolve()
                try:
                    relative_path = abs_path.relative_to(uploads_root_abs)
                    result[key] = str(relative_path).replace("\\", "/")
                except ValueError:
                    result[key] = abs_path.name

        yield json.dumps(result) + "\n"


    return StreamingResponse(generate(), media_type="text/plain")



@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "error": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"status": "error", "error": str(exc)},
    )


