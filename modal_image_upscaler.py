import glob
import os
import shutil
import subprocess
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# CONFIGURATION & MODAL SECRETS
# ---------------------------------------------------------------------------
APP_NAME = "modal-image-upscaler"
SECRET_NAME = os.environ.get("MODAL_SECRET_NAME", "r2-credentials")

# Built-in Modal Secret injection (reads R2/S3 keys securely from Modal)
storage_secret = modal.Secret.from_name(SECRET_NAME)

WEIGHTS_DIR = "/root/weights"
MODEL_NAME = "RealESRGAN_x4plus"
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
INFERENCE_SCRIPT_URL = "https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/inference_realesrgan.py"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")

# ---------------------------------------------------------------------------
# CONTAINER IMAGES
# ---------------------------------------------------------------------------
# 1. Ultra-lightweight CPU image for the web endpoint trigger (rapid cold starts)
trigger_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("boto3", "fastapi[standard]")
)

# 2. GPU-optimized PyTorch container with Real-ESRGAN & model weights baked in
upscaler_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "wget", "git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu118",
    )
    .pip_install("boto3", "tqdm", "opencv-python", "fastapi[standard]")
    .run_commands(
        "pip install setuptools==69.5.1",
        "pip install git+https://github.com/xinntao/BasicSR.git --no-build-isolation",
        "pip install realesrgan --no-deps",
    )
    .run_commands("pip install 'numpy<2' --force-reinstall")
    .run_commands(
        f"wget {INFERENCE_SCRIPT_URL} -O /root/inference_realesrgan.py",
        f"mkdir -p {WEIGHTS_DIR} && wget {MODEL_URL} -O {WEIGHTS_DIR}/{MODEL_NAME}.pth",
    )
)

app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# STORAGE HELPER (Cloudflare R2 / AWS S3 Compatible)
# ---------------------------------------------------------------------------
def make_s3_client():
    """Constructs an S3/R2 boto3 client using official Modal secrets injected as environment variables."""
    import boto3
    from botocore.config import Config

    endpoint_url = (
        os.environ.get("R2_ENDPOINT_URL")
        or os.environ.get("R2_ENDPOINT")
        or os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("S3_ENDPOINT_URL")
    )
    access_key = (
        os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("R2_ACCESS_KEY_ID")
        or os.environ.get("R2_ACCESS_KEY")
    )
    secret_key = (
        os.environ.get("AWS_SECRET_ACCESS_KEY")
        or os.environ.get("R2_SECRET_ACCESS_KEY")
        or os.environ.get("R2_SECRET_KEY")
    )
    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "auto"
    )

    client_kwargs = {
        "service_name": "s3",
        "config": Config(signature_version="s3v4"),
    }
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    if access_key and secret_key:
        client_kwargs["aws_access_key_id"] = access_key
        client_kwargs["aws_secret_access_key"] = secret_key
    if region:
        client_kwargs["region_name"] = region

    return boto3.client(**client_kwargs)


def get_storage_config():
    """Retrieves bucket name and input/output prefixes from secrets or defaults."""
    bucket_name = (
        os.environ.get("BUCKET_NAME")
        or os.environ.get("R2_BUCKET_NAME")
        or "playground"
    )
    input_prefix = os.environ.get("INPUT_PREFIX", "input/")
    output_prefix = os.environ.get("OUTPUT_PREFIX", "output/")

    if input_prefix and not input_prefix.endswith("/"):
        input_prefix += "/"
    if output_prefix and not output_prefix.endswith("/"):
        output_prefix += "/"

    return bucket_name, input_prefix, output_prefix


# ---------------------------------------------------------------------------
# FUNCTION 1: UPSCALER WORKER (GPU Serverless Function)
# ---------------------------------------------------------------------------
@app.function(
    image=upscaler_image,
    gpu="T4",
    timeout=600,
    cpu=2.0,
    memory=4096,
    secrets=[storage_secret],
)
def upscale_image_worker(
    filename: str,
    custom_input_prefix: Optional[str] = None,
    custom_output_prefix: Optional[str] = None,
    delete_input: bool = True,
    tile_size: int = 0,
) -> dict:
    """
    Downloads image from R2/S3, upscales with Real-ESRGAN x4plus on T4 GPU,
    uploads high-resolution output back to R2/S3, and cleans up input.
    """
    s3 = make_s3_client()
    bucket_name, default_input_prefix, default_output_prefix = get_storage_config()

    input_prefix = custom_input_prefix or default_input_prefix
    output_prefix = custom_output_prefix or default_output_prefix
    input_key = f"{input_prefix}{filename}"
    output_key = f"{output_prefix}{filename}"

    local_in = f"/tmp/{filename}"
    local_out_dir = f"/tmp/out_{os.path.splitext(filename)[0]}"

    if os.path.exists(local_out_dir):
        shutil.rmtree(local_out_dir)
    os.makedirs(local_out_dir, exist_ok=True)

    try:
        print(f"[↓] Downloading: s3://{bucket_name}/{input_key}")
        s3.download_file(bucket_name, input_key, local_in)

        model_weights_path = f"{WEIGHTS_DIR}/{MODEL_NAME}.pth"
        cmd = [
            "python3",
            "/root/inference_realesrgan.py",
            "-n",
            MODEL_NAME,
            "--model_path",
            model_weights_path,
            "-i",
            local_in,
            "-o",
            local_out_dir,
            "-s",
            "4",
            "--fp32",
        ]
        if tile_size > 0:
            cmd.extend(["-t", str(tile_size)])

        print(f"[⚙] Processing with {MODEL_NAME}: {filename}")
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Real-ESRGAN outputs upscaled file into local_out_dir
        produced = glob.glob(os.path.join(local_out_dir, "*"))
        if not produced:
            raise FileNotFoundError(
                f"Upscale script finished but no file generated. Stderr: {res.stderr}"
            )

        local_output_path = produced[0]

        print(f"[↑] Uploading high-res asset: s3://{bucket_name}/{output_key}")
        s3.upload_file(local_output_path, bucket_name, output_key)

        if delete_input:
            print(f"[✂] Cleaning up original input: s3://{bucket_name}/{input_key}")
            s3.delete_object(Bucket=bucket_name, Key=input_key)

        print(f"[✓] Upscaling succeeded: {filename}")
        return {
            "status": "ok",
            "file": filename,
            "output_key": output_key,
            "bucket": bucket_name,
        }

    except Exception as e:
        print(f"[✗] Error during upscaling of {filename}: {e}")
        return {
            "status": "failed",
            "file": filename,
            "error": str(e),
        }
    finally:
        # Guarantee local temporary files are cleaned up
        if os.path.exists(local_in):
            os.remove(local_in)
        if os.path.exists(local_out_dir):
            shutil.rmtree(local_out_dir)


# ---------------------------------------------------------------------------
# FUNCTION 2: TRIGGER ENDPOINT (FastAPI Web Function)
# ---------------------------------------------------------------------------
@app.function(
    image=trigger_image,
    timeout=300,
    secrets=[storage_secret],
)
@modal.fastapi_endpoint(method="GET")
def trigger_upscaler(
    filename: Optional[str] = None,
    prefix: Optional[str] = None,
    delete_input: bool = True,
    tile_size: int = 0,
):
    """
    Public HTTP Web Endpoint (FastAPI) to trigger upscaling tasks.
    - If `filename` is specified: dispatches a single target file.
    - If `filename` is omitted: scans the input prefix and batch dispatches all matching images.
    """
    s3 = make_s3_client()
    bucket_name, default_input_prefix, default_output_prefix = get_storage_config()
    scan_prefix = prefix or default_input_prefix

    # Mode 1: Single specified target
    if filename:
        clean_filename = filename.removeprefix(scan_prefix)
        print(f"[🚀] Explicit target requested: {clean_filename}. Spawning upscaler...")
        upscale_image_worker.spawn(
            filename=clean_filename,
            custom_input_prefix=scan_prefix,
            custom_output_prefix=default_output_prefix,
            delete_input=delete_input,
            tile_size=tile_size,
        )
        return {
            "status": "dispatched",
            "mode": "single_target",
            "file": clean_filename,
            "bucket": bucket_name,
            "input_prefix": scan_prefix,
            "output_prefix": default_output_prefix,
        }

    # Mode 2: Batch scan and dispatch
    print(f"[🔍] Scanning s3://{bucket_name}/{scan_prefix} for new images...")
    resp = s3.list_objects_v2(Bucket=bucket_name, Prefix=scan_prefix)

    files = [
        obj["Key"].removeprefix(scan_prefix)
        for obj in resp.get("Contents", [])
        if obj["Key"] != scan_prefix
        and obj["Key"].lower().endswith(SUPPORTED_EXTENSIONS)
        and not obj["Key"].endswith("/")
    ]

    if not files:
        return {
            "status": "ignored",
            "message": f"Input folder is empty or contains no supported images in s3://{bucket_name}/{scan_prefix}",
            "targets_acquired": 0,
            "files": [],
        }

    print(f"[🚀] Found {len(files)} target(s). Spawning workers...")
    for f in files:
        upscale_image_worker.spawn(
            filename=f,
            custom_input_prefix=scan_prefix,
            custom_output_prefix=default_output_prefix,
            delete_input=delete_input,
            tile_size=tile_size,
        )

    return {
        "status": "success",
        "targets_acquired": len(files),
        "files": files,
        "bucket": bucket_name,
        "input_prefix": scan_prefix,
        "output_prefix": default_output_prefix,
    }


# ---------------------------------------------------------------------------
# LOCAL CLI ENTRYPOINT
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(filename: Optional[str] = None):
    """
    Run locally from your machine:
      modal run modal_image_upscaler.py
      modal run modal_image_upscaler.py --filename "sample.jpg"
    """
    print("🚀 Triggering upscale task remotely via Modal local entrypoint...")
    result = trigger_upscaler.remote(filename=filename)
    print("Response from remote endpoint:")
    print(result)
