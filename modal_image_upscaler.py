import os
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
        f"mkdir -p {WEIGHTS_DIR} && wget {MODEL_URL} -O {WEIGHTS_DIR}/{MODEL_NAME}.pth",
    )
)

app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# STORAGE HELPERS (Cloudflare R2 / AWS S3 Compatible)
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


def list_s3_files(s3_client, bucket: str, prefix: str) -> set:
    """Lists relative file paths under a given prefix using pagination."""
    paginator = s3_client.get_paginator("list_objects_v2")
    files = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key != prefix and not key.endswith("/"):
                files.add(key.removeprefix(prefix))
    return files


# ---------------------------------------------------------------------------
# FUNCTION 1: UPSCALER WORKER CLASS (GPU Serverless Class with In-VRAM Model)
# ---------------------------------------------------------------------------
@app.cls(
    image=upscaler_image,
    gpu="T4",
    timeout=600,
    cpu=2.0,
    memory=4096,
    secrets=[storage_secret],
    scaledown_window=60,
)
class RealESRGANWorker:
    @modal.enter()
    def setup(self):
        """Loads Real-ESRGAN model into GPU VRAM once during container startup."""
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model_path = f"{WEIGHTS_DIR}/{MODEL_NAME}.pth"
        print(f"[⚡] Preloading {MODEL_NAME} weights into GPU VRAM...")

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        use_half = torch.cuda.is_available()

        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=use_half,
            gpu_id=0 if torch.cuda.is_available() else None,
        )
        print(f"[✓] Model loaded into VRAM (half_precision={use_half}). Ready for requests.")

    @modal.method()
    def upscale_image(
        self,
        filename: str,
        custom_input_prefix: Optional[str] = None,
        custom_output_prefix: Optional[str] = None,
        delete_input: bool = True,
        tile_size: int = 0,
        skip_existing: bool = True,
    ) -> dict:
        """
        Downloads image from R2/S3, upscales directly in GPU VRAM without subprocess overhead,
        uploads high-resolution output back to R2/S3, and cleans up input.
        If `skip_existing=True` and output asset is already present, bypasses GPU computation.
        """
        import cv2

        s3 = make_s3_client()
        bucket_name, default_input_prefix, default_output_prefix = get_storage_config()

        input_prefix = custom_input_prefix or default_input_prefix
        output_prefix = custom_output_prefix or default_output_prefix
        input_key = f"{input_prefix}{filename}"
        output_key = f"{output_prefix}{filename}"

        base_name = os.path.basename(filename)
        local_in = f"/tmp/in_{base_name}"
        local_out = f"/tmp/out_{base_name}"

        # Level 2 Safety Guard: Skip if output already exists
        if skip_existing:
            try:
                s3.head_object(Bucket=bucket_name, Key=output_key)
                print(f"[⏭] Output already exists in s3://{bucket_name}/{output_key}. Skipping GPU upscaling.")
                if delete_input:
                    print(f"[✂] Cleaning up original input: s3://{bucket_name}/{input_key}")
                    try:
                        s3.delete_object(Bucket=bucket_name, Key=input_key)
                    except Exception:
                        pass
                return {
                    "status": "skipped",
                    "reason": "already_exists",
                    "file": filename,
                    "output_key": output_key,
                    "bucket": bucket_name,
                }
            except Exception:
                # Not found, continue with upscale
                pass

        try:
            print(f"[↓] Downloading: s3://{bucket_name}/{input_key}")
            s3.download_file(bucket_name, input_key, local_in)

            img = cv2.imread(local_in, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to decode image at {local_in}")

            # Dynamic tiling control (prevents CUDA OOM on ultra-large images)
            self.upsampler.tile_size = tile_size if tile_size > 0 else 0

            print(f"[⚙] Processing with {MODEL_NAME} in GPU VRAM: {filename}")
            output, _ = self.upsampler.enhance(img, outscale=4)

            cv2.imwrite(local_out, output)

            print(f"[↑] Uploading high-res asset: s3://{bucket_name}/{output_key}")
            s3.upload_file(local_out, bucket_name, output_key)

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
            # Clean up local temporary files
            if os.path.exists(local_in):
                try:
                    os.remove(local_in)
                except Exception:
                    pass
            if os.path.exists(local_out):
                try:
                    os.remove(local_out)
                except Exception:
                    pass


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
    skip_existing: bool = True,
):
    """
    Public HTTP Web Endpoint (FastAPI) to trigger upscaling tasks.
    - If `filename` is specified: dispatches a single target file.
    - If `filename` is omitted: scans the input prefix and batch dispatches all matching images.
    - `skip_existing=True`: skips dispatching GPU workers if the image already exists in output/.
    """
    s3 = make_s3_client()
    bucket_name, default_input_prefix, default_output_prefix = get_storage_config()
    scan_prefix = prefix or default_input_prefix

    worker = RealESRGANWorker()

    # Mode 1: Single specified target
    if filename:
        clean_filename = filename.removeprefix(scan_prefix)
        output_key = f"{default_output_prefix}{clean_filename}"
        input_key = f"{scan_prefix}{clean_filename}"

        # Level 1 Check: Avoid spawning GPU worker if output already exists
        if skip_existing:
            try:
                s3.head_object(Bucket=bucket_name, Key=output_key)
                print(f"[⏭] Target already exists: s3://{bucket_name}/{output_key}. Skipping dispatch.")
                if delete_input:
                    try:
                        s3.delete_object(Bucket=bucket_name, Key=input_key)
                    except Exception:
                        pass
                return {
                    "status": "skipped",
                    "mode": "single_target",
                    "reason": "already_exists",
                    "file": clean_filename,
                    "bucket": bucket_name,
                    "output_key": output_key,
                }
            except Exception:
                # Object not in output, proceed to dispatch
                pass

        print(f"[🚀] Explicit target requested: {clean_filename}. Spawning upscaler...")
        worker.upscale_image.spawn(
            filename=clean_filename,
            custom_input_prefix=scan_prefix,
            custom_output_prefix=default_output_prefix,
            delete_input=delete_input,
            tile_size=tile_size,
            skip_existing=skip_existing,
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
    all_input_files = sorted(
        [
            f
            for f in list_s3_files(s3, bucket_name, scan_prefix)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ]
    )

    if not all_input_files:
        return {
            "status": "ignored",
            "message": f"Input folder is empty or contains no supported images in s3://{bucket_name}/{scan_prefix}",
            "targets_acquired": 0,
            "files": [],
        }

    # Level 1 Batch Filter: Skip files already in output
    skipped_files = []
    if skip_existing:
        existing_outputs = list_s3_files(s3, bucket_name, default_output_prefix)
        targets_to_process = [f for f in all_input_files if f not in existing_outputs]
        skipped_files = [f for f in all_input_files if f in existing_outputs]

        if skipped_files:
            print(f"[⏭] Skipping {len(skipped_files)} file(s) already present in {default_output_prefix}")
            if delete_input:
                for f in skipped_files:
                    try:
                        s3.delete_object(Bucket=bucket_name, Key=f"{scan_prefix}{f}")
                    except Exception:
                        pass
        files_to_dispatch = targets_to_process
    else:
        files_to_dispatch = all_input_files

    if not files_to_dispatch:
        return {
            "status": "all_skipped" if skipped_files else "ignored",
            "message": f"All {len(skipped_files)} matching image(s) already exist in s3://{bucket_name}/{default_output_prefix}. None dispatched.",
            "targets_acquired": 0,
            "skipped_count": len(skipped_files),
            "skipped_files": skipped_files,
            "bucket": bucket_name,
        }

    print(f"[🚀] Found {len(files_to_dispatch)} target(s) to process ({len(skipped_files)} skipped). Spawning workers...")
    for f in files_to_dispatch:
        worker.upscale_image.spawn(
            filename=f,
            custom_input_prefix=scan_prefix,
            custom_output_prefix=default_output_prefix,
            delete_input=delete_input,
            tile_size=tile_size,
            skip_existing=skip_existing,
        )

    return {
        "status": "success",
        "targets_acquired": len(files_to_dispatch),
        "files": files_to_dispatch,
        "skipped_count": len(skipped_files),
        "skipped_files": skipped_files,
        "bucket": bucket_name,
        "input_prefix": scan_prefix,
        "output_prefix": default_output_prefix,
    }


# ---------------------------------------------------------------------------
# LOCAL CLI ENTRYPOINT
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    filename: Optional[str] = None,
    skip_existing: bool = True,
):
    """
    Run locally from your machine:
      modal run modal_image_upscaler.py
      modal run modal_image_upscaler.py --filename "sample.jpg"
      modal run modal_image_upscaler.py --no-skip-existing
    """
    print("🚀 Triggering upscale task remotely via Modal local entrypoint...")
    result = trigger_upscaler.remote(filename=filename, skip_existing=skip_existing)
    print("Response from remote endpoint:")
    print(result)
