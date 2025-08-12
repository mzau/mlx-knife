import json
import os
import subprocess
import sys
import tempfile

try:
    from .cache_utils import (
        MODEL_CACHE,
        hf_to_cache_dir,
        is_model_healthy,
        parse_model_spec,
    )
except ImportError:
    from pathlib import Path
    def parse_model_spec(x): return (x, None)
    def hf_to_cache_dir(x): return x
    MODEL_CACHE = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub")))
    def is_model_healthy(x): return False

def describe_http_exception(exc):
    if hasattr(exc, "response") and exc.response is not None:
        status = getattr(exc.response, "status_code", None)
        url = getattr(exc.response, "url", None)
        if status == 401:
            return f"[ERROR] Unauthorized (401): Check your HuggingFace token or login.\nURL: {url}"
        elif status == 403:
            return f"[ERROR] Forbidden (403): Access denied.\nURL: {url}"
        elif status == 404:
            return f"[ERROR] Not Found (404): Resource does not exist.\nURL: {url}"
        elif status >= 500:
            return f"[ERROR] Server Error ({status}): Problem on HuggingFace's side.\nURL: {url}\nTry again later."
        else:
            return f"[ERROR] HTTP Error {status}: {exc}\nURL: {url}"
    return f"[ERROR] HTTP Error: {exc}"

def configure_download_environment():
    os.environ['HF_HUB_DOWNLOAD_THREADS'] = '1'
    os.environ['HF_HUB_DOWNLOAD_CHUNK_SIZE'] = '1048576'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = 'false'

def pull_model(model_spec):
    original_spec = model_spec
    model_name, commit_hash = parse_model_spec(model_spec)

    if "/" not in original_spec.split("@")[0] and "/" in model_name:
        confirm = input(f"Download '{model_name}'? [Y/n] ")
        if confirm.lower() == "n":
            print("Download cancelled.")
            return

    base_cache_dir = MODEL_CACHE / hf_to_cache_dir(model_name)
    if commit_hash:
        hash_dir = base_cache_dir / "snapshots" / commit_hash
        if hash_dir.exists() and is_model_healthy(f"{model_name}@{commit_hash}"):
            print("Model already exists")
            return
    else:
        if base_cache_dir.exists() and is_model_healthy(model_name):
            print("Model already exists")
            return

    print(f"Downloading {model_name}...")

    # Build kwargs dict for the worker
    kwargs_dict = {
        "repo_id": model_name,
        "local_dir_use_symlinks": False,
        "max_workers": 1
    }
    if commit_hash:
        kwargs_dict["revision"] = commit_hash
    if "mlx-community" in model_name:
        kwargs_dict["allow_patterns"] = [
            "*.json", "*.txt", "*.safetensors", "*.md", "*.gitattributes", "LICENSE"
        ]
    if "mlx-community" not in model_name:
        confirm = input(f"[WARNING] {model_name} is not an MLX model (may be >1GB). Continue? [y/N] ")
        if confirm.lower() != "y":
            print("Download cancelled.")
            return

    kwargs_str = json.dumps(kwargs_dict, indent=2)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(kwargs_str)
        kwargs_file = f.name

    # Call the worker as subprocess with nice priority
    worker_path = os.path.join(os.path.dirname(__file__), "throttled_download_worker.py")
    try:
        result = subprocess.run(
            ['nice', '-n', '19', sys.executable, worker_path, kwargs_file],
            check=False
        )
        if result.returncode == 0:
            print("Download completed successfully.")
        elif result.returncode in (10, 11, 12, 13, 14, 15):
            # Already handled in worker, do NOT retry fallback
            print("[WARNING] Fatal error encountered in throttled download, not attempting fallback.")
            return
        else:
            print("[WARNING] Throttled download failed or was interrupted.")
            print("Attempting fallback download with standard throttling...")
            try:
                import requests
                from huggingface_hub import snapshot_download
                configure_download_environment()
                snapshot_download(**kwargs_dict)
                print("Download completed successfully.")
            except requests.exceptions.HTTPError as e:
                print(describe_http_exception(e))
                return
            except requests.exceptions.ConnectionError:
                print("[ERROR] Network connection error. Please check your internet connection and try again.")
                return
            except requests.exceptions.Timeout:
                print("[ERROR] Download timed out. Please try again.")
                return
            except KeyboardInterrupt:
                print("\n[WARNING] Download cancelled by user.")
                return
            except Exception as e:
                print(f"[ERROR] Unexpected error during fallback download: {type(e).__name__}: {e}")
                return
    except KeyboardInterrupt:
        print("\n[WARNING] Download cancelled by user.")
        return
    except ImportError:
        print("huggingface-hub is not installed. Please install it with: pip install huggingface-hub")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {type(e).__name__}: {e}")
