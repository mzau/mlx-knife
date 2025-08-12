import json
import os
import signal
import sys
import time


def signal_handler(signum, frame):
    print("\n[WARNING] Download cancelled by user.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

os.environ["HF_HUB_DOWNLOAD_THREADS"] = "1"
os.environ["HF_HUB_DOWNLOAD_CHUNK_SIZE"] = "1048576"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "false"

try:
    import requests
    from huggingface_hub import snapshot_download
except ImportError:
    print("[ERROR] huggingface_hub or requests not installed in worker environment!")
    sys.exit(2)

# Throttle all HTTP(S) requests
original_get = requests.get
original_post = requests.post

def throttled_get(*args, **kwargs):
    response = original_get(*args, **kwargs)
    time.sleep(1.0)
    return response

def throttled_post(*args, **kwargs):
    response = original_post(*args, **kwargs)
    time.sleep(0.5)
    return response

requests.get = throttled_get
requests.post = throttled_post

def main():
    if len(sys.argv) != 2:
        print("Usage: python throttled_download_worker.py <kwargs_file.json>")
        sys.exit(1)

    kwargs_file = sys.argv[1]
    try:
        with open(kwargs_file) as f:
            kwargs_dict = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not read worker kwargs: {e}")
        sys.exit(1)

    try:
        snapshot_download(**kwargs_dict)
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        url = getattr(e.response, "url", None)
        if status == 401:
            print(f"[ERROR] Unauthorized (401): Check your HuggingFace token or login.\nURL: {url}")
            sys.exit(10)
        elif status == 403:
            print(f"[ERROR] Forbidden (403): Access denied.\nURL: {url}")
            sys.exit(11)
        elif status == 404:
            print(f"[ERROR] Not Found (404): Resource does not exist.\nURL: {url}")
            sys.exit(12)
        else:
            print(f"[ERROR] HTTP Error: {e}")
            sys.exit(2)
    except requests.exceptions.ConnectionError:
        print("[ERROR] Network connection error. Please check your internet connection and try again.")
        sys.exit(20)
    except PermissionError as e:
        print(f"[ERROR] Permission denied: {e.filename if hasattr(e, 'filename') else 'check file permissions'}")
        print("   Ensure you have write access to the cache directory.")
        sys.exit(13)
    except OSError as e:
        import errno
        if e.errno == errno.ENOSPC:
            print("[ERROR] No space left on device. Please free up disk space and try again.")
            sys.exit(14)
        elif e.errno == errno.EACCES:
            print(f"[ERROR] Access denied: {e.filename if hasattr(e, 'filename') else 'check permissions'}")
            sys.exit(13)
        else:
            print(f"[ERROR] OS Error during download: {e}")
            sys.exit(15)
    except Exception as e:
        print(f"[ERROR] Unexpected error during download: {type(e).__name__}: {e}")
        sys.exit(2)
    finally:
        try:
            os.unlink(kwargs_file)
        except Exception:
            pass

    sys.exit(0)

if __name__ == "__main__":
    main()
