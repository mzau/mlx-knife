import json
import os
import signal
import sys
import time
from typing import Any

# Global tracking for accurate download rate
_download_stats = {
    'bytes_downloaded': 0,
    'start_time': None,
    'last_update': None,
    'actual_download_time': 0.0  # Time spent actually downloading (without delays)
}


def signal_handler(signum: int, frame: Any) -> None:
    print("\n[WARNING] Download cancelled by user.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

os.environ["HF_HUB_DOWNLOAD_THREADS"] = "1"
os.environ["HF_HUB_DOWNLOAD_CHUNK_SIZE"] = "524288"  # 512KB chunks (half size)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "false"

try:
    import requests
    from huggingface_hub import snapshot_download
except ImportError:
    print("[ERROR] huggingface_hub or requests not installed in worker environment!")
    sys.exit(2)

# Throttle all HTTP(S) requests with adaptive delays
original_get = requests.get
original_post = requests.post

def get_adaptive_delay(url: str, response: Any) -> float:
    """Calculate delay based on file type and size"""
    if not url:
        return 1.0
    
    # Check if this is a large model file download
    if any(ext in url.lower() for ext in ['.safetensors', '.bin', '.pth']):
        # For large model files, use more aggressive throttling
        content_length = response.headers.get('content-length')
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > 100:  # Files larger than 100MB
                return 3.0     # 3 second delay between chunks
            elif size_mb > 10: # Files larger than 10MB
                return 2.0     # 2 second delay
        return 2.0  # Default for model files
    
    # Regular files (config.json, tokenizer files, etc.)
    return 0.5

def throttled_get(*args: Any, **kwargs: Any) -> Any:
    download_start = time.time()
    response = original_get(*args, **kwargs)
    download_end = time.time()
    
    # Track actual download time (without delays)
    actual_download_time = download_end - download_start
    _download_stats['actual_download_time'] += actual_download_time
    
    # Track bytes if we can determine them
    url = args[0] if args else kwargs.get('url', '')
    if hasattr(response, 'headers') and 'content-length' in response.headers:
        content_length = int(response.headers['content-length'])
        _download_stats['bytes_downloaded'] += content_length
        
        # Initialize timing if first download
        if _download_stats['start_time'] is None:
            _download_stats['start_time'] = download_start
        
        # Print accurate rate every ~5MB or every 10 seconds
        now = time.time()
        if (_download_stats['last_update'] is None or
            now - _download_stats['last_update'] > 10 or
            _download_stats['bytes_downloaded'] % (5 * 1024 * 1024) < content_length):
            
            if _download_stats['actual_download_time'] > 0:
                real_rate_mbps = (_download_stats['bytes_downloaded'] / _download_stats['actual_download_time']) / (1024 * 1024)
                total_mb = _download_stats['bytes_downloaded'] / (1024 * 1024)
                print(f"[THROTTLE] Downloaded {total_mb:.1f}MB at real rate: {real_rate_mbps:.1f}MB/s (excluding delays)")
            _download_stats['last_update'] = now
    
    delay = get_adaptive_delay(url, response)
    time.sleep(delay)
    return response

def throttled_post(*args: Any, **kwargs: Any) -> Any:
    response = original_post(*args, **kwargs)
    time.sleep(0.5)
    return response

requests.get = throttled_get
requests.post = throttled_post

def main() -> None:
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
