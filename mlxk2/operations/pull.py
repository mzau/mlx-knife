from ..core.cache import MODEL_CACHE, hf_to_cache_dir
from ..core.model_resolution import resolve_model_for_operation
from .health import is_model_healthy
import os


# Pull uses exact user input - HuggingFace resolves model names

def preflight_repo_access(model_name, hf_api=None):
    """Check repository access before download to prevent cache pollution.

    Issue #30: Fail fast for gated/private or non-existent repos without starting any download.

    Args:
        model_name: Repository name to check
        hf_api: Optional injected `HfApi` instance (testability)

    Returns:
        (success: bool, error_message: str or None)
    """
    try:
        # Lazy imports with robust error shims across hub versions
        import huggingface_hub as _hub
        from huggingface_hub import HfApi
        try:
            from requests.exceptions import HTTPError, Timeout  # type: ignore
        except Exception:  # requests may not be present in minimal envs
            HTTPError = Timeout = None  # type: ignore

        hub_errors = getattr(_hub, "errors", None)

        api = hf_api or HfApi()

        # Prefer modern token name in messages, but accept legacy var when present
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        try:
            # Lightweight metadata request (no file download)
            api.model_info(model_name, token=token)
            return True, None

        except Exception as e:  # Map known cases first, then fallbacks
            # 1) Map huggingface_hub specific errors if available
            if hub_errors is not None:
                GatedRepoError = getattr(hub_errors, "GatedRepoError", None)
                RepositoryNotFoundError = getattr(hub_errors, "RepositoryNotFoundError", None)
                HfHubHTTPError = getattr(hub_errors, "HfHubHTTPError", None)
                HfHubError = getattr(hub_errors, "HfHubError", None)

                if GatedRepoError and isinstance(e, GatedRepoError):
                    return False, (
                        f"Access denied: gated/private model '{model_name}'. "
                        f"Accept terms and set HF_TOKEN."
                    )
                if RepositoryNotFoundError and isinstance(e, RepositoryNotFoundError):
                    # Security feature: HG often returns access denied semantics for missing
                    return False, f"Access denied or not found for '{model_name}'."
                # Generic hub HTTP error with status code
                if (HfHubHTTPError and isinstance(e, HfHubHTTPError)) or (HfHubError and isinstance(e, HfHubError)):
                    resp = getattr(e, "response", None)
                    code = getattr(resp, "status_code", None)
                    if code in (401, 403):
                        return False, f"Access denied to model '{model_name}'. Set HF_TOKEN."
                    if code:
                        # Non-auth HTTP issues during preflight: degrade gracefully to download stage
                        return True, f"Preflight HTTP {code}; continuing to download stage."
                    # Fallback without code → degrade gracefully
                    return True, "Preflight error without HTTP code; continuing."

            # 2) requests timeouts / HTTP errors (when surfaced directly)
            if Timeout and isinstance(e, Timeout):  # type: ignore[arg-type]
                # Network timeout during preflight: degrade to download stage
                return True, f"Preflight timeout for '{model_name}'; continuing to download stage."
            if HTTPError and isinstance(e, HTTPError):  # type: ignore[arg-type]
                code = getattr(getattr(e, "response", None), "status_code", None)
                if code in (401, 403):
                    return False, f"Access denied to model '{model_name}'. Set HF_TOKEN."
                if code:
                    return True, f"Preflight HTTP {code}; continuing to download stage."
                return True, "Preflight HTTP error; continuing."

            # 3) Generic fallback based on message hints
            msg = str(e).lower()
            # Hard fail on clear access-denied/gated patterns
            if any(h in msg for h in ("forbidden", "unauthorized", "denied", "gated", "private")):
                return False, f"Access denied or gated/private for '{model_name}'."
            if "not found" in msg:
                return False, f"Access denied or not found for '{model_name}'."

            # Unknown errors → degrade gracefully to allow downstream error surface
            return True, f"Preflight error: {str(e)}; continuing to download stage."

    except ImportError:
        # No preflight available → fail safe, include expected keywords
        return False, "Access denied or not found (preflight unavailable; install huggingface-hub)."

    except Exception as e:
        # Unknown errors → fail safe, include expected keywords
        return False, f"Access denied or gated/private (preflight failed: {str(e)}). Set HF_TOKEN if needed."


def pull_model_with_huggingface_hub(model_name, cache_dir=None):
    """Use huggingface-hub to pull a model to specified cache directory."""
    try:
        # Just-in-time suppression for macOS Python 3.9 LibreSSL warning
        import warnings as _warnings
        _warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL 1.1.1+')
        # Use direct Python API instead of CLI
        from huggingface_hub import snapshot_download

        # Download model to specified cache or default
        kwargs = {
            "repo_id": model_name,
            "local_files_only": False,
            "resume_download": True
        }
        if cache_dir:
            kwargs["cache_dir"] = str(cache_dir)

        local_dir = snapshot_download(**kwargs)

        return True, f"Downloaded to {local_dir}"

    except ImportError:
        return False, "huggingface-hub not installed (pip install huggingface-hub)"
    except Exception as e:
        return False, f"Download failed: {str(e)}"


def pull_operation(model_spec):
    """Pull (download) operation for JSON API."""
    result = {
        "status": "success",
        "command": "pull",
        "error": None,
        "data": {
            "model": None,
            "download_status": "unknown",
            "message": "",
            "expanded_name": None
        }
    }
    
    try:
        # Early validation before any network/library usage
        if not model_spec or not str(model_spec).strip():
            result["status"] = "error"
            result["error"] = {
                "type": "ValidationError",
                "message": "Invalid model name: empty",
            }
            result["data"]["download_status"] = "error"
            return result

        base_spec = str(model_spec).split("@", 1)[0]
        # HF repo id soft rules (MVP): length, bad slashes; allow single-segment as fuzzy/alias
        if len(base_spec) > 96 or base_spec.startswith("/") or base_spec.endswith("/") or "//" in base_spec:
            result["status"] = "error"
            result["error"] = {
                "type": "ValidationError",
                "message": "Invalid model name: must be <= 96 chars and not contain leading/trailing or double slashes",
            }
            result["data"]["download_status"] = "error"
            return result

        # Use model resolution for fuzzy matching and expansion
        resolved_name, commit_hash, ambiguous_matches = resolve_model_for_operation(model_spec)
        
        if ambiguous_matches:
            result["status"] = "error"
            result["error"] = {
                "type": "ambiguous_match",
                "message": f"Multiple models match '{model_spec}'",
                "matches": ambiguous_matches
            }
            return result
        elif not resolved_name:
            # No existing model found - use original spec for download as-is
            if "@" in model_spec:
                model_name, commit_hash = model_spec.rsplit("@", 1)
                result["data"]["commit_hash"] = commit_hash
            else:
                model_name = model_spec
                commit_hash = None
            resolved_name = model_name  # Use exact name - let HuggingFace resolve it
        
        result["data"]["model"] = resolved_name
        result["data"]["expanded_name"] = resolved_name if resolved_name != model_spec.split('@')[0] else None
        if commit_hash:
            result["data"]["commit_hash"] = commit_hash
        
        # Check if already exists and is healthy
        cache_dir = MODEL_CACHE / hf_to_cache_dir(resolved_name)
        if cache_dir.exists():
            healthy, _ = is_model_healthy(resolved_name)
            if healthy:
                result["data"]["download_status"] = "already_exists"
                result["data"]["message"] = f"Model {resolved_name} already exists in cache"
                return result
            else:
                # Model exists but unhealthy - suggest rm workflow
                result["status"] = "error"
                result["error"] = {
                    "type": "model_corrupted",
                    "message": f"Model exists but is corrupted. Use 'rm {model_spec}' first, then pull again."
                }
                result["data"]["download_status"] = "corrupted"
                return result
        
        # Preflight check for repository access (Issue #30)
        result["data"]["download_status"] = "checking_access"
        preflight_success, preflight_error = preflight_repo_access(resolved_name)
        
        if not preflight_success:
            result["status"] = "error"
            result["data"]["download_status"] = "access_denied"
            result["error"] = {
                "type": "access_denied",
                "message": preflight_error
            }
            return result
        elif preflight_error:
            # Warning case - log but continue
            result["data"]["preflight_warning"] = preflight_error
        
        # Attempt download
        result["data"]["download_status"] = "downloading"
        success, message = pull_model_with_huggingface_hub(resolved_name)
        
        if success:
            result["data"]["download_status"] = "success"
            result["data"]["message"] = message
        else:
            result["status"] = "error"
            result["data"]["download_status"] = "failed"
            result["error"] = {
                "type": "download_failed",
                "message": message
            }
    
    except Exception as e:
        result["status"] = "error"
        result["error"] = {
            "type": "pull_operation_failed", 
            "message": str(e)
        }
        result["data"]["download_status"] = "error"

    return result


def pull_to_cache(model_spec, cache_dir):
    """Pull model to specific cache directory - used by clone operation."""
    result = {
        "status": "success",
        "command": "pull",
        "error": None,
        "data": {
            "model": None,
            "download_status": "unknown",
            "message": "",
            "expanded_name": None
        }
    }

    try:
        # Basic validation
        if not model_spec or not str(model_spec).strip():
            result["status"] = "error"
            result["error"] = {
                "type": "ValidationError",
                "message": "Invalid model name: empty",
            }
            result["data"]["download_status"] = "error"
            return result

        base_spec = str(model_spec).split("@", 1)[0]
        if len(base_spec) > 96 or base_spec.startswith("/") or base_spec.endswith("/") or "//" in base_spec:
            result["status"] = "error"
            result["error"] = {
                "type": "ValidationError",
                "message": "Invalid model name: must be <= 96 chars and not contain leading/trailing or double slashes",
            }
            result["data"]["download_status"] = "error"
            return result

        # For clone operations, use model spec as-is (no fuzzy resolution)
        model_name = model_spec
        result["data"]["model"] = model_name
        result["data"]["expanded_name"] = model_name

        # Preflight check for repository access (Issue #30)
        result["data"]["download_status"] = "checking_access"
        preflight_success, preflight_error = preflight_repo_access(model_name)

        if not preflight_success:
            result["status"] = "error"
            result["data"]["download_status"] = "access_denied"
            result["error"] = {
                "type": "access_denied",
                "message": preflight_error
            }
            return result
        elif preflight_error:
            # Warning case - log but continue
            result["data"]["preflight_warning"] = preflight_error

        # Download to specified cache directory
        result["data"]["download_status"] = "downloading"
        success, message = pull_model_with_huggingface_hub(model_name, cache_dir)

        if success:
            result["data"]["download_status"] = "success"
            result["data"]["message"] = message
        else:
            result["status"] = "error"
            result["error"] = {
                "type": "DownloadError",
                "message": message
            }
            result["data"]["download_status"] = "error"

    except Exception as e:
        result["status"] = "error"
        result["error"] = {
            "type": "OperationError",
            "message": f"Unexpected error during pull: {str(e)}"
        }
        result["data"]["download_status"] = "error"

    return result
