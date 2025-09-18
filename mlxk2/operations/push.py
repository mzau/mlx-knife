"""Experimental push operation for MLX-Knife 2.0 (M0: upload only).

This is a minimal, JSON-first implementation that uploads a local folder
to a Hugging Face model repository using huggingface_hub.upload_folder.

Scope (M0):
- No validation, no filters, no manifests.
- Requires HF_TOKEN environment variable.
- Default branch is main (configurable via CLI).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import json as _json

# Import APFS check from clone operation and cache utilities
from mlxk2.operations.clone import _is_apfs_filesystem
from mlxk2.core.cache import get_current_cache_root


DEFAULT_PUSH_BRANCH = "main"


def push_operation(
    local_dir: str,
    repo_id: str,
    create: bool = False,
    private: bool = False,
    branch: str = DEFAULT_PUSH_BRANCH,
    commit_message: str | None = None,
    check_only: bool = False,
    quiet: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Perform a minimal push (upload) to Hugging Face Hub.

    Returns a JSON-serializable result dict following the 2.0 pattern.
    """
    result: Dict[str, Any] = {
        "status": "success",
        "command": "push",
        "error": None,
        "data": {
            "repo_id": repo_id,
            "branch": branch or DEFAULT_PUSH_BRANCH,
            "commit_sha": None,
            "commit_url": None,
            "repo_url": f"https://huggingface.co/{repo_id}",
            # Number of actually uploaded/changed files (when available).
            "uploaded_files_count": None,
            # Local count of files scanned in the folder (approximation, optional).
            "local_files_count": None,
            # Indicates whether the Hub performed a no-op (no changes to commit).
            "no_changes": None,
            # Whether the repository was created in this operation.
            "created_repo": False,
            # Optional short message for humans (kept in JSON too for clarity).
            "message": None,
            "experimental": True,
            "disclaimer": (
                "Experimental feature (M0: upload only). No validation/filters; "
                "review results on the Hub."
            ),
        },
    }

    try:
        # 1) Token (skip for check-only)
        hf_token = os.environ.get("HF_TOKEN")
        if not check_only and not hf_token:
            result["status"] = "error"
            result["error"] = {
                "type": "auth_error",
                "message": "HF_TOKEN not set",
            }
            return result

        # 2) Local folder
        p = Path(local_dir)
        if not p.exists() or not p.is_dir():
            result["status"] = "error"
            result["error"] = {
                "type": "workspace_not_found",
                "message": f"Workspace not found or not a directory: {local_dir}",
            }
            return result

        # Optional approximate count (local view)
        try:
            approx_count = sum(1 for _ in p.rglob("*") if _.is_file())
            result["data"]["local_files_count"] = approx_count
        except Exception:
            pass

        # 2a) Build ignore patterns early (used by dry-run and upload)
        ignore_patterns = [
            "**/.git/**",
            "**/.git",
            "**/.DS_Store",
            ".DS_Store",
            "**/.hfignore",
            ".hfignore",
            "**/.gitignore",
            ".gitignore",
            "**/__pycache__/**",
            "**/.venv/**",
            "**/venv/**",
            "**/*.pyc",
        ]
        hfignore = p / ".hfignore"
        if hfignore.exists():
            try:
                extra_patterns = []
                for line in hfignore.read_text().splitlines():
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    extra_patterns.append(s)
                if extra_patterns:
                    seen = set()
                    merged = []
                    for pat in ignore_patterns + extra_patterns:
                        if pat not in seen:
                            merged.append(pat)
                            seen.add(pat)
                    ignore_patterns = merged
            except Exception:
                # Ignore read/parse errors silently in M0
                pass

        # 2b) Check-only: analyze workspace and return without contacting HF
        if check_only:
            diag = _analyze_workspace(p)
            result["data"]["no_changes"] = None
            result["data"]["message"] = "Check-only: no upload performed."
            result["data"]["workspace_health"] = diag
            return result

        # 3) Import hub pieces lazily and perform repo checks / upload
        # Suppress macOS Python 3.9 LibreSSL warning like pull operation
        import warnings as _warnings

        _warnings.filterwarnings(
            "ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+"
        )

        try:
            from huggingface_hub import HfApi, upload_folder
            from huggingface_hub.errors import (
                HfHubHTTPError,
                RepositoryNotFoundError,
                RevisionNotFoundError,
            )
        except Exception:
            result["status"] = "error"
            result["error"] = {
                "type": "dependency_missing",
                "message": "huggingface-hub not installed (pip install huggingface-hub)",
            }
            return result

        api = HfApi(token=hf_token)

        # 4) Ensure repo exists (model type). Do not auto-create branch here.
        created_repo = False
        try:
            # If branch does not exist, this raises RevisionNotFoundError.
            api.repo_info(repo_id=repo_id, repo_type="model", revision=branch)
        except RepositoryNotFoundError:
            if dry_run:
                # For dry-run, do not create; compute that all files would be added
                local_files = _collect_local_files(p, ignore_patterns)
                result["data"].update({
                    "dry_run": True,
                    "no_changes": False if local_files else True,
                    "uploaded_files_count": 0,
                    "change_summary": {"added": len(local_files), "modified": 0, "deleted": 0},
                    "dry_run_summary": {"added": len(local_files), "modified": 0, "deleted": 0},
                    "message": "Dry-run: repository does not exist; would create and add all files.",
                    "would_create_repo": True,
                    "would_create_branch": True,
                })
                return result
            if not create:
                result["status"] = "error"
                result["error"] = {
                    "type": "repo_not_found",
                    "message": f"Repository not found: {repo_id} (use --create)",
                }
                return result
            # Try create repository (exist_ok=True covers races)
            api.create_repo(
                repo_id=repo_id, repo_type="model", private=private, exist_ok=True
            )
            # After create, no guarantee branch exists; upload_folder below will target revision
            created_repo = True
            # Ensure target branch exists if not default
            try:
                if branch and branch != DEFAULT_PUSH_BRANCH:
                    api.create_branch(repo_id=repo_id, repo_type="model", branch=branch)
            except HfHubHTTPError as e:
                result["status"] = "error"
                result["error"] = {
                    "type": "branch_create_failed",
                    "message": str(e),
                }
                return result
        except RevisionNotFoundError:
            # Repo exists but branch doesn't.
            if dry_run:
                local_files = _collect_local_files(p, ignore_patterns)
                result["data"].update({
                    "dry_run": True,
                    "no_changes": False if local_files else True,
                    "uploaded_files_count": 0,
                    "change_summary": {"added": len(local_files), "modified": 0, "deleted": 0},
                    "dry_run_summary": {"added": len(local_files), "modified": 0, "deleted": 0},
                    "message": "Dry-run: branch does not exist; would create branch and add all files.",
                    "would_create_repo": False,
                    "would_create_branch": True,
                })
                return result
            # If user asked to create, proactively create the branch to avoid 404 on preupload;
            # otherwise, tolerate and let upload_folder attempt (offline tests expect this).
            if create:
                try:
                    api.create_branch(repo_id=repo_id, repo_type="model", branch=branch)
                except HfHubHTTPError:
                    # Do not fail early; fall through and let upload attempt once
                    pass

        # 4b) If dry-run and repo/branch exist: compute diff vs remote and return
        if dry_run:
            try:
                remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="model", revision=branch or DEFAULT_PUSH_BRANCH) or [])
            except Exception:
                remote_files = set()
            local_files = set(_collect_local_files(p, ignore_patterns))
            added = sorted(list(local_files - remote_files))
            deleted = sorted(list(remote_files - local_files))
            # Modified cannot be reliably computed without fetching metadata
            modified = None
            no_changes = (len(added) == 0 and len(deleted) == 0)
            result["data"].update({
                "dry_run": True,
                "no_changes": True if no_changes else False,
                "uploaded_files_count": 0,
                "change_summary": {"added": len(added), "modified": 0, "deleted": len(deleted)},
                "dry_run_summary": {"added": len(added), "modified": modified, "deleted": len(deleted)},
                "message": ("Dry-run: no changes" if no_changes else f"Dry-run: +{len(added)} ~? -{len(deleted)}"),
                "would_create_repo": False,
                "would_create_branch": False,
                "added_files": added[:20] if added else [],
                "deleted_files": deleted[:20] if deleted else [],
            })
            return result

        # 5) Upload folder
        commit_msg = commit_message or "mlx-knife push"
        # ignore_patterns prepared earlier

        # Capture hub logs to enrich JSON (e.g., no-op messages) and optionally silence console noise in JSON mode
        hf_logs = None
        try:
            import logging as _logging
            import contextlib as _contextlib
            _hf_logger = _logging.getLogger("huggingface_hub")

            class _BufHandler(_logging.Handler):
                def __init__(self):
                    super().__init__()
                    self.buf = []
                def emit(self, record):
                    try:
                        msg = self.format(record)
                    except Exception:
                        msg = str(record.getMessage()) if hasattr(record, "getMessage") else str(record)
                    self.buf.append(msg)

            _handler = _BufHandler()
            _handler.setLevel(_logging.INFO)
            _old_level = _hf_logger.level
            _old_handlers = list(_hf_logger.handlers)
            _old_propagate = _hf_logger.propagate

            # In quiet mode (JSON without --verbose), avoid emitting hub logs/progress to the console
            # 1) disable progress bars via env (respected by huggingface_hub/tqdm)
            _prev_pbar_env = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
            if quiet:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

            try:
                _hf_logger.setLevel(_logging.INFO)
                _hf_logger.addHandler(_handler)
                if quiet:
                    _hf_logger.propagate = False
                    _hf_logger.handlers = [_handler]  # keep only our buffer in quiet mode

                # Silence tqdm progress bars to stderr as an extra safety in quiet mode
                def _do_upload():
                    return upload_folder(
                        repo_id=repo_id,
                        repo_type="model",
                        folder_path=str(p),
                        revision=branch or DEFAULT_PUSH_BRANCH,
                        commit_message=commit_msg,
                        token=hf_token,
                        ignore_patterns=ignore_patterns,
                    )

                if quiet:
                    with open(os.devnull, "w") as _devnull:
                        with _contextlib.redirect_stderr(_devnull):
                            info = _do_upload()
                else:
                    info = _do_upload()
                hf_logs = getattr(_handler, "buf", None)
            finally:
                # Restore logger state
                try:
                    _hf_logger.removeHandler(_handler)
                except Exception:
                    pass
                try:
                    _hf_logger.setLevel(_old_level)
                    _hf_logger.propagate = _old_propagate
                    _hf_logger.handlers = _old_handlers
                except Exception:
                    pass
                # Restore env var
                try:
                    if quiet:
                        if _prev_pbar_env is None:
                            del os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]
                        else:
                            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = _prev_pbar_env
                except Exception:
                    pass
        except HfHubHTTPError as he:
            # In some hub versions, uploading to a non-existent branch raises here.
            # If --create was given, try to create the branch and retry once.
            msg = str(he)
            if create and ("Revision Not Found" in msg or "Invalid rev id" in msg):
                try:
                    api.create_branch(repo_id=repo_id, repo_type="model", branch=branch)
                    # Retry upload once
                    try:
                        info = upload_folder(
                            repo_id=repo_id,
                            repo_type="model",
                            folder_path=str(p),
                            revision=branch or DEFAULT_PUSH_BRANCH,
                            commit_message=commit_msg,
                            token=hf_token,
                            ignore_patterns=ignore_patterns,
                        )
                        hf_logs = hf_logs or []
                    except HfHubHTTPError as he2:
                        result["status"] = "error"
                        result["error"] = {"type": "upload_failed", "message": str(he2)}
                        return result
                except HfHubHTTPError as ce:
                    result["status"] = "error"
                    result["error"] = {"type": "branch_create_failed", "message": str(ce)}
                    return result
            else:
                result["status"] = "error"
                result["error"] = {
                    "type": "upload_failed",
                    "message": str(he),
                }
                return result
        except Exception as e:
            result["status"] = "error"
            result["error"] = {
                "type": "upload_failed",
                "message": str(e),
            }
            return result

        # 6) Success — extract details from CommitInfo (robust across hub versions)
        commit_id = None
        commit_url = None
        uploaded_count = None
        no_changes = None

        try:
            commit_id = getattr(info, "commit_id", None) or getattr(info, "oid", None)
            commit_url = getattr(info, "commit_url", None) or getattr(info, "html_url", None)

            # Try to compute number of committed files and a change summary
            change_summary = {"added": 0, "modified": 0, "deleted": 0}
            files_seq = getattr(info, "files", None) or getattr(info, "operations", None)
            if files_seq is not None:
                for f in files_seq:
                    # Infer operation in a version-agnostic way
                    op = None
                    # object attribute style
                    if hasattr(f, "operation"):
                        op = getattr(f, "operation")
                    elif hasattr(f, "op"):
                        op = getattr(f, "op")
                    # mapping/dict style
                    elif isinstance(f, dict):
                        op = f.get("operation") or f.get("op") or f.get("type")
                    # class name fallback
                    if op is None:
                        cls = f.__class__.__name__ if hasattr(f, "__class__") else ""
                        op = cls

                    op_s = str(op).lower()
                    if "add" in op_s or "+" in op_s:
                        change_summary["added"] += 1
                    elif "del" in op_s or "remove" in op_s or "-" in op_s:
                        change_summary["deleted"] += 1
                    elif "update" in op_s or "modify" in op_s or "mod" in op_s:
                        change_summary["modified"] += 1
                    else:
                        # treat unknown as modified
                        change_summary["modified"] += 1

                uploaded_count = sum(change_summary.values())
                result["data"]["change_summary"] = change_summary

            # Determine no-op (no changes)
            if commit_id in (None, ""):
                no_changes = True
            else:
                # Some hub versions may still create a commit even with no file changes; treat zero operations as no-op
                no_changes = (uploaded_count == 0) if uploaded_count is not None else False
        except Exception:
            # Be conservative if introspection fails
            pass

        # If hub logs indicate empty commit was skipped, prefer that signal
        try:
            if any(
                isinstance(m, str) and (
                    "Skipping to prevent empty commit" in m or "No files have been modified" in m
                )
                for m in (hf_logs or [])
            ):
                no_changes = True
                commit_id = None
                commit_url = None
                uploaded_count = 0
        except Exception:
            pass

        # Populate result fields
        result["data"]["commit_sha"] = commit_id
        result["data"]["commit_url"] = commit_url
        result["data"]["uploaded_files_count"] = uploaded_count if uploaded_count is not None else (0 if no_changes else None)
        result["data"]["no_changes"] = bool(no_changes) if no_changes is not None else (commit_id is None)
        result["data"]["created_repo"] = created_repo

        if hf_logs:
            result["data"]["hf_logs"] = hf_logs

        # Human-friendly message retained in JSON
        if result["data"]["no_changes"]:
            # Prefer hub-provided message if available
            hub_msg = None
            try:
                hub_msg = next(
                    (m for m in reversed(hf_logs or []) if isinstance(m, str) and ("Skipping" in m or "No files" in m)),
                    None,
                )
            except Exception:
                hub_msg = None
            result["data"]["message"] = hub_msg or "No files changed; skipped empty commit."
        elif uploaded_count is not None:
            cs = result["data"].get("change_summary") or {"added": 0, "modified": 0, "deleted": 0}
            result["data"]["message"] = f"Committed {uploaded_count} files (+{cs['added']} ~{cs['modified']} -{cs['deleted']})."
        else:
            result["data"]["message"] = "Commit created."

        # ADR-007 Response Matrix: Add APFS hint to push success message (Alpha only)
        try:
            cache_root = get_current_cache_root()
            if not _is_apfs_filesystem(cache_root):
                result["data"]["message"] += " Clone operations require APFS filesystem."
        except Exception as e:
            # Safe fallback - don't fail push if APFS check fails
            # Debug: Log the exception to understand what's failing
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"APFS warning check failed: {e}")
            pass

        return result

    except Exception as e:
        result["status"] = "error"
        result["error"] = {"type": "push_operation_failed", "message": str(e)}
        return result


def _is_lfs_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > 200:
            return False
        head = path.read_text(errors="ignore")[:200]
        return "version https://git-lfs.github.com/spec/v1" in head
    except Exception:
        return False


def _analyze_workspace(root: Path) -> Dict[str, Any]:
    """Rudimentary, content-oriented health check for a local workspace.

    Returns a JSON-serializable dict with summary and issues.
    """
    files: List[Path] = [p for p in root.rglob("*") if p.is_file()]
    total_bytes = 0
    for f in files:
        try:
            total_bytes += f.stat().st_size
        except Exception:
            pass

    # config.json
    config_path: Optional[Path] = None
    cfg_exists = False
    cfg_valid = False
    for candidate in (root / "config.json",):
        if candidate.exists() and candidate.is_file():
            config_path = candidate
            cfg_exists = True
            try:
                data = _json.loads(candidate.read_text(encoding="utf-8"))
                cfg_valid = isinstance(data, dict) and len(data) > 0
            except Exception:
                cfg_valid = False
            break

    # weights detection
    weights: List[Path] = []
    ggufs = list(root.rglob("*.gguf"))
    safes = list(root.rglob("*.safetensors"))
    bins = list(root.rglob("pytorch_model.bin"))
    # Exclude the index file from safetensors weights list
    safes = [s for s in safes if not s.name.endswith(".safetensors.index.json")]
    weights = ggufs + safes + bins

    # index-aware check
    index_files = list(root.rglob("*.safetensors.index.json"))
    index_info: Dict[str, Any] = {"has_index": bool(index_files), "missing": []}
    if index_files:
        try:
            idx_obj = _json.loads(index_files[0].read_text(encoding="utf-8"))
            # HF index has weight_map: {param_name: filename}
            weight_map = idx_obj.get("weight_map", {}) if isinstance(idx_obj, dict) else {}
            referenced = set(weight_map.values()) if isinstance(weight_map, dict) else set()
            for fname in sorted(referenced):
                p = root / fname
                if not p.exists() or p.stat().st_size == 0 or _is_lfs_pointer(p):
                    index_info["missing"].append(fname)
        except Exception:
            index_info["parse_error"] = True

    # pattern-based shards (model-xxxxx-of-yyyyy.safetensors)
    import re as _re

    shard_re = _re.compile(r"model-(\d{5})-of-(\d{5})\.safetensors$")
    pattern_files = []
    for s in safes:
        if shard_re.search(s.name):
            pattern_files.append(s)
    pattern_ok = None
    if pattern_files:
        try:
            xs = [s.name for s in pattern_files]
            ys = sorted(xs)
            last = shard_re.search(ys[-1])
            if last:
                total = int(last.group(2))
                present = set()
                for nm in ys:
                    m = shard_re.search(nm)
                    if m:
                        present.add(int(m.group(1)))
                pattern_ok = (len(present) == total)
        except Exception:
            pattern_ok = False

    # anomalies
    anomalies: List[Dict[str, Any]] = []
    if not cfg_exists:
        anomalies.append({"severity": "error", "code": "config_missing", "message": "config.json not found"})
    elif not cfg_valid:
        anomalies.append({"severity": "error", "code": "config_invalid_json", "message": "config.json invalid or empty"})

    # weight presence and sanity
    if not weights:
        anomalies.append({"severity": "error", "code": "no_weights_found", "message": "No weights (*.gguf/*.safetensors/pytorch_model.bin)"})
    else:
        # LFS or zero-size detection
        for w in weights:
            try:
                if w.stat().st_size == 0:
                    anomalies.append({"severity": "error", "code": "empty_weight_file", "message": f"Empty file: {w.name}", "path": str(w.relative_to(root))})
                elif _is_lfs_pointer(w):
                    anomalies.append({"severity": "error", "code": "lfs_pointer_detected", "message": f"LFS pointer: {w.name}", "path": str(w.relative_to(root))})
            except Exception:
                pass

    # index completeness if present
    if index_info.get("has_index"):
        if index_info.get("parse_error"):
            anomalies.append({"severity": "error", "code": "index_parse_error", "message": "model.safetensors.index.json parse error"})
        missing = index_info.get("missing") or []
        if missing:
            anomalies.append({"severity": "error", "code": "index_missing_shard", "message": f"Missing/invalid shards: {len(missing)}", "missing": missing})

    # partial/tmp markers
    for f in files:
        nm = f.name.lower()
        if ".partial" in nm or nm.endswith(".tmp") or "partial" in nm:
            anomalies.append({"severity": "warn", "code": "partial_marker", "message": f"Partial/tmp marker: {f.name}", "path": str(f.relative_to(root))})

    # Determine health: strictly require config valid and some non-empty non-LFS weights
    has_good_weight = True if weights else False
    if weights:
        has_good_weight = any(
            (w.stat().st_size > 0 and not _is_lfs_pointer(w))
            for w in weights
        )
    healthy = bool(cfg_valid and has_good_weight and not any(a["severity"] == "error" for a in anomalies if a["code"] not in {"config_missing", "config_invalid_json", "no_weights_found", "empty_weight_file", "lfs_pointer_detected", "index_parse_error", "index_missing_shard"}))
    # In practice, healthy becomes False if any error-level anomalies present or config/weights invalid.
    if any(a["severity"] == "error" for a in anomalies):
        healthy = False

    return {
        "files_count": len(files),
        "total_bytes": total_bytes,
        "config": {"exists": cfg_exists, "valid_json": cfg_valid, "path": str(config_path) if config_path else None},
        "weights": {
            "count": len(weights),
            "formats": sorted(list({w.suffix.lstrip('.') if w.suffix else 'bin' for w in weights})),
            "index": index_info,
            "pattern_complete": pattern_ok,
        },
        "anomalies": anomalies,
        "healthy": healthy,
    }


def _collect_local_files(root: Path, ignore_patterns: list[str]) -> list[str]:
    """Return a list of relative POSIX paths for files under root, honoring ignore patterns.

    This is a best-effort approximation of upload_folder's ignore behavior, using
    glob-like matching. It is sufficient for dry-run summaries.
    """
    from pathlib import PurePosixPath
    import fnmatch

    def ignored(rel: str) -> bool:
        p = PurePosixPath(rel)
        base = p.name
        for pat in ignore_patterns:
            try:
                # Normalize simple relative names (match basenames too)
                if pat == base or pat == rel:
                    return True
                # Try both PurePath.match and fnmatch as a fallback
                if p.match(pat) or fnmatch.fnmatch(rel, pat):
                    return True
            except Exception:
                # Be permissive on pattern errors
                if fnmatch.fnmatch(rel, pat):
                    return True
        return False

    files: list[str] = []
    for fp in root.rglob("*"):
        if fp.is_file():
            rel = fp.relative_to(root).as_posix()
            if not ignored(rel):
                files.append(rel)
    return files
