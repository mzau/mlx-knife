from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def humanize_size(num_bytes: Optional[int]) -> str:
    if not isinstance(num_bytes, int):
        return "-"
    n = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1000:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1000.0
    return f"{n:.1f}PB"


def fmt_hash7(h: Optional[str]) -> str:
    if not h:
        return "-"
    return h[:7]


def fmt_time(iso_utc_z: Optional[str]) -> str:
    if not iso_utc_z:
        return "-"
    try:
        # Expected like 2025-08-30T12:34:56Z (UTC)
        dt = datetime.strptime(iso_utc_z, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - dt
        seconds = int(delta.total_seconds())

        if seconds < 45:
            return "just now"
        if seconds < 90:
            return "1m ago"
        minutes = round(seconds / 60)
        if minutes < 45:
            return f"{minutes}m ago"
        if minutes < 90:
            return "1h ago"
        hours = round(minutes / 60)
        if hours < 24:
            return f"{hours}h ago"
        if hours < 36:
            return "1d ago"
        days = round(hours / 24)
        if days < 30:
            return f"{days}d ago"
        # For older entries, fall back to a compact date
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return iso_utc_z


def _table(rows: List[List[str]], headers: List[str]) -> str:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
            else:
                widths.append(len(cell))

    def fmt_row(cols: List[str]) -> str:
        return " | ".join(col.ljust(widths[i]) for i, col in enumerate(cols))

    lines = []
    lines.append(fmt_row(headers))
    lines.append("-+-".join("-" * w for w in widths))
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)


def render_list(data: Dict[str, Any], show_health: bool, show_all: bool, verbose: bool) -> str:
    models: List[Dict[str, Any]] = data.get("data", {}).get("models", [])
    compact = (not show_all) and (not verbose)
    if compact:
        headers = ["Name", "Hash", "Size", "Modified", "Type"]
    else:
        headers = ["Name", "Hash", "Size", "Modified", "Framework", "Type"]
    if show_health:
        headers.append("Health")

    # Human filter:
    # - --all: show everything
    # - default: show only MLX chat models (safer for run/server selection)
    # - --verbose (without --all): show all MLX models (chat + base)
    filtered: List[Dict[str, Any]] = []
    for m in models:
        fw = str(m.get("framework", "")).upper()
        typ = str(m.get("model_type", "")).lower()
        if show_all:
            filtered.append(m)
        else:
            if fw != "MLX":
                continue
            if verbose:
                # In verbose mode, show all MLX models
                filtered.append(m)
            else:
                # Default compact mode: only MLX chat
                if typ == "chat":
                    filtered.append(m)

    rows: List[List[str]] = []
    for m in filtered:
        name = str(m.get("name", "-"))
        if not verbose and name.startswith("mlx-community/"):
            # Compact name without the default org prefix
            name = name.split("/", 1)[1]
        if compact:
            row = [
                name,
                fmt_hash7(m.get("hash")),
                humanize_size(m.get("size_bytes")),
                fmt_time(m.get("last_modified")),
                str(m.get("model_type", "-")),
            ]
        else:
            row = [
                name,
                fmt_hash7(m.get("hash")),
                humanize_size(m.get("size_bytes")),
                fmt_time(m.get("last_modified")),
                str(m.get("framework", "-")),
                str(m.get("model_type", "-")),
            ]
        if show_health:
            row.append(str(m.get("health", "-")))
        rows.append(row)

    # Note: show_all/verbose are reserved for future detail; table remains deterministic
    return _table(rows, headers)


def render_health(data: Dict[str, Any]) -> str:
    d = data.get("data", {})
    summary = d.get("summary", {})
    total = summary.get("total", 0)
    healthy_count = summary.get("healthy_count", 0)
    unhealthy_count = summary.get("unhealthy_count", 0)

    lines = [f"Summary: total {total}, healthy {healthy_count}, unhealthy {unhealthy_count}"]
    for entry in d.get("healthy", []):
        lines.append(f"healthy   {entry.get('name','-')} — {entry.get('reason','')}".rstrip())
    for entry in d.get("unhealthy", []):
        lines.append(f"unhealthy {entry.get('name','-')} — {entry.get('reason','')}".rstrip())
    return "\n".join(lines)


def render_show(data: Dict[str, Any]) -> str:
    d = data.get("data", {})
    model = d.get("model", {})
    name = model.get("name", "-")
    h7 = fmt_hash7(model.get("hash"))
    header = f"Model: {name}{('@'+h7) if h7 != '-' else ''}"
    details = [
        f"Framework: {model.get('framework','-')}",
        f"Type: {model.get('model_type','-')}",
        f"Size: {humanize_size(model.get('size_bytes'))}",
        f"Modified: {fmt_time(model.get('last_modified'))}",
        f"Health: {model.get('health','-')}",
    ]

    # Optional sections
    out: List[str] = [header, *details]
    if "files" in d and isinstance(d["files"], list):
        out.append("")
        out.append("Files:")
        for f in d["files"]:
            out.append(f"  - {f.get('name','?')} ({f.get('type','other')}, {f.get('size','?')})")
    elif "config" in d and isinstance(d["config"], dict):
        out.append("")
        out.append("Config:")
        for k, v in d["config"].items():
            out.append(f"  {k}: {v}")
    elif d.get("metadata"):
        out.append("")
        out.append("Metadata:")
        for k, v in d["metadata"].items():
            out.append(f"  {k}: {v}")
    return "\n".join(out)


def render_pull(data: Dict[str, Any]) -> str:
    d = data.get("data", {})
    status = data.get("status", "error")
    model = d.get("model", "-")
    msg = d.get("message", "")
    if status == "success":
        return f"pull: {model} — {msg}".rstrip()
    err = data.get("error", {})
    return f"pull: {model} — {err.get('message', msg)}".rstrip()


def render_rm(data: Dict[str, Any]) -> str:
    d = data.get("data", {})
    status = data.get("status", "error")
    model = d.get("model", "-")
    action = d.get("action", "-")
    msg = d.get("message", "")
    if status == "success":
        return f"rm: {model} — {action}: {msg}".rstrip()
    err = data.get("error", {})
    return f"rm: {model} — {err.get('message', msg)}".rstrip()


def render_clone(data: Dict[str, Any], quiet: bool = False) -> str:
    """Render clone operation result for human output."""
    d = data.get("data", {})
    status = data.get("status", "error")
    model = d.get("model", "-")
    target_dir = d.get("target_dir", "-")
    msg = d.get("message", "")
    clone_status = d.get("clone_status", "unknown")

    if status == "success":
        if quiet:
            return f"clone: {model} → {target_dir}"

        # Show additional info for successful clone
        cache_cleanup = d.get("cache_cleanup", False)
        health_check = d.get("health_check", True)

        status_parts = []
        if health_check:
            status_parts.append("✓ health")
        if cache_cleanup:
            status_parts.append("✓ cleanup")

        status_info = f" ({', '.join(status_parts)})" if status_parts else ""
        return f"clone: {model} → {target_dir}{status_info} — {msg}".rstrip()

    # Error case
    err = data.get("error", {})
    error_msg = err.get("message", msg)

    # Show the specific phase where it failed
    if clone_status in ["pull_failed", "health_check_failed", "copy_failed", "cache_not_found"]:
        phase = clone_status.replace("_", " ")
        return f"clone: {model} → {target_dir} — {phase}: {error_msg}".rstrip()

    return f"clone: {model} → {target_dir} — {error_msg}".rstrip()


def render_push(data: Dict[str, Any], verbose: bool = False) -> str:
    d = data.get("data", {})
    status = data.get("status", "error")
    repo = d.get("repo_id", "-")
    branch = d.get("branch", "-")
    cs = d.get("commit_sha")
    h7 = cs[:7] if isinstance(cs, str) and len(cs) >= 7 else "-"
    prefix = "push (experimental):"
    # Dry-run handling
    if d.get("dry_run"):
        if d.get("no_changes") is True:
            return f"{prefix} {repo}@{branch} — dry-run: no changes".rstrip()
        summ = d.get("dry_run_summary") or d.get("change_summary") or {}
        added = summ.get("added")
        modified = summ.get("modified")
        deleted = summ.get("deleted")
        mod_part = str(modified) if isinstance(modified, int) else "?"
        line = f"{prefix} {repo}@{branch} — dry-run: +{added or 0} ~{mod_part} -{deleted or 0}"
        if verbose and (d.get("would_create_repo") or d.get("would_create_branch")):
            hints = []
            if d.get("would_create_repo"):
                hints.append("create repo")
            if d.get("would_create_branch"):
                hints.append("create branch")
            if hints:
                line = f"{line} ({', '.join(hints)})"
        return line.rstrip()
    if status == "success":
        if d.get("no_changes"):
            msg = d.get("message")
            base = f"{prefix} {repo}@{branch} — no changes"
            if verbose and isinstance(msg, str) and msg and "no changes" not in msg.lower():
                return f"{base} ({msg})".rstrip()
            return base.rstrip()
        # If we have a commit, show it and include a compact summary when available
        if isinstance(cs, str) and cs:
            summary = d.get("change_summary") or {}
            added = summary.get("added")
            modified = summary.get("modified")
            deleted = summary.get("deleted")
            if all(isinstance(x, int) for x in (added, modified, deleted)):
                line = f"{prefix} {repo}@{branch} — commit {h7} (+{added} ~{modified} -{deleted})"
            else:
                line = f"{prefix} {repo}@{branch} — commit {h7}"

            # Workaround: Show important warnings from message (e.g., APFS warning)
            msg = d.get("message", "")
            if isinstance(msg, str) and "Clone operations require APFS filesystem" in msg:
                line = f"{line} (Clone operations require APFS filesystem)"

            if verbose:
                url = d.get("commit_url")
                if isinstance(url, str) and url:
                    line = f"{line} <{url}>"
            return line.rstrip()
        # Fallback
        msg = d.get("message")
        if isinstance(msg, str) and msg:
            return f"{prefix} {repo}@{branch} — {msg}".rstrip()
        return f"{prefix} {repo}@{branch} — done".rstrip()
    err = data.get("error", {})
    msg = err.get("message", "")
    return f"{prefix} {repo}@{branch} — {msg}".rstrip()
