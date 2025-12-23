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


def _table(rows: List[List[str]], headers: List[str], max_col_width: Optional[int] = None) -> str:
    """
    Build a table with optional column width limit for last column.

    Args:
        rows: Table rows
        headers: Column headers
        max_col_width: If set, limits last column to this width (wraps text to new lines)
    """
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
            else:
                widths.append(len(cell))

    # Apply max width limit to last column if specified
    if max_col_width and len(widths) > 0:
        widths[-1] = min(widths[-1], max_col_width)

    def fmt_row(cols: List[str]) -> str:
        return " | ".join(col.ljust(widths[i]) for i, col in enumerate(cols))

    def wrap_cell(text: str, width: int) -> List[str]:
        """Wrap text to width, breaking at word boundaries."""
        if len(text) <= width:
            return [text]
        words = text.split()
        lines = []
        current = []
        current_len = 0
        for word in words:
            word_len = len(word)
            if current and current_len + 1 + word_len > width:
                lines.append(" ".join(current))
                current = [word]
                current_len = word_len
            else:
                current.append(word)
                current_len += (1 if current_len > 0 else 0) + word_len
        if current:
            lines.append(" ".join(current))
        return lines

    lines = []
    lines.append(fmt_row(headers))
    lines.append("-+-".join("-" * w for w in widths))

    for r in rows:
        # Check if last column needs wrapping
        if max_col_width and len(r) > 0 and len(r[-1]) > max_col_width:
            wrapped_lines = wrap_cell(r[-1], max_col_width)
            # First line with all columns
            first_row = r[:-1] + [wrapped_lines[0]]
            lines.append(fmt_row(first_row))
            # Additional lines with empty cells except last column
            for wrapped_line in wrapped_lines[1:]:
                continuation_row = [""] * (len(r) - 1) + [wrapped_line]
                lines.append(fmt_row(continuation_row))
        else:
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
        if verbose:
            # Verbose mode: split health into Integrity + Runtime + Reason columns
            headers.extend(["Integrity", "Runtime", "Reason"])
        else:
            # Compact mode: single Health column
            headers.append("Health")

    # Human filter:
    # - --all: show everything (no filter)
    # - default/verbose: only healthy + runtime_compatible (runnable models)
    # Same filter as Server /v1/models - single source of truth via build_model_object
    filtered: List[Dict[str, Any]] = []
    for m in models:
        if show_all:
            filtered.append(m)
        else:
            # Filter: healthy AND runtime_compatible
            if m.get("health") != "healthy":
                continue
            if not m.get("runtime_compatible"):
                continue
            filtered.append(m)

    rows: List[List[str]] = []
    for m in filtered:
        name = str(m.get("name", "-"))
        if not verbose and name.startswith("mlx-community/"):
            # Compact name without the default org prefix
            name = name.split("/", 1)[1]
        caps = set(m.get("capabilities") or [])
        type_label = str(m.get("model_type", "-"))
        if "vision" in caps and type_label != "-":
            type_label = f"{type_label}+vision"
        if compact:
            row = [
                name,
                fmt_hash7(m.get("hash")),
                humanize_size(m.get("size_bytes")),
                fmt_time(m.get("last_modified")),
                type_label,
            ]
        else:
            row = [
                name,
                fmt_hash7(m.get("hash")),
                humanize_size(m.get("size_bytes")),
                fmt_time(m.get("last_modified")),
                str(m.get("framework", "-")),
                type_label,
            ]
        if show_health:
            if verbose:
                # Verbose mode: Integrity | Runtime | Reason columns
                health = m.get("health", "unknown")
                runtime_compatible = m.get("runtime_compatible")
                reason = m.get("reason", "")

                # Integrity column
                integrity = "healthy" if health == "healthy" else "unhealthy" if health == "unhealthy" else "-"

                # Runtime column (only meaningful if integrity is healthy)
                if health == "healthy" and runtime_compatible is not None:
                    runtime = "yes" if runtime_compatible else "no"
                else:
                    runtime = "-"

                # Reason column (truncate to 60 chars)
                reason_str = str(reason) if reason else "-"
                if len(reason_str) > 60:
                    reason_str = reason_str[:57] + "..."

                row.extend([integrity, runtime, reason_str])
            else:
                # Compact mode: single Health column (healthy/healthy*/unhealthy)
                health = m.get("health", "unknown")
                runtime_compatible = m.get("runtime_compatible")

                if health == "healthy":
                    if runtime_compatible is True:
                        health_str = "healthy"
                    elif runtime_compatible is False:
                        health_str = "healthy*"
                    else:
                        # No runtime check performed
                        health_str = "healthy"
                elif health == "unhealthy":
                    health_str = "unhealthy"
                else:
                    health_str = "-"

                row.append(health_str)
        rows.append(row)

    # Note: show_all/verbose are reserved for future detail; table remains deterministic
    # Apply 26 char limit to Reason column in verbose mode
    max_col_width = 26 if (show_health and verbose) else None
    return _table(rows, headers, max_col_width=max_col_width)


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

    # Build health status string
    health = model.get('health', '-')
    runtime_compatible = model.get('runtime_compatible')
    if health == 'healthy' and runtime_compatible is True:
        health_str = 'healthy'
    elif health == 'healthy' and runtime_compatible is False:
        health_str = 'healthy (files OK, runtime incompatible)'
    else:
        health_str = health

    details = [
        f"Framework: {model.get('framework','-')}",
        f"Type: {model.get('model_type','-')}",
        f"Size: {humanize_size(model.get('size_bytes'))}",
        f"Modified: {fmt_time(model.get('last_modified'))}",
        f"Health: {health_str}",
    ]

    # Add reason if present
    reason = model.get('reason')
    if reason:
        details.append(f"Reason: {reason}")

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
