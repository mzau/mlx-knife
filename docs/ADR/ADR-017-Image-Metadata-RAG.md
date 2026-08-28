# ADR-017: Image Metadata Extraction (EXIF)

**Status:** Implemented (Phase 1)
**Date:** 2025-12-15
**Implementation:** 2.0.4-beta.1
**Related ADRs:** ADR-012 (Vision Support)

---

## Context

Vision models (ADR-012) generate text descriptions of images, but lack contextual metadata (GPS coordinates, capture time, camera info) that would enable photo organization and advanced workflows.

**Problem:**
```bash
mlxk run llava "describe" --image vacation/photo1.jpg vacation/photo2.jpg

# Output: "Image 1 shows a beach. Image 2 shows mountains."
# → Which file is which?
# → Where/when was it taken?
# → No metadata for filtering or organization
```

**User needs:**
1. Track original filenames (which description belongs to which file)
2. Access EXIF metadata (GPS, DateTime, Camera) for organization
3. Privacy control (GPS data is sensitive)

---

## Decision

**Extract EXIF metadata during vision runs, with user control.**

### Implementation (2.0.4-beta.1)

**Feature Flag:**
```bash
# Enabled by default
MLXK2_EXIF_METADATA=1 mlxk run llava --image *.jpg "describe"

# Disable for privacy
MLXK2_EXIF_METADATA=0 mlxk run llava --image *.jpg "describe"
```

**Output Format:**
Collapsible HTML table (`<details>` + markdown table) appended to vision response.

**Data Extracted:**
- Original filename (image ID mapping)
- GPS coordinates (lat/lon, if available)
  - Precision: 4 decimal places (~11m accuracy)
  - Rationale: Street-level precision for text models without excessive privacy exposure
- DateTime (ISO 8601 format)
  - **Primary source:** GPS-Timestamp (EXIF Tag 7+29: GPSTimeStamp + GPSDateStamp)
    - Precise UTC timestamp from satellite
    - Cannot be misconfigured (unlike camera clock)
  - **Fallback:** EXIF Tag 36867 (DateTimeOriginal) if no GPS timestamp
  - **Never used:** Tag 306 (DateTime) - gets updated by image editors to modification time
  - **Display format:** Date only (YYYY-MM-DD), time component not shown in Phase 1
  - **Timezone:** Always UTC for consistency (avoids Europe/Asia timezone ambiguity)
  - **Note:** File modification timestamps are separate from EXIF and not extracted
- Camera model (device info)

**Privacy Controls:**
- Default: **Enabled** (opt-out design — see §Why Default Enabled below)
- Opt-out: `MLXK2_EXIF_METADATA=0` (disables extraction)
- No logging: EXIF data never logged to server logs

**User Documentation:** See README.md "Multi-Modal Support → Vision → Metadata Output Format" for complete output examples.

---

## Rationale

### Why EXIF Extraction?

Enables photo organization:
- **Filename mapping:** Identify which description → which file
- **Temporal queries:** Organize by date/trip/season
- **Spatial context:** GPS-based workflows (see `examples/photo-rag`)

### Why Collapsible HTML Table?

**Alternatives considered:**
1. **Inline metadata:** Clutters output, hard to parse
2. **JSON sidecar:** Requires two files, coordination overhead
3. **Structured JSON mode:** Breaks streaming, requires API changes
4. **Collapsible table (chosen):** Non-intrusive, works everywhere

**Benefits:**
- Collapsed by default (no clutter)
- Works in terminals + web UIs (universal markdown)
- Copy/paste friendly (no separate files)
- Human-readable (visual icons: 📍 📅 📷)

### Why Default Enabled (Opt-Out)?

**Trade-off: Usability vs. Privacy**

Most users want metadata:
- Identifies which file → which description
- Enables date/location-based organization
- Travel photos benefit from GPS context

**Critical: Web client compatibility**
- Server API NOT extended (no `extract_exif` request parameter)
- Web clients (nChat, etc.) cannot control EXIF extraction
- **Default ON:** Works automatically for all clients without API changes
- **Default OFF would break:** Web clients couldn't get metadata

Privacy-conscious users can disable:
- Simple env var (`MLXK2_EXIF_METADATA=0`)
- Server-side control (no client changes needed)

**Alternative considered:** Opt-in (default disabled)
- **Rejected:** Web clients couldn't enable without API extension
- **Better:** Enable by default, document privacy controls

**Performance:** ~20ms per image (negligible)

---

## Privacy & Security

**EXIF data is sensitive:**
- **GPS:** Home/work locations → stalking/tracking risk
- **DateTime patterns:** Travel schedules → security risk
- **Camera model:** Device fingerprinting

**Mitigation:**
1. Opt-out available (`MLXK2_EXIF_METADATA=0`)
2. No logging (EXIF never in server logs)
3. README.md documents privacy controls prominently
4. Collapsible by default (user must expand)

**Best practice:**
```bash
# Strip EXIF before sharing publicly
exiftool -all= photo.jpg
```

---

## What's NOT Included

**Deliberate non-decisions:**

- **Structured JSON output:** Examples show how (`examples/photo-rag`), not in core
- **Server API extensions:** HTML table sufficient for current use cases
- **Reverse geocoding:** Requires external service (out of scope)

**Rationale:** Examples demonstrate advanced patterns without committing to stable API surface. Users can adapt to their needs.

---

## Future Considerations

**Phase 2: Extended EXIF (if demand emerges):**
- **Time display:** Show time component in Date column (currently date-only)
- **Additional GPS data:** Altitude, Direction, Positioning Error
- **CLI flag:** `--extended-exif` or `MLXK2_EXIF_EXTENDED=1`

**Other potential enhancements:**
- JSON output mode (API stability cost)
- Server API: `mlxk_options.extract_exif` in request
- Reverse geocoding (dependency cost: timezonefinder, pytz)

**Current approach:** Keep core minimal (date-only, basic GPS), examples show extension patterns.
