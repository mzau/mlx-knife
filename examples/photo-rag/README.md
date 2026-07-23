# Photo RAG — private photo library, made searchable

Describe a personal photo library with a vision model, then search it by text,
by capture date, and by location — without any private data entering the
project tree.

**Status:** 📋 Planned — target **2.0.8**
**Runnable against:** not yet; this README is the design record, not code
**Requires (planned):** a vision-capable model in your local cache; a running
mlx-knife server; macOS system tooling for HEIC/RAW
**Run:** not yet

> This document records the **decisions**, not the implementation. The scripts
> land in 2.0.8. Their job there is as much to harden mlx-knife as to be useful:
> a real multi-day run over tens of thousands of photos is a load test no unit
> test provides, and it is the groundwork for image embeddings.

---

## The use case

You have tens of thousands of personal photos, spread over a directory tree that
grew for decades and follows no single convention. You want to find things in it:
by what is *in* the picture, by when it was taken, by where it was taken.

This is a **consumer** of mlx-knife — an example of using it from the outside. It
is not part of mlx-knife, it requires no change to it, and nothing here is a
promise about what mlx-knife will do.

---

## Decisions

### Private data never enters the tree

Input and output both live outside the project, and both are named by environment
variable — `PHOTO_VAULT` for the library, `PHOTO_CATALOG` for everything produced
from it. **Neither has a default.** A missing variable is an error, not a fallback
to some directory inside the repo. The point is that there is nothing private *in*
the tree to protect in the first place.

Capture locations are sensitive. Coordinates are not sent anywhere, and not folded
into searchable text, unless you ask for it: once embedded into a vector, a
coordinate cannot be removed again without rebuilding everything.

### It talks HTTP to the model, not the command line

Captioning runs against the OpenAI-compatible server endpoint rather than by
invoking the CLI per photo. That is what keeps the model loaded between photos —
the difference decides whether a large library is hours or weeks of work.

The same decision is what makes the example portable later: a tool that speaks
OpenAI-compatible HTTP moves to a multi-node backend by pointing at a different
base URL. No backend switch, no second code path — that switch would be exactly
the change the portability claim forbids.

### One photo per request

Sending several images in one request is measurably faster and produces
**worse** results: the descriptions blend, and details of individual pictures are
lost or invented. This is verified against the fixture shipped with the project
(`tests_2.0/assets/geo-test/`), where a known subject is described correctly one
at a time and incorrectly in a batch. Throughput is not worth a catalog you
cannot trust, so photos are described one at a time.

This also marks where image embeddings will help and why. The blending happens
because the model *generates* across several pictures at once. An embedding does
not generate; it is defined per image. Grouping therefore belongs in vector
space, after the fact — not in the prompt.

### A long-running job, not a session

Tens of thousands of photos is not interactive work; it runs for days. So it is
built as a batch that can be interrupted at any moment — deliberately or by a
crash — and picked up again without redoing finished work or losing what was in
flight. Progress is readable while it runs.

**New photos can be added at any time.** Adding is the same operation as
resuming: a photo is identified by its *content*, not by its path or timestamp,
so a later run processes only what is genuinely new. Renaming, re-sorting or
re-importing does not cause re-work, and an interruption today and an addition
three months from now are the same case to the job.

### A written record, and a view derived from it

The batch appends to a log that is the single source of truth. The browsable
catalog a backend or a small frontend queries is *derived* from that log and can
be discarded and rebuilt at any time. Keeping the two apart is what makes the job
safe to kill: the expensive work is never inside the thing being rewritten.

### Formats

JPEG and the common web formats are read directly. HEIC (iPhone) and RAW
(including Sony ARW) go through macOS system tooling, which is present on every
supported machine — **no new dependency**, and capture metadata survives the
conversion. Anything that cannot be read is skipped loudly, never silently.

Where a camera wrote both a RAW file and a JPEG of the same shot, only one is
described by default. Describing the same picture twice costs hours and adds
nothing.

Richer metadata than the file itself carries — from cataloging applications, via
sidecar files next to the photo — is an **extension point**, not part of the
example. Because records are identified by content, such a source can be joined
in later without re-describing a single photo.

### What it will and will not do for throughput

A single node answers requests one at a time. Parallelism comes from putting more
nodes behind the same address, and the example is built so that this needs no
change beyond the address. Until then, that is a seam, not a claim.

### Extending it

The first cut searches the descriptions as text and by metadata. An image
embedding model is an **additive later stage**, not a precondition — it replaces
how vectors are produced, not how they are queried. Grouping a set of photos to
see what they have in common (for instance, everything within a radius of a
point) is part of the design from the start and is unaffected by that swap.

---

## Privacy notes

Photo metadata contains capture times and, often, coordinates. Before sharing any
photo — or building any sample set — strip it. `exiftool` (optional, external) is
the usual tool for inspecting and removing it.

mlx-knife's own metadata extraction can be turned off entirely:

```bash
export MLXK2_EXIF_METADATA=0
```

**Never commit photos with location data to git.**

---

## See also

- [`../rag-server/`](../rag-server/) — embedding and vector search over text
- [`../pipes/`](../pipes/) — vision → text pipeline building blocks
- Vision and embeddings sections in the project [README](../../README.md)

---

## License

Same as MLX Knife (Apache 2.0).
