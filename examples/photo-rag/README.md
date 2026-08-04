# Photo RAG — a family album you can actually search

Describe every photograph in a private collection with a vision model, then find things
in it by what is *in* the picture, by when it was taken, and by where.

**Nothing leaves your machine.** The model runs here; the photographs are read here, the
descriptions are written here, and the search happens here. That is the whole reason this
is worth doing at all — a family album is exactly the kind of collection nobody wants to
upload to a service, and it is exactly the kind of collection that is unsearchable without
one. A local model closes that gap.

This is written for **one person's own collection**. Everything below assumes that.

**Status:** ✅ Runnable
**Runnable against:** mlx-knife **≥ 2.0.7**. Needs three surfaces, not a version: vision
through `mlxk serve` with base64 `image_url`; the `<!-- mlxk:filenames -->` metadata block
the server prepends to every vision answer; and `mlxk embed --batch` / `--query`
(experimental, alpha-gated — the scripts set the flag for their own subprocesses).
Verified on 2.0.7. **Nothing here requires a change to mlx-knife.**
**Requires:** a vision model you serve yourself; an embedding model in your cache; Python
`httpx pillow numpy`; macOS `/usr/bin/sips` for HEIC/RAW
**Run:** see *Try it in five minutes* — no photo library needed

> **How far this has been taken.** Everything measured below comes from the thirteen
> photographs published here, not from a large collection. The design is built for a
> multi-day run over tens of thousands — that is what the resume, quarantine and locking
> machinery is for — but that run has not happened yet. Treat the timings as a starting
> estimate and the long-run behaviour as designed-for rather than demonstrated.
>
> The **Decisions** section is the original design record and still holds unchanged.

---

## Try it in five minutes

You need no photographs of your own. This repository ships thirteen for exactly this
purpose — real pictures with real capture metadata, published so the example can be run,
taken apart and played with by anyone who clones it: nine JPEGs, and four HEIC files in a
subdirectory so the conversion branch is something you can watch happen rather than take on
trust.

**These are the only photographs that ever live in this repository, and your own never
join them.** They are a fixture: published deliberately, checked into git, and there to be
poked at. Your album is the other case entirely — it stays wherever it already is, outside
this tree, and is reached only by pointing `PHOTO_VAULT` at it. The two are never mixed,
and this section is about the first one. The next section is about yours.

```bash
# 1. one vision model, served by you, in its own shell
mlxk serve --model pixtral-12b-4bit --port 8000

# 2. back in the repository root
export PHOTO_VAULT="$PWD/tests_2.0/assets/geo-test"    # the photographs shipped here
export PHOTO_CATALOG="$(mktemp -d)"                    # scratch: nothing kept, nothing in the repo
cd examples/photo-rag

./photo-walk.py                     # what is there?           ~1 s
                                    #   -> 9 direct, 4 sips (HEIC via macOS)
./caption-photos.py --model pixtral-12b-4bit           # describe them  ~5 s each
./build-catalog.py --stats
./index-photos.py --model bge-small-en-v1.5-4bit

./photo-search.py "a boat at a pier"
./photo-search.py "a statue" --top-k 2
```

The last command prints this — an actual run, not an illustration:

```
[0.709] coll2_7.jpeg  2025-05-12T16:15:15  Apple iPhone 13 Pro  59.3283N 18.0644E
        /…/tests_2.0/assets/geo-test/coll2_7.jpeg
        The image depicts a bronze statue of a woman in a garden setting, positioned on
        a pedestal with a book. The statue is situated in front of an old building with…

[0.702] coll2_4.jpeg  2025-05-12T17:55:57  Apple iPhone 13 Pro  59.3250N 18.0738E
        /…/tests_2.0/assets/geo-test/coll2_4.jpeg
        The image depicts a bronze statue situated on a pedestal in an urban setting.
        The statue features multiple figures in dynamic poses, with one figure…
```

Open the files it names and see whether you agree. That is the whole point: the
descriptions come from the pixels, the dates and coordinates come from the files, and the
search combines them. Nobody told the model these were statues.

Worth trying once it works:

- `./photo-search.py "water" --since 2025-05-12` — text and metadata together.
- `./photo-search.py "boats" --near 59.33,18.07 --radius-km 3` — everything within three
  kilometres of a point.
- Delete `$PHOTO_CATALOG/catalog.jsonl` and re-run `./build-catalog.py`. Nothing is lost:
  the catalog is derived, the log is the original.
- Interrupt `./caption-photos.py` with Ctrl-C and start it again. It picks up where it
  stopped, because a photograph is identified by its content rather than its name.
- Point `PHOTO_VAULT` at a folder of your own — anywhere outside this repository — and
  repeat. HEIC from a phone works; so does raw, through macOS. Read *Your own album* first
  if that folder is one you care about.

---

## Your own album

Everything above used the photographs shipped in this repository. This section is about
yours, and the difference is not cosmetic: those are a published fixture inside a git tree,
yours are irreplaceable and live outside it. Nothing here ever writes into this repository,
and `PHOTO_VAULT` is how you say where your album actually is.

Read this once before pointing anything at a collection that cannot be replaced; every
warning in it was measured rather than imagined.

### What can and cannot go wrong

**Your collection is never written to.** Descriptions, catalog, index, converted uploads —
everything produced goes to `$PHOTO_CATALOG` or a temporary directory. The photographs are
only ever read: nothing is moved, renamed, re-tagged or deleted.

That used to be a claim and is now a check. A `PHOTO_CATALOG` equal to, or inside,
`PHOTO_VAULT` is **refused before any file is created**, with no override, for three
reasons — and the third is why there is no override:

1. `vault.json`, `log/`, `tmp/` and `prepared/` would be written among your photographs.
2. The cached uploads are JPEGs, a format this tool reads, so the next run would find its
   own output as new photographs — and every run after that would find more.
3. The batch clears its own `tmp/` at startup, because that scratch still carries full
   EXIF. Pointed at a real directory named `tmp`, that is **deletion**. Guarded twice now:
   the containment check, and a marker so the sweep only empties a directory this tool
   created.

So the risks that remain are not about your photographs. They are:

- **days spent on descriptions you do not trust** — a wrong model, a wrong prompt, or a
  resolution that makes the model invent text rather than admit it cannot read it;
- **a catalog that repeats private content in plain text** — descriptions transcribe what
  the model can read, so a photographed document or screen becomes searchable prose;
- **a run that dies at hour thirty and has to start over** — which it does not, but that is
  worth proving to yourself rather than believing.

Every one of those is cheaper to find in the first ten minutes than on day three. Hence the
order below.

### Rehearse before you commit days

Four steps, each answering one question. The first three cost no GPU time at all.

```bash
export PHOTO_VAULT="/Volumes/Photos/Album"          # wherever your pictures live
export PHOTO_CATALOG="$HOME/photo-rag/album"        # local disk, and not the same volume
cd examples/photo-rag
```

Neither variable has a default and neither can be passed as a flag: a missing one is an
error, never a quiet fallback to somewhere inside this repository.

**1 — How big is this, really?** Minutes, no model, no decoding.

```bash
./photo-walk.py
```

Counts, formats, duplicates, raw+JPEG pairs, and an estimate of the captioning time. On a
network share the walk itself takes a while; that is the price of looking at every file
once, and it is paid before any GPU time is.

**2 — Can it actually read my files?** No model contacted; every photograph decoded,
converted and its metadata read.

```bash
./caption-photos.py --model pixtral-12b-4bit --dry-run
```

This is where you find out that a thousand files are a format you did not expect, or that a
panorama is large enough to be a problem — before spending a day on it rather than after. It
also fills the conversion cache, so the real run does not repeat the work.

**3 — Are the descriptions any good?** Twenty photographs, about two minutes.

```bash
./caption-photos.py --model pixtral-12b-4bit --limit 20
jq -r 'select(.type=="result" and .ok) | .text' "$PHOTO_CATALOG/log/captions.jsonl" | head -5
```

**Read them.** This is the single most valuable minute in the whole procedure. If the
descriptions are vague, wrong, or about the wrong thing, change the prompt
(`--prompt-file`) or the model now — not after four days. Try a search too:
`./build-catalog.py && ./index-photos.py --model bge-small-en-v1.5-4bit` and ask for
something you know is in those twenty.

**4 — The real run.** Only now.

```bash
./caption-photos.py --model pixtral-12b-4bit
```

Roughly 4.7 s per photograph — about eleven hours per ten thousand. The twenty already
described are not repeated. Ctrl-C whenever you like; starting it again continues where it
stopped. Nothing done in steps 1–3 has to be undone: the same catalog carries straight
through.

### Where to put the catalog

**Keep it on a different volume from the photographs**, not merely outside them. A catalog
beside your album on the same network share passes the containment check and still shares
its fate: unmounting or filling that share loses the descriptions and the pictures together,
and days of work with them. The scripts warn when they see this, and refuse a catalog on
network storage outright unless you pass `--allow-nonlocal-catalog` — over SMB one round
trip per log line is the bottleneck of the whole run, and the lock that stops two runs doing
the same work is only advisory there.

**Room to work:** the cached uploads are around 75 KB each — roughly 2 GB for thirty
thousand photographs — and the log about 21 MB. Both live under `$PHOTO_CATALOG`.

### Worth knowing once

- **Symlinks are not followed.** A symlinked file among your photographs is skipped and
  counted as such. The boundary you drew by naming `PHOTO_VAULT` should hold for files, not
  just directories: a link is otherwise read, uploaded and described while its record shows
  an unremarkable album-relative path. If your collection genuinely uses links to organise
  itself, pass `--follow-symlinks` — and know you have widened what reaches the model.
- **Descriptions transcribe what the model can read.** A photographed document,
  prescription, screen or envelope becomes searchable plaintext in the catalog and an
  irreversible part of a vector. `$PHOTO_CATALOG` deserves the same care as the album.
- **Coordinates never enter the embedded text**, and records store album-relative paths —
  the absolute root lives in `vault.json` and nowhere else. What is printed on your own
  terminal is not what is stored.
- **Being killed is safe.** Ctrl-C finishes the picture in flight. A hard kill leaves an
  attempt without a result, which the next run counts; three of those on one photograph
  quarantine it rather than letting a single file stall a multi-day run forever.
- **The log is the only thing expensive to lose.** Catalog and index are derived and
  rebuild in minutes. If you back up one file, back up `log/captions.jsonl`.
- **One run at a time.** The lock is never broken automatically. If a run died and left one
  behind, `--break-lock` is explicit on purpose.
- **Where a camera wrote both a raw file and a JPEG** of one shot, only one is described.
  `--prefer-raw` picks the other. Byte-identical duplicates are described once, however they
  are named or wherever they sit.
- **Photographs with no coordinates are normal** — most cameras have no receiver. Date and
  camera are the axes you can rely on; place is a bonus where it exists.

### Starting over

Delete the catalog directory, or point `PHOTO_CATALOG` somewhere new. Your photographs are
untouched either way, so there is nothing to restore — only captioning time to spend again.
To keep the descriptions and rebuild everything downstream, delete `catalog.jsonl` and
`index.jsonl` instead and re-run the two derived steps.

---

## The scripts

You start the server; these scripts never do. One they started would die with them and take
a multi-day run along, so they fail loudly instead.

| Script | What it is for |
|---|---|
| `photo-walk.py` | Inventory: counts, formats, duplicates. No model, no decoding. |
| `caption-photos.py` | The batch. Lock, resume, quarantine, canary. `--dry-run` checks the format branch over the whole collection without touching the GPU. |
| `build-catalog.py` | Derives the catalog from the log. Discardable and rebuildable. |
| `index-photos.py` | Embeds the descriptions via `mlxk embed --batch`. |
| `photo-search.py` | Query by description, date, place, camera. |
| `geo-test-run.py` | The self-verifying run over the shipped photographs. |
| `photo_lib.py` | Shared machinery. Not a CLI. |

---

## What it checks about itself

`./geo-test-run.py` runs the whole pipeline over the photographs shipped here and prints a
PASS/FAIL table with a verdict. Every row is graded against ground truth that exists
independently of the tool — geometry measured from the produced bytes, strings legibly
painted on a photograph, predicates fed inputs they must reject. **Nothing in it asks a
model or a script whether it thinks it did well.**

The first row is the one that makes the rest mean anything: it feeds every predicate the
inputs it is supposed to refuse. Without it the table would only prove that its checks
*fire*, never that they *discriminate*. A further group builds its own miniature album from
generated images and checks the boundaries — that a catalog inside the album is refused, that
a pre-existing `tmp/` is not emptied, that a symlinked file is skipped, that a moved
photograph is relocated rather than described again — each with the opposite case as a
control, because "it refused" alone would also be true of a tool that refuses everything.

Rows are mandatory or optional. If a mandatory row has to skip, the verdict is
`INCONCLUSIVE` and the exit code is non-zero — a green table with skips must not read like a
pass. Two rows are measurements rather than assertions and never fail: the perceptual-distance
calibration and the resolution ladder.

Run the acceptance with the server's EXIF handling at its **default**, not disabled. The
uploads carry no EXIF either way, but only at the default can the table *observe* that the
server saw nothing while the client saw everything — which turns the central privacy claim
from an assumption into a measurement.

---

## What it costs

Against `pixtral-12b-4bit` with the model resident, over HTTP:

| Longest edge | Seconds per photograph | Extrapolated to 30,000 |
|---|---|---|
| 512 px | **4.7** (3.8 – 5.9) | ~39 h |
| 1024 px | 14.4 | ~120 h |

Resolution dominates the cost. It is emphatically **not** the lever for reading text, and
this is the one place where the obvious intuition is not merely wrong but dangerous.

Measured twice. On a photograph with *short, large* lettering — a ship's nameplate and a
clock — 1024 px recovered nothing 512 px had missed and turned one omission into a confident
misreading. On a photograph of an *information board covered in small print*, tested with a
prompt that asked only for transcription against nine known strings:

| Longest edge | Seconds | Strings read | How it failed |
|---|---|---|---|
| 512 px | 4.4 | 2 of 9 | read both headlines, then stopped |
| 768 – 1536 px | 18 – 29 | 2 of 9 | read the headlines, **invented the body text** |
| 2048 px | 29 | 0 of 9 | collapsed into one fabricated sentence, repeated |

At 512 px the model is honest about what it cannot resolve. Above that it keeps reading the
headings correctly and fills in the small print underneath with fluent, plausible prose that
is not on the sign at all.

That particular board happens to be bilingual, alternating between two languages section by
section, which is the only reason the failure was legible at all: the model reproduced the
*alternation* and then wrote body text in the wrong one of the two — the language the sign
does not use in that block. It read the structure and invented the substance. That property
belongs to one photograph and is worth nothing as an assumption; it was a lucky diagnostic,
not a rule about signs.

For a searchable album this is the worst possible failure: the text is well-formed, in a
plausible language, about the right subject, and entirely made up — and nothing downstream
can tell. **Turning the resolution up is not a text tool here; it is a hallucination
amplifier.**

If legible text really decides how your photographs are grouped, the answer is a different
model, or cropping the sign and describing that — not a larger `--max-edge`. That second
model needs no new machinery: nothing in these scripts asks *what* is answering at the other
end of `--base-url`, so a text-heavy subset can be described again against a model chosen for
reading rather than describing.

```bash
./photo-search.py "a sign or information board with text" --output-json \
  | jq -r '.results[].photo_id' > text-heavy.txt
./caption-photos.py --model <a model that reads> --only-ids text-heavy.txt --force-recaption
```

The subset is findable because at 512 px the model *announces* what it is not transcribing —
"a detailed informational map titled …" is itself the signal. What matters in that second
model is not accuracy but how it fails: a describer invents when it cannot read, a reader
returns nothing. Honesty under uncertainty is the property to select for.

One thing deliberately not done: deriving an expected language from the coordinates and
telling the model about it. The measurement above is the argument against it. The sign is
half in each language, and the invented passages were in the wrong one — a "you are here, so
expect this language" prior would not have prevented the error, it would have reinforced it
exactly where the sign switches. Supplying an expectation primes the behaviour that fails;
the colour canary works for the mirror-image reason, because its prompt names no colour. It
would also put a location back into the prompt, which is the one thing stripping EXIF from
the upload exists to prevent.

---

## What it stores

The log is append-only and is the only thing expensive to lose. Everything else is derived
and can be deleted.

```
$PHOTO_CATALOG/
  vault.json              the album root, written down exactly once
  log/captions.jsonl      append-only source of truth
  log/.lock               held while a batch runs; never broken automatically
  prepared/<xx>/<id>.jpg  the EXIF-free uploads, cached
  catalog.jsonl           derived
  index.jsonl             derived
```

A catalog line, with placeholders — **no field below is real data**; coordinate- and
timestamp-shaped literals do not appear anywhere in this repository:

```json
{"photo_id": "<16 hex>", "text": "<the description>",
 "filename": "photo.heic", "filepath": "album/photo.heic",
 "captured": "<ISO-8601>", "gps": null, "camera": "<Make Model>",
 "model": "<vision model>", "max_edge": 512, "dhash": "<16 hex>"}
```

`text` is the description and nothing but the description. Coordinates, dates and camera
names travel as separate fields on the same line: filter inputs, never embedding inputs.
`filepath` is relative to the album root, which appears only in `vault.json` — so neither a
search result nor an index line can carry it.

The three key names `text`, `filename` and `filepath` are not free choices:
[`../rag-server/cosine-search.py`](../rag-server/cosine-search.py) reads exactly those, which
is what lets the search stage reuse that file — and its same-model guard — unchanged.

### The same picture twice

A camera that writes HEIC often stores a JPEG of the same shot as well, and a raw file may
sit beside a derivative carrying metadata the raw does not. Those need not share a name, a
folder or a plausible order, so neither the filename heuristic nor the byte-exact identity
can see them. A perceptual hash over the prepared image can.

It is **reported, never acted on** in the catalog: such a hash cannot tell one shot in two
formats from two frames of a burst, and quietly discarding the second would cost pictures.
`--inherit-metadata` fills a missing date, place or camera from a matched partner but writes
it to `exif_inherited` with the partner's id — never merged into `exif`, so it stays
answerable whether a value was measured or inferred.

Where it shows up is the result list: both halves of a pair match the same query, so half
your hits can be one picture twice. `photo-search.py --collapse-duplicates` shows one per
group and pulls the next distinct photograph into the freed slot. It is off by default and
the search points it out when it would apply — folding a *view* deletes nothing and is undone
by dropping the flag, but you should see what is there before deciding to hide part of it.

The distance threshold is a knob because it has to be measured, not guessed: a format change
costs 0 bits, a 5 % crop costs 16. A raw file and its camera-developed JPEG differ in framing
more than any of that, so their threshold must come from real raw files.

---

## What it is not

Stated as limits rather than plans, because an example that hints at a roadmap ages badly.

Most of the list follows from one boundary. This is a way to *search* an album; it is not a
photo manager. The difference is not the feature count — it is who owns the truth. Here the
**album** does: everything under `$PHOTO_CATALOG` is derived, and deleting all of it costs
computing time and nothing else. A manager owns truth that exists nowhere but in its own
database — which album a picture belongs to, who is in it, what you rated it — and losing
that database loses information no amount of recomputation brings back.

So there is a test for anything proposed here, and it is a short one:

> **Delete `$PHOTO_CATALOG` entirely and run again. Do you get the same state back?**
> Yes — it belongs here. No — it belongs to whatever tool owns that data.

Joining metadata that already exists elsewhere passes the test. Originating metadata that
exists nowhere else fails it.

- **It does not recognise people.** Recognising a face would make this tool the *origin* of
  an identity claim, so its catalog would hold something unrecoverable and the test above
  would fail. Reading person names a cataloguing application already wrote into a sidecar is
  a different proposition: joined rather than originated, re-derivable, and it would sit
  beside the capture date — a filter, never part of the embedded text, on exactly the
  argument that keeps coordinates out of it. An extension point, not a plan.
- **It does not read dense small print.** Measured above: a sign covered in small text yields
  its headlines and nothing more, and raising the resolution makes the model invent the rest
  rather than admit it.
- **It does not group photographs by what they look like** — only by what the descriptions
  say. Perceptual matching is used to *report* that two files are the same picture, never to
  decide anything. Image embeddings would change how vectors are produced, not how they are
  queried; the cached uploads exist so that stage costs no re-conversion when it arrives.
- **It does not turn coordinates into place names.** A reverse-geocoded string folded into a
  description would be embedded, and a coordinate inside a vector cannot be removed again
  without rebuilding everything.
- **It does not describe several photographs in one request.** Measurably faster and
  measurably wrong: the descriptions blend and details are invented.
- **It does not run anything in parallel.** One node answers one request at a time.
  Parallelism is a different `--base-url`, not a code change — a seam, not a promise.
- **It does not touch video**, does not update the index incrementally, does not ship a user
  interface, and does not notice when a photograph disappears ("deleted" and "the drive is
  not mounted" look identical from here, and guessing wrong destroys descriptions).
- **It does not start or stop a server** — see *The scripts*.

---

## Privacy

Four mechanisms, and then the honest limits of each.

**1. Nothing leaves the machine.** The model is served by you, on loopback, and the scripts
refuse a non-loopback plain-HTTP address unless you pass `--allow-remote-base-url` — mlx-knife's
server enforces no authentication and answers any origin, so pointed elsewhere it would send
every photograph in the clear to a host that authenticates nobody. The multi-node seam this
example is built around is a portability property, not a private-by-default path, and the
flag's use is stamped into the run record.

**2. Nothing private is in the repository to begin with.** Input and output are named by
environment variable with no defaults, so there is no in-tree location to protect. The
thirteen photographs shipped here are a published fixture and the only ones that ever live
in this tree.

**3. Records carry relative paths.** The album root is written once, in `vault.json`. A
catalog line, an index line and a search result cannot contain it. Error text that arrives
from a decoder with an absolute path in it is redacted before it reaches the log.

**4. The upload carries no EXIF, by construction.** Pillow writes an EXIF block only when
one is passed to `save()`; not passing it is a guarantee, not an oversight. So the server has
nothing to extract and nothing to fold into the prompt, and a description cannot be
contaminated with a location. This does not depend on remembering an environment variable —
and unlike an environment variable, the client can *verify* the result, which is what the
self-check does. `MLXK2_EXIF_METADATA=0` and `MLXK2_VISION_METADATA_CONTEXT=0` on the server
remain worth setting as depth, but run the acceptance without them so the table can measure
the strip rather than assume it.

**What these do not cover:**

- **The description itself.** It is generated from the pixels and will transcribe text the
  model can read. A photographed document, prescription, screen or envelope therefore becomes
  searchable plaintext in `catalog.jsonl` and an irreversible component of a vector. The
  prompt is yours to change (`--prompt-file`) and can ask the model not to transcribe
  identifying text — a mitigation, not a guarantee.
- **The server's scratch space.** mlx-knife writes each uploaded image to a temporary file
  for the duration of the request and removes it afterwards, best-effort. After an abnormal
  server exit the residue is a downscaled, EXIF-free thumbnail — small, but not nothing.
- **Coordinates you type.** `--near` puts a coordinate on your own command line and in your
  shell history. It never reaches the server.

Photo metadata contains capture times and, often, coordinates. Before sharing any photograph
— or building any sample set — strip it. `exiftool` (optional, external) is the usual tool
for inspecting and removing it.

**Never commit photographs with location data to git.**

---

## Why it is built this way

What follows is the **design record**: written before any of this existed, when it was a
plan rather than a program, and kept unchanged since. It is preserved deliberately — every
decision in it still holds, and reading the reasoning in its original form is more useful
than a tidied-up restatement.

Read it as *why*, not as *what*. It is written in the voice of something not yet built, it
refers to the shipped photographs by their path in this repository, and its measurements are
the ones that produced the decisions rather than the ones the implementation later confirmed.
Everything above is what became of it.

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

---

## See also

- [`../rag-server/`](../rag-server/) — embedding and vector search over text
- [`../pipes/`](../pipes/) — vision → text pipeline building blocks
- Vision and embeddings sections in the project [README](../../README.md)

---

## License

Same as MLX Knife (Apache 2.0).
