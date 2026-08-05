# Photo RAG — a family album you can actually search

Describe every photograph in a private collection with a vision model, then find things
in it by what is *in* the picture, by when it was taken, and by where.

**Nothing leaves your machine.** The model runs here; the photographs are read here, the
descriptions are written here, and the search happens here. That is the whole reason this
is worth doing at all — a family album is exactly the kind of collection nobody wants to
upload to a service, and it is exactly the kind of collection that is unsearchable without
one. A local model closes that gap.

This is written for **one person's own collection**. Everything here assumes that.

**Status:** ✅ Runnable
**Runnable against:** mlx-knife **≥ 2.0.7**. Needs three surfaces, not a version: vision
through `mlxk serve` with base64 `image_url`; the `<!-- mlxk:filenames -->` metadata block
the server prepends to every vision answer; and `mlxk embed --batch` / `--query`
(experimental, alpha-gated — the scripts set the flag for their own subprocesses).
Verified on 2.0.7. **Nothing here requires a change to mlx-knife.**
**Requires:** a vision model you serve yourself; an embedding model in your cache; Python
`httpx pillow numpy`; macOS `/usr/bin/sips` for HEIC/RAW
**Run:** see *Try it in five minutes* — no photo library needed

> **How far this has been taken.** Every figure published here and in the manual comes from
> the thirteen photographs shipped with this example, never from a private collection. That
> is a rule rather than a gap: what a particular library measures describes that library, not
> this tool, so those numbers stay out of these pages while what they teach about *mechanism*
> does not. The design is built for a multi-day run over tens of thousands — that is what the
> resume, quarantine and locking machinery is for. Treat the timings as a starting estimate
> and the long-run behaviour as designed-for rather than demonstrated.
>
> The design record in [`MANUAL.md`](MANUAL.md) is the original reasoning, written before any
> of this existed and unchanged since.

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
  repeat. HEIC from a phone works; so does raw, through macOS. Read
  [*Your own album*](MANUAL.md#your-own-album) first if that folder is one you care about.

---

## The manual

**[`MANUAL.md`](MANUAL.md)** documents everything past the five-minute run: pointing this at
your own library, what each script does, what a run costs, what it writes down, and the
original design record.

One section there is worth knowing about before you search anything for real —
[*Reading the scores*](MANUAL.md#reading-the-scores). A cosine score does not start at zero,
so the `--min-score` default of `0.0` is not a usable threshold: ask the index for a few
things it cannot possibly contain, and the best score any of them returns is your floor.
Measuring it takes under a minute and is the difference between a result list you can trust
and one that always looks plausible.

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
- **It does not read dense small print.** Measured in
  [*What it costs*](MANUAL.md#what-it-costs): a sign covered in small text yields
  its headlines and nothing more, and raising the resolution makes the model invent the rest
  rather than admit it. The consequence reaches further than a wrong sentence, and it has been
  observed: a photographed price sign whose product was named wrongly is *findable under the
  wrong word and unfindable under the right one*. One invented word produces a false positive
  and a false negative at once, and no layer above can see it — the wrong hit scores like a good
  match, the missing one like an honest "nothing found". Wherever meaning comes from text inside
  the picture — labels, products, prices, documents — the index inherits the description model's
  error rate silently. Searching for scenes, places and objects is on much firmer ground.
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
- **It does not start or stop a server** — see [*The scripts*](MANUAL.md#the-scripts).

---

## Privacy

Four mechanisms, each with an honest limit; both halves are set out in
[`MANUAL.md`](MANUAL.md#privacy). In short:

- **Nothing leaves the machine.** The model is served by you, on loopback, and the scripts
  refuse a non-loopback plain-HTTP address unless you say so explicitly.
- **Nothing private is in this repository to begin with.** Album and catalog are named by
  environment variable with no defaults, so there is no in-tree location to protect.
- **Records carry album-relative paths.** The absolute root is written once, to `vault.json`;
  a catalog line, an index line and a search result cannot contain it.
- **The upload carries no EXIF, by construction** — the server has nothing to extract, so a
  description cannot be contaminated with a location. The self-verifying run *measures* that
  rather than assuming it.

What they do not cover — a description transcribing text out of a photograph, the server's
temporary scratch file, and coordinates you type on your own command line — is set out in
the manual with the same candour.

**Never commit photographs with location data to git.**

---

## See also

- [`../rag-server/`](../rag-server/) — embedding and vector search over text
- [`../pipes/`](../pipes/) — vision → text pipeline building blocks
- Vision and embeddings sections in the project [README](../../README.md)

---

## License

Same as MLX Knife (Apache 2.0).
