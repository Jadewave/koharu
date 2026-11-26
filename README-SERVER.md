# Koharu server

`koharu-server` runs Koharu's pipeline (detect, OCR, inpaint, analyse, translate) behind HTTP, for a
container with no screen, such as a Hugging Face Docker Space. It has a page of its own at `/` to
upload a chapter (images, a ZIP/CBZ, or a `.khr` project) to, follow, read, correct and download; it
has no editor canvas, which stays the desktop app's.

## Editor

`/editor?job=<id>` (the *Open editor* and *Edit* buttons) is a browser editor on the project's own
scene. It draws the page as the server renders it and puts the texts' boxes over it.

- **Select / move / resize** a text box (drag, 8 handles, arrow keys nudge, Shift for 10 px); Shift-click
  for several; **Add text** draws a new box; **Pan** (or Space) and the wheel to zoom; **F** fits.
- **Inspector**: source and translation text, font (the renderer's list), size, weight, italic, fill,
  stroke and its width, align, line height, direction, role (text / sound effect); duplicate, delete.
- **Re-run** on the selected texts: Re-OCR, Re-translate, OCR + translate, Re-inpaint; or on the whole
  page (Detect + OCR, Inpaint, Translate, all). The settings under *Settings for runs from here*
  (target language, translation model, instructions) apply to those runs.
- **Inpaint brush**: paint an area, with an optional prompt; it becomes an inpaint run on that area.
- **Undo / redo** (Ctrl+Z, Ctrl+Shift+Z) over every edit, the pipeline's included; the project is saved
  after each one.
- A page rail of the scans, and the original/translated toggle.

Every edit is the scene's own `Edit` (`POST /api/jobs/{id}/edit`), the JSON the desktop editor and the
agents send, so anything its ops do can be sent by hand. What it lacks next to the desktop editor:
a layers panel, per-character rich-text styling, rotation and path or shape frames, painting a block's
mask or clean-up by hand, and the agent panel.

## Batch images

The analyzer takes pages in batches: *Batch: pages per analyzer request* (default 12, the context
pages included), *overlap* (default 2, the pages a window repeats), image size and WebP quality, in the
upload form, in a run's settings, or as `KOHARU_ANALYZER_*`. A batch of 1 sends one page at a time.
Any number of images, or a ZIP/CBZ of them, can be uploaded at once.

## Library

Every job is kept: a folder `<data>/jobs/<id>/` holds its project, `pages.khr`, and `job.json` (its
name, date, page count, last state, translator and target). The page's *Saved works* list shows them
after a restart; a job opens from its file when asked for. A job the server was running when it
stopped comes back as `interrupted`.

- **Upload a `.khr`** (one file, from the desktop app or from a download here) to keep it and edit its
  texts directly. It runs nothing unless you name `stages`; any job can later be run again with the
  stages and settings of your choice (*Run again*), such as `translate` with another model.
- **Download `.khr`** gives a consistent copy of the project made while the job stays open, which the
  desktop app opens.
- Rename or delete a job from the list.

For the library to survive a container restart, `KOHARU_DATA` must be on persistent storage (on a
Hugging Face Space, the `/data` volume).

## What a run does

1. **Detect, recognize, inpaint** every page. The OCR is Hayai by default; PP-OCRv6 Manga, Baberu
   and Manga OCR AR are the others (`recognizer`).
2. **Analyse** the whole work, once all pages are read and cleaned. A hosted model that reads images
   (an OpenAI-compatible server's) gets windows of 12 pages, 2 of them repeated from the window
   before as context. Each page goes as an image with its lettering erased and its texts' boxes
   numbered, encoded as WebP at quality 90 and at most 1600 px on a side, beside the OCR lines. It
   answers with the characters, and for each page a summary and, for each line, the speaker, the tone
   and a corrected reading where the OCR was wrong.
3. **Translate** with the model you chose (local `koharu/...` or hosted). Its system prompt holds the
   characters and every page's summary; each page's turn holds that page's own notes. The model's
   whole context window is the budget: a local model asks for the most its training allows and gets
   what the free memory holds, and summaries that cannot fit are left off the end.

If the analyzer is not set, or fails, translation goes on without notes.

## Settings

Koharu's own `[pipeline]` and `[providers]` sections in `~/.koharu/config.toml` apply, then these
environment variables over them, then the form of an upload.

| Variable | Meaning |
|---|---|
| `KOHARU_TRANSLATOR_MODEL` | `vendor/model` of the translator |
| `KOHARU_TARGET_LANGUAGE` | a language CLDR names in English, such as `Vietnamese` |
| `KOHARU_INSTRUCTIONS` | your own instructions to the translator |
| `KOHARU_RECOGNIZER` | `hayai-ocr`, `pp-ocr-v6-manga`, `baberu-ocr`, `manga-ocr-ar` |
| `KOHARU_INPAINTER` | an inpainter id, as `[pipeline]` writes it |
| `KOHARU_ANALYZER_MODEL` | `openai-compatible/<model id>`; empty turns the analysis off |
| `KOHARU_ANALYZER_PAGES` / `_OVERLAP` / `_SIZE` / `_QUALITY` | window (12), overlap (2), longest side in px (1600), WebP quality (90) |
| `KOHARU_OPENAI_COMPATIBLE_BASE_URL` | the server's URL, written to `[providers]` |
| `KOHARU_OPENAI_COMPATIBLE_CONTEXT_WINDOW` | its context window in tokens; set it to the real value |
| `KOHARU_SECRET_<VENDOR>` | a vendor's API key, such as `KOHARU_SECRET_OPENAI_COMPATIBLE` |
| `KOHARU_HOST`, `KOHARU_PORT`, `KOHARU_DATA` | listen address, port and the folder of the library (default `~/.koharu/server`) |

A container has no OS credential store, so API keys come only from `KOHARU_SECRET_*` there.

## Hugging Face Space

Use the repository's `Dockerfile` and put this at the top of the Space's `README.md`:

```yaml
---
title: Koharu
sdk: docker
app_port: 7860
---
```

Put the API keys under the Space's *Secrets* (`KOHARU_SECRET_OPENAI_COMPATIBLE`, …) and the rest
under *Variables*. Enable persistent storage and set `KOHARU_DATA=/data` so the saved works survive a
restart. Models download on first use into the container's home, so without persistent storage
they download again after a restart too.

## API

| Route | |
|---|---|
| `GET /api/health`, `GET /api/config` | liveness; the settings in force |
| `GET /api/models` | every model the page may offer, as `{reference, name, vendor, images, tools}`: Koharu's own, and each vendor's that has a key; `images` says it reads page images (what the analyzer needs) |
| `GET /api/provider`, `POST /api/provider` | the OpenAI-compatible server's address: `{"base_url", "context_window"}`. POST sets it (empty `base_url` forgets it; a missing `context_window` keeps the old one), then lists that server's models: `{"models": [...], "error"?}`. The address is kept even when the server can't be asked. Its key is the secret `KOHARU_SECRET_OPENAI_COMPATIBLE`; the page doesn't take one. |
| `GET /api/jobs` | the library, newest first |
| `POST /api/jobs` | multipart: `files` (images, ZIP/CBZ of them, or one `.khr`) and optional `name`, `translator`, `target`, `instructions`, `analyzer` (`off`), `recognizer`, `stages` |
| `GET /api/jobs/{id}` | name and progress |
| `PATCH /api/jobs/{id}` | `{"name": "..."}` renames |
| `POST /api/jobs/{id}/run` | JSON of strings: `stages` (default `translate`) and the same settings; runs again |
| `GET /api/jobs/{id}/blocks` | each text and its translation |
| `GET /api/jobs/{id}/pages/{n}/scene` | the page's layers, as the editor reads them |
| `POST /api/jobs/{id}/edit` | a scene `Edit`: `{"label": ..., "ops": [...]}` |
| `POST /api/jobs/{id}/undo`, `/redo` | `{"changed": bool}` |
| `POST /api/jobs/{id}/process` | `{stages, blocks?, pages?, area?, settings?}`: runs stages on blocks, pages or a painted area |
| `GET /api/fonts` | the font families |
| `PUT /api/jobs/{id}/blocks/{layer}` | `{"text": "..."}` replaces a translation |
| `GET /api/jobs/{id}/pages/{n}/image[?view=original[&w=N]]` | a page as drawn, as uploaded, or a JPEG thumbnail |
| `GET /api/jobs/{id}/download?format=cbz\|psd\|khr` | a CBZ of PNGs, a ZIP of PSDs, or the project |
| `POST /api/stop`, `DELETE /api/jobs/{id}` | stop the runs; delete a job and its project |

## Not verified

Nothing here was compiled or run. The environment this was written in had no `libclang`, so
`bindgen` crates would not build; the new Rust code only passed `rustfmt`'s parser, the page's
JavaScript only parsed, and the unit tests that were written have not been run. The API calls into
`rig`, `koharu-scene` and `koharu-renderer` follow the existing code, but expect to fix compile
errors on the first full build. Also untested: the Docker image, Vello on a software Vulkan device,
`Repository::tag` against Hugging Face, and PP-OCRv6 manga's ONNX Runtime loading in this build.
