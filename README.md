# RetinaGrade AI

[![Build Status](https://github.com/mahmudul-hasan-hasib/Retinagrade_AIi/actions/workflows/ci.yml/badge.svg)](https://github.com/mahmudul-hasan-hasib/Retinagrade_AIi/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch CPU](https://img.shields.io/badge/PyTorch-2.14.0%2Bcpu-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Deploy](https://img.shields.io/badge/deploy-Render-46E3FF?logo=render&logoColor=black)](https://render.com)

**Automated diabetic retinopathy screening from retinal scans - CPU only, explainable by design.**

RetinaGrade AI is a two-service application: a Python/FastAPI backend that runs real
EfficientNet-B0 checkpoints on the CPU, and a React/TypeScript web client that drives them.
It grades a retinal capture on the ICDR 5-class scale, returns the full probability
distribution, produces a Grad-CAM activation map for the same checkpoint, and can render
the result as prose via Gemini. Nothing on screen is mocked, and no GPU is required.

> **Research and educational use only.** This project is not a medical device, is not
> validated for clinical decision-making, and produces no diagnosis. Every prediction
> carries a fixed disclaimer, and no accuracy claim is made anywhere in this repository:
> the only images available for verification are unlabelled.

> **Badge placeholders:** the build badge points at a CI workflow that does not exist yet,
> and there is no `LICENSE` file in the tree. Both are marked in the sections below.

---

## Table of Contents

- [Overview and Key Features](#overview-and-key-features)
- [Architecture and Tech Stack](#architecture-and-tech-stack)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Local Installation and Setup](#local-installation-and-setup)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Deployment Guide](#deployment-guide)
- [Roadmap and Progress Status](#roadmap-and-progress-status)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Overview and Key Features

RetinaGrade AI accepts a single retinal photograph and returns a screening-grade result
with the evidence behind it. The backend runs two independent, purpose-trained models -
one per imaging modality - and never falls back from one to the other.

### Supported imaging modalities

| Modality | Full name | Field of view | Checkpoint | Head layout | Input |
|---|---|---|---|---|---|
| `cfp` | Color Fundus Photography | Circular posterior pole | `backend/models/EfficientNetB0_CFP_final.pth` | Multi-head: `dr_head: Linear(1280, 5)` + 7 auxiliary `lesion_heads: Linear(1280, 1)` | 224 px **(assumed)** |
| `uwf` | Ultra-Wide Field | Full retina + periphery | `backend/models/EfficientNetB0_UWF_final.pt` | Single-head: stock `efficientnet_b0(num_classes=5)` | 512 px (read from the checkpoint) |

Both trunks are `torchvision.models.efficientnet_b0`. The two checkpoints were trained
separately and are loaded separately, each with `load_state_dict(strict=True)`.

### What the system does

- **Real 5-class ICDR grading.** Every prediction returns the winning grade, its
  confidence, and the complete 5-class distribution (`No DR`, `Mild`, `Moderate`,
  `Severe`, `Proliferative DR`) - not just the top class.
- **Grad-CAM explainability.** A forward *and* backward pass through the same checkpoint
  attributes the grade to image regions at `features.8`, the last convolutional stage.
  The heatmap is returned as a base64 PNG data URI sized to the model input, alongside
  the native CAM grid (`7x7` for CFP, `16x16` for UWF).
- **Preprocessing provenance.** UWF's 512 px input size is read from its checkpoint
  config. CFP's checkpoint stores no preprocessing config at all, so 224 px is a
  documented **assumption** and every CFP response says so in machine-readable form via
  `preprocessing.values_verified_by_checkpoint: false`.
- **Optional AI diagnostics.** `POST /explain` turns an existing prediction into
  clinician-facing prose using Gemini. It runs no model of this project, never sees the
  image, and echoes the caller's numbers back verbatim under `source`, so the explanation
  provably cannot have altered the grade.
- **Uniform error contract.** Failures across both layers return
  `{"success": false, "error": {"code", "message"}}` with real status codes
  (`400`, `413`, `422`, `500`, `502`, `503`). Tracebacks are logged server-side and never
  returned.
- **Non-blocking UI.** The client requests `/predict` first and `/gradcam` as a follow-up,
  sharing one monotonic request-generation guard, so the grade appears as soon as it is
  ready and a slow or failing heatmap can never delay or discard it - nor can a stale
  response overwrite a newer capture.
- **CPU-only by construction.** `DEVICE = "cpu"` is pinned, `torch.load` uses
  `map_location="cpu"` with `weights_only=True`, every parameter and buffer is asserted to
  be on the CPU after load, and the requirements pin the `+cpu` wheel index. `cuda_available`
  is reported by `/health` for transparency only.

## Architecture and Tech Stack

```
  Browser (React 19 + TypeScript + Vite 8 + Tailwind 4)
      |  multipart/form-data: image + modality (+ optional target_class)
      |  VITE_API_URL is inlined at build time
      v
  FastAPI app (backend/main.py)  -- CORS allow-list, global exception handler
      |
      +-- POST /predict  -> inference.predictor  -> model_loader (CFP | UWF, CPU, strict)
      +-- POST /gradcam  -> inference.gradcam     -> forward + backward at features.8
      +-- POST /explain  -> services.gemini       -> optional, no model of ours runs
      |
      v
  Pillow / torchvision preprocessing -> (1, 3, S, S) float32 CPU tensor
```

**Backend** - Python 3.10.11, FastAPI 0.141.1, Uvicorn, Pydantic 2, CPU-only
PyTorch 2.14.0+cpu / torchvision 0.29.0+cpu, NumPy 2.2.6, Pillow 12.3.0,
python-dotenv, python-multipart. CPU-bound inference and network calls are dispatched
with `run_in_threadpool` so the event loop stays free.

Imaging uses **Pillow + torchvision transforms + NumPy**. OpenCV is *not* a dependency -
it was removed and is not imported anywhere in the backend. The Grad-CAM PNG colour ramp is
applied directly in NumPy and encoded with Pillow, so no plotting library is required.

**Frontend** - React 19, TypeScript 5.9, Vite 8, Tailwind CSS 4, ESLint 10. The API base
URL comes from `VITE_API_URL`, inlined at build time; blank or unset falls back to
`http://localhost:8000`. No client-side heatmap decoding, colour mapping, or compositing
happens - the `data_uri` from the server is used directly as an `<img src>`.

**Deployment** - `render.yaml` defines two independent services: a Python web service
(`retinagrade-api`) for the API and a static site (`retinagrade-web`) for the built
frontend. No Dockerfile, no GPU runtime, no managed database.

---

## Directory Structure

```
retinagrade-ai/
|-- .gitignore                        # Ignores venvs, .env, dist, uploads; KEEPS the two checkpoints
|-- render.yaml                       # Render blueprint: retinagrade-api + retinagrade-web
|-- README.md
|
|-- backend/                          # --- FastAPI inference service (CPU only) ---
|   |-- main.py                       # App, CORS, routes, Pydantic response models, error envelope
|   |-- requirements.txt              # Pinned CPU-only runtime deps + documented exclusions
|   |-- runtime.txt                   # python-3.10.11 (Render runtime pin)
|   |-- .python-version               # 3.10.11 (local toolchain pin)
|   |-- .env.example                  # Tracked env template; contains no real values
|   |
|   |-- inference/                    # The core of the project
|   |   |-- model_architecture.py     # EfficientNet-B0 reconstructed once, from checkpoint evidence
|   |   |-- model_loader.py           # Strict CPU load, per-modality ModelManager, 503 on failure
|   |   |-- preprocessing.py          # decode -> resize -> normalise, with verified/assumed provenance
|   |   |-- predictor.py              # Single source of truth: DR_CLASSES + predict()
|   |   |-- gradcam.py                # Grad-CAM at features.8, CAM grid read from activations
|   |   |-- exceptions.py             # Domain errors carrying `code` + `http_status`
|   |   |-- inspect_models.py         # CLI: read-only checkpoint inspection
|   |   |-- verify_architecture.py    # CLI: architecture vs. checkpoint (needs `timm`, offline only)
|   |   `-- test_model_loading.py     # CLI: loader smoke check
|   |
|   |-- schemas/
|   |   `-- api.py                    # Pydantic schemas for /, /health and the error envelope
|   |
|   |-- services/
|   |   `-- gemini.py                 # Optional Gemini prose explanations (lazy SDK import)
|   |
|   |-- utils/                        # Reserved for shared helpers (currently empty)
|   |
|   |-- models/                       # Tracked checkpoints - see "Model weights" below
|   |   |-- EfficientNetB0_CFP_final.pth   # ~15.7 MB, multi-head CFP network
|   |   `-- EfficientNetB0_UWF_final.pt    # ~46.4 MB, single-head UWF network
|   |
|   |-- test_images/
|   |   |-- cfp/tr000001.jpg          # Unlabelled sample capture
|   |   `-- uwf/tr000001.jpg          # Unlabelled sample capture (a different photograph)
|   |
|   |-- test_models.py                # CLI: load both checkpoints, print verified facts
|   |-- test_prediction.py            # CLI: end-to-end prediction + output-contract checks
|   `-- .venv/                        # Local virtualenv (git-ignored)
|
|-- frontend/                         # --- React + TypeScript web client ---
|   |-- package.json                  # Node engines: ^20.19.0 || >=22.12.0
|   |-- vite.config.ts                # Vite + React + Tailwind; dev server on 127.0.0.1:5173
|   |-- eslint.config.js
|   |-- tsconfig.json / .app.json / .node.json
|   |-- index.html
|   |-- .env.example                  # Sets VITE_API_URL=http://localhost:8000
|   |
|   |-- src/
|   |   |-- App.tsx                   # Shell, shared capture state, sequential predict -> gradcam
|   |   |-- main.tsx
|   |   |-- types.ts                  # ModalityKey, PredictApiResponse, NavItemId, ...
|   |   |-- index.css
|   |   |-- api/
|   |   |   `-- client.ts             # The ONLY network code: predict, gradcam, health, explain
|   |   |-- data/
|   |   |   `-- clinical.ts           # DR grade map, modalities, per-modality model metadata
|   |   |-- hooks/
|   |   |   `-- useImageSelection.ts  # File selection, object-URL preview lifecycle
|   |   |-- lib/
|   |   |   `-- format.ts             # Percent / "no value" formatting
|   |   |-- components/
|   |   |   |-- analyze/              # ImageDropzone, ImagePreview, ModalitySelector,
|   |   |   |                         # AnalysisResultCard, DrGradeScale
|   |   |   |-- layout/               # AppShell, Brand, SidebarNav, SidebarContent, TopBar
|   |   |   |-- background/           # AppBackground
|   |   |   |-- ui/                   # Badge, Button, Card, EmptyState, Feedback, Placeholder
|   |   |   `-- icons/                # Inline SVG icon set
|   |   `-- pages/
|   |       |-- AnalyzePage.tsx       # Drop a capture, pick a modality, run the analysis
|   |       `-- ModelPage.tsx         # Architecture, checkpoints, last-run inference config
|   |
|-- docs/                             # Reserved for documentation (.gitkeep only)
|-- tests/                            # Reserved - NO automated suite yet (.gitkeep only)
|
`-- *_PROGRESS*.md / *_STATUS*.md    # Point-in-time engineering status reports (4 files, root)
```

The four `test_*.py` files are **CLI verification scripts**, not a pytest suite. `pytest`
and `httpx` are intentionally absent from `requirements.txt`, and `tests/` holds only a
`.gitkeep`.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | **3.10.x** (pinned to 3.10.11 in `backend/.python-version` and `runtime.txt`) | 3.10 or 3.11 works; 3.10.11 is what is validated |
| Node.js | **^20.19.0 \|\| >=22.12.0** | Enforced by `engines` in `frontend/package.json` |
| npm | 10+ | Ships with Node 20.19 / 22.12 |
| GPU / CUDA | **None** | See below |
| Disk | ~2-3 GB free | CPU-only PyTorch wheels plus the two checkpoints |
| RAM | 2 GB+ | Torch plus a 46 MB checkpoint loaded in-process |

> **CPU-only setup, no CUDA overhead.** `backend/requirements.txt` sets
> `--extra-index-url https://download.pytorch.org/whl/cpu` so `torch==2.14.0+cpu` and
> `torchvision==0.29.0+cpu` are resolved instead of the ~2.5 GB CUDA-enabled default
> wheels from PyPI. Do not install a CUDA build: the models were trained on CUDA but this
> project loads and runs them with `map_location="cpu"`, and a GPU build would only add
> hundreds of megabytes and the temptation to bypass the CPU assertion. Verify after
> installing:
>
> ```bash
> python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
> # 2.14.0+cpu None False
> ```

---

## Local Installation and Setup

### 1. Backend

```powershell
# from the repository root
cd backend

# create the virtualenv (the repo already ships one at backend/.venv - reuse it)
py -3.10 -m venv .venv

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS / Linux:

```bash
cd backend
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Verify the environment and confirm CPU-only:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### 2. Environment configuration

The backend needs **no credentials and no host/port variables** for local development.
Copy the template if you want one:

```powershell
copy .env.example .env     # Windows
cp .env.example .env       # macOS / Linux
```

`.env` is git-ignored and must never be committed. `main.py` loads it with
`load_dotenv(..., override=False)`, so real environment variables always win and a local
file can never silently override a deployment.

| Variable | Required | Purpose |
|---|---|---|
| `CORS_ORIGINS` | Deployment only | Comma-separated extra browser origins. Unset locally; must be set on Render. |
| `GEMINI_API_KEY` | No | Enables `POST /explain`. Also needs `google-genai` installed (see below). |
| `GEMINI_MODEL` | No | Defaults to `gemini-2.5-flash`. |
| `GEMINI_TIMEOUT_SECONDS`, `GEMINI_MAX_OUTPUT_TOKENS`, `GEMINI_TEMPERATURE` | No | Generation tuning. |
| `CFP_IMAGE_SIZE` | No | Overrides the assumed CFP input size (positive integer). Never touches weights. |

**Enabling the optional Gemini diagnostics.** `google-genai` is deliberately *not* in
`requirements.txt`, so without it `/explain` answers `503 gemini_not_configured` and
`/health` reports `gemini_configured: false` even when a key is present. `/predict`,
`/gradcam` and `/health` are unaffected. To enable it:

```bash
.\.venv\Scripts\python.exe -m pip install google-genai==2.28.0
# then add to backend/.env
# GEMINI_API_KEY=your-key-here
# GEMINI_MODEL=gemini-2.5-flash
```

### 3. Model weights

Both production checkpoints are **tracked in git on purpose**, so a fresh clone already
contains them and there is nothing to download:

```
backend/models/EfficientNetB0_CFP_final.pth   # ~15.7 MB
backend/models/EfficientNetB0_UWF_final.pt    # ~46.4 MB
```

Both are well under GitHub's 100 MB per-file limit. They must be committed because
`inference/model_loader.py` resolves them from `__file__` as `<backend>/models/<name>`, and
a Render build with `rootDir: backend` finds them only if they are in the repository.

> **Do not** add `*.pt`, `*.pth` or `models/` to `.gitignore`. The root `.gitignore`
> excludes `backend/models/*` and then re-includes exactly these two files plus
> `.gitkeep`; everything else in that directory stays ignored. If a checkpoint is missing
> or mismatched, `/predict` and `/gradcam` answer `503 model_unavailable` and
> `/health` stays green.

Paths are derived from the module location, so the working directory does not matter.
There is no `MODEL_DIR` / `MODEL_PATH` variable in this codebase.

Confirm the weights load before starting the server:

```powershell
cd backend
.\.venv\Scripts\python.exe test_models.py
```

### 4. Frontend

```powershell
cd frontend
npm install
copy .env.example .env     # sets VITE_API_URL=http://localhost:8000
```

Vite inlines `VITE_*` variables at **build time**, so a production build must have
`VITE_API_URL` set before `npm run build`. Unset or blank falls back to
`http://localhost:8000`; trailing slashes are stripped.

Quality gates (both currently pass with zero findings):

```powershell
npm run typecheck   # tsc -b --noEmit
npm run lint        # eslint .
npm run check       # both
```

---

## Running the Application

Use two terminals. Both servers are required for the UI to be functional.

**Terminal 1 - API (from `backend/`):**

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 - web client (from `frontend/`):**

```powershell
npm run dev
```

| Service | URL |
|---|---|
| API | http://127.0.0.1:8000 |
| Interactive OpenAPI docs | http://127.0.0.1:8000/docs |
| OpenAPI schema | http://127.0.0.1:8000/openapi.json |
| Web client | http://127.0.0.1:5173 |

Both origins are pre-allowed by the backend's CORS configuration, so no extra setup is
needed locally.

> Models load **lazily on the first request for a modality**, so that first call also
> pays the checkpoint load - roughly 6-8 s. UWF Grad-CAM at 512x512 takes ~20 s on CPU
> because it runs a forward and a backward pass. Both are normal, not hangs.

Optional end-to-end CLI checks:

```powershell
cd backend
.\.venv\Scripts\python.exe test_prediction.py --image test_images\cfp\tr000001.jpg --modality cfp
.\.venv\Scripts\python.exe test_prediction.py --image test_images\uwf\tr000001.jpg --modality uwf
```

Sample requests:

```bash
# health
curl http://127.0.0.1:8000/health

# grade an image
curl -X POST http://127.0.0.1:8000/predict \
  -F "image=@backend/test_images/cfp/tr000001.jpg" \
  -F "modality=cfp"

# grade + Grad-CAM heatmap (optionally explain a specific grade 0-4)
curl -X POST http://127.0.0.1:8000/gradcam \
  -F "image=@backend/test_images/uwf/tr000001.jpg" \
  -F "modality=uwf"

# prose explanation of an existing prediction (requires Gemini + google-genai)
curl -X POST http://127.0.0.1:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"modality":"cfp","predicted_class":2,"confidence":0.703081,
       "probabilities":{"No DR":0.004808,"Mild":0.018070,"Moderate":0.703081,
                        "Severe":0.273637,"Proliferative DR":0.000403}}'
```

---

## API Endpoints

Base URL: `http://127.0.0.1:8000` locally, the `retinagrade-api` service URL when deployed.
API version: `0.6.0-dev`.

| Method | Path | Purpose | Model run | Success |
|---|---|---|---|---|
| `GET` | `/` | Service banner (`app`, `status`) | No | `200` |
| `GET` | `/health` | Liveness, `device`, `cuda_available`, `gemini_configured` | No | `200` |
| `POST` | `/predict` | Grade an uploaded capture; returns class, label, confidence, full 5-class distribution, preprocessing provenance | **Yes** - forward pass | `200` |
| `POST` | `/gradcam` | Same grading plus a Grad-CAM heatmap as a base64 PNG `data_uri` | **Yes** - forward + backward | `200` |
| `POST` | `/explain` | Gemini prose for an existing prediction; echoes inputs under `source` | No | `200` |
| `GET` | `/docs`, `/redoc`, `/openapi.json` | FastAPI-generated interactive documentation | No | `200` |

### Request fields

| Endpoint | Fields | Notes |
|---|---|---|
| `/predict` | `image` (file), `modality` (`cfp`\|`uwf`) | `multipart/form-data`; max 25 MB |
| `/gradcam` | `image` (file), `modality`, `target_class` (optional, 0-4) | Omit `target_class` to explain the model's own prediction |
| `/explain` | JSON: `modality`, `predicted_class`, `confidence`, `probabilities`, optional `gradcam` metadata | No image is accepted or transmitted |

Accepted image formats: `JPEG`, `PNG`, `BMP`, `TIFF`, `WEBP`. EXIF orientation is applied;
images are converted to RGB, bilinearly resized, scaled to `[0, 1]` and normalised with
ImageNet mean/std. Training-time augmentation (`RandomHorizontalFlip`, `RandomRotation`)
is deliberately **not** applied at inference.

### Error contract

Every failure - from either layer - returns the same envelope, never a traceback:

```json
{ "success": false, "error": { "code": "unsupported_modality", "message": "..." } }
```

| Status | Typical `code` | Cause |
|---|---|---|
| `400` | `unsupported_modality`, `invalid_target_class`, `invalid_confidence`, `invalid_probabilities` | Client sent an unusable request |
| `413` | `image_too_large` | Upload exceeds 25 MB |
| `422` | `invalid_image` | Missing, empty or undecodable image |
| `500` | `gradcam_failed`, `inference_failed`, `internal_error` | Inference or encoding failure |
| `502` | `gemini_upstream_error`, `gemini_empty_response` | Gemini refused or returned nothing usable |
| `503` | `model_unavailable`, `gemini_not_configured` | Checkpoint missing/mismatched, or Gemini not set up |

> `POST /gradcam` returns `500 gradcam_failed` rather than a heatmap when no gradient
> reached the target layer, because an all-zero map is not evidence. `gradients_are_nonzero`
> is reported on every successful response.

---

## Deployment Guide

`render.yaml` is a Render Blueprint that declares both services. The frontend is a static
site and the API is a Python web service; they are separate origins, which is why CORS
configuration is mandatory in production.

### Deploy

1. **Push the repository to GitHub.** The two checkpoints must be committed - a Render
   build with `rootDir: backend` will not find them otherwise, and `/predict` and
   `/gradcam` will answer `503`.
2. **Create the Blueprint.** In Render: *New -> Blueprint* -> select this repository. Render
   parses `render.yaml` and creates both services.
3. **Fill in the two prompted values.** Both are declared `sync: false`, so Render asks for
   them on first deploy:

   | Service | Variable | Value |
   |---|---|---|
   | `retinagrade-api` | `CORS_ORIGINS` | `https://retinagrade-web.onrender.com` - the **static site's** public URL. Comma-separated for multiple origins, scheme included, no trailing slash. |
   | `retinagrade-web` | `VITE_API_URL` | `https://retinagrade-api.onrender.com` - the **API's** public URL. Read at build time by Vite. |

4. **Order matters.** The static site's URL is needed for `CORS_ORIGINS`, and the API's URL
   is needed for `VITE_API_URL`. Either let the first deploy fail the prompt and fill it in
   afterwards, or note the generated names (`https://<service-name>.onrender.com`) and set
   both values before the first successful deploy. Without `CORS_ORIGINS` the browser
   blocks every `/predict` and `/gradcam` call as cross-origin, so the UI appears healthy
   but never returns a grade.

### Blueprint contents

| Service | Type | Root | Build | Start / publish |
|---|---|---|---|---|
| `retinagrade-api` | `web` (Python, `free`) | `backend` | `pip install -r requirements.txt` | `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`, health check `/health` |
| `retinagrade-web` | `web` (static, `free`) | `frontend` | `npm install && npm run build` | publishes `./dist` |

### Free-tier behaviour to plan for

- **Cold starts.** Free web services spin down after inactivity. The first request, and the
  first request for each modality, pays the lazy checkpoint load (~6-8 s).
- **Single worker.** `--workers 1` is required: each worker would load its own copy of the
  checkpoints into RAM.
- **Grad-CAM on UWF is slow.** A forward + backward pass at 512x512 takes ~20 s on CPU, which
  can approach free-tier request timeouts. CFP Grad-CAM is ~1 s. A timeout does not affect
  `/predict`, which is ~1 s of compute after the load.
- **Gemini is not deployed.** `google-genai` is absent from `requirements.txt`, so `/explain`
  answers `503` in production by design. Add `google-genai==2.28.0` to
  `requirements.txt` plus a `GEMINI_API_KEY` environment variable on `retinagrade-api` to
  enable it.

### Verify

```bash
curl https://retinagrade-api.onrender.com/health
# {"status":"healthy","device":"cpu","cuda_available":false,"gemini_configured":false}

curl -X POST https://retinagrade-api.onrender.com/predict \
  -F "image=@retina.jpg" -F "modality=cfp"
```

Then open the static site, drop in a capture and press **Analyze Image**: the grade should
appear first, followed by the Grad-CAM panel.

---

## Roadmap and Progress Status

### Implemented

- [x] CPU-only FastAPI service (`main.py`) with `DEVICE = "cpu"` pinned and asserted
- [x] Endpoints: `/`, `/health`, `/predict`, `/gradcam`, `/explain` with a uniform error envelope
- [x] Architecture reconstructed once from checkpoint evidence (`model_architecture.py`)
- [x] Strict, CPU-pinned, `weights_only=True` checkpoint loading for CFP and UWF
- [x] Real 5-class ICDR prediction on both modalities, full probability distribution returned
- [x] Preprocessing with machine-readable verified-vs-assumed provenance per modality
- [x] Grad-CAM at `features.8`, CAM grid read from activations, PNG encoded server-side
- [x] Optional Gemini explanation endpoint that never sees the image and echoes its inputs
- [x] React 19 + TypeScript client: Analyze -> Explainability -> Model, no mock data anywhere
- [x] Monotonic request-generation guard so grade and heatmap can never describe different images
- [x] CORS allow-list including any localhost port, plus deployment `CORS_ORIGINS`
- [x] `render.yaml` blueprint for separate API and static-site services
- [x] CLI verification scripts for checkpoint loading and end-to-end prediction contracts
- [x] Frontend `typecheck` and `lint` pass with zero findings

### Upcoming

- [ ] **Grad-CAM overlay compositing** - blend the heatmap onto the source photograph in the
      model-input frame (currently the raw activation map ships alone)
- [ ] **`/explain` UI** - `requestExplanation` and `fetchHealth` exist in the client but no
      component calls them
- [ ] **PDF export** of a screening report (`reportlab` is not a dependency yet)
- [ ] **Screening history and persistence** for graded captures
- [ ] **Automated test suite** - `tests/` is empty; add pytest + httpx for the API and a
      frontend test runner
- [ ] **CI workflow** - the build badge in this README is a placeholder until one exists
- [ ] **Ground-truth evaluation** - dataset labels, QWK / sensitivity reporting
- [ ] **Resolve the CFP input size** once the real CFP training resolution is known
- [ ] **Frontend routing** - navigation is currently state-driven in `App.tsx`, no router
- [ ] **Auth and rate limiting** for a public deployment
- [ ] **`LICENSE` file** - the badge is a placeholder until the license is chosen

---

## Known Limitations

- **Not a medical device.** Screening support only; no diagnosis, no accuracy claim, and a
  fixed disclaimer is appended to every generated explanation.
- **CFP input size is an assumption.** `EfficientNetB0_CFP_final.pth` stores no `config`
  key - no image size, no normalisation, no colour mode. 224 px is the canonical
  EfficientNet-B0 input, but the sibling UWF checkpoint records 512. EfficientNet-B0 is
  fully convolutional with global pooling, so a wrong size **fails silently**. Every CFP
  response therefore carries `values_verified_by_checkpoint: false`. Set `CFP_IMAGE_SIZE` if
  the real resolution becomes known.
- **The class-index to ICDR-grade mapping is inferred, not stored** in either checkpoint.
- **The CFP model's seven auxiliary lesion heads are never read.** The checkpoints contain
  no label map for them, so their outputs are not surfaced as clinical findings.
- **No accuracy metrics are possible** from the two unlabelled sample images in the repo.
- **UWF checkpoint metadata is internally inconsistent** - its `config` dict carries stale
  `experiment` / `dataset` labels. Only `image_size` and `normalization` are read from it.

---

## Contributing

Contributions are welcome. Please read the notes below before opening a pull request - they
describe constraints this codebase enforces deliberately.

1. **Open an issue first** for anything beyond a small fix, so the approach can be agreed
   before the work is done.
2. **Keep inference CPU-only.** Do not add CUDA device selection, GPU dependencies, or
   `map_location` changes. `DEVICE` is pinned and every tensor is asserted to be on the CPU.
3. **Never relax `strict=True`.** A partial `load_state_dict` leaves randomly initialised
   weights inside the network and produces confident nonsense. Do not reshape the
   architecture to force a load either - fail loudly instead.
4. **Do not commit secrets.** `.env` is git-ignored; edit `backend/.env.example` and
   `frontend/.env.example` instead. They are the only tracked templates and contain no real
   values.
5. **Do not git-ignore the checkpoints**, and do not add `*.pt` / `*.pth` / `models/` to any
   `.gitignore`. Fresh clones and Render builds depend on them being tracked.
6. **Match the API contract.** Responses and errors flow through the
   `{"success": ..., "error": {"code", "message"}}` envelope; tracebacks are logged, never
   returned. Add new error codes to `inference/exceptions.py` / `services/gemini.py` rather
   than raising bare exceptions.
7. **No fabricated data.** Nothing in the frontend may hardcode a grade, confidence,
   probability or heatmap - every rendered value must come from a server response.
8. **Keep the frontend clean.** `npm run check` (typecheck + lint) must pass with zero
   findings before you push.
9. **Document assumptions as assumptions.** If a value is not read from a checkpoint, say so
   in the payload rather than presenting it as fact.
10. **One logical change per pull request**, with a description of what you verified and how.

Commit messages use short, imperative, lowercase subjects (`add gradcam endpoint and heatmap
panel`, `clean stale gradcam ui`).

---

## License

Released under the **MIT License**.

> **Placeholder.** No `LICENSE` file exists in the repository yet; the badge at the top of
> this file is a placeholder. Add the license text before publishing.

```
MIT License

Copyright (c) 2026 RetinaGrade AI contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```