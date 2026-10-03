# RetinaGrade AI — Current Status Report

**Date:** 3 October 2026 · **Repo:** `F:\retinagrade-ai` · **Branch:** `main` (in sync with `origin/main`)
**HEAD:** `80e198b` "alter design" — Mahmudul Hasan Hasib, Sat Oct 3 05:26:16 2026 +0600
**Basis for this report:** `git status`/`log`, source inspection, live HTTP calls against the running API on `127.0.0.1:8000`, `npm run typecheck`, `npm run lint`.
**No source modified. No packages installed. Nothing committed or pushed.** The only file created is this report.

Legend: ✅ verified now · ⚠️ caveat/limitation · ❌ not implemented

---

## 1. Git state

Working tree is **DIRTY** — 6 modified, 7 untracked (before this report).

| State | Path | Note |
|---|---|---|
| M | `.gitignore` | now tracks `frontend/dist/`, re-includes the two checkpoints |
| M | `backend/.env.example` | deployment-oriented rewrite |
| M | `backend/.gitignore` | checkpoints no longer ignored; `.env.*` ignored, `!.env.example` kept |
| M | `backend/main.py` | adds `CORS_ORIGINS` env allow-list (`_extra_cors_origins()`, used at `main.py:155-176`) |
| M | `backend/requirements.txt` | CPU-only pins, `opencv-python` removed, deps annotated |
| M | `frontend/src/data/clinical.ts` | `BACKEND_BASE_URL` now from `VITE_API_URL`, default `http://localhost:8000` |
| ?? | `backend/models/EfficientNetB0_CFP_final.pth` | 15.65 MB — deliberately meant to be committed |
| ?? | `backend/models/EfficientNetB0_UWF_final.pt` | 46.40 MB — deliberately meant to be committed |
| ?? | `backend/.python-version` (`3.10`), `backend/runtime.txt` (`python-3.10.11`) | Render/uv pin |
| ?? | `frontend/.env.example`, `render.yaml` | deployment config |

Recent commits (newest first): `80e198b` alter design · `67e57bf` FRONT END DESIGN · `d72b545` Clean stale Grad-CAM UI · `d56b5fd` add gridcam · `42835af` add gradcam endpoint and heatmap panel · `ab1deeb` connect api.

✅ `backend/inference/gradcam.py` is **tracked** (the previous report's blocker is resolved). Test images `backend/test_images/{cfp,uwf}/tr000001.jpg` are tracked and unmodified. `frontend/dist/assets/` is correctly git-ignored.

---

## 2. Backend status and endpoints

✅ The API is **live and current** — not stale. `GET /openapi.json` reports `info.version = 0.6.0-dev` with paths `/, /health, /predict, /gradcam, /explain`.

| Method | Path | Model run | Live result |
|---|---|---|---|
| GET | `/` | no | ✅ 200, service banner |
| GET | `/health` | no | ✅ 200 `{"status":"healthy","device":"cpu","cuda_available":false,"gemini_configured":true}` |
| POST | `/predict` | **yes** | ✅ 200 both modalities (below) |
| POST | `/gradcam` | **yes** | ✅ 200 both modalities (below) |
| POST | `/explain` | no | ⚠️ implemented + Gemini key present, **not exercised** in this session; no UI caller |

`API_VERSION = "0.6.0-dev"` (`main.py:96`). Errors use one envelope `{"success": false, "error": {code, message}}`; the global handler logs tracebacks server-side and never returns them. CPU-bound inference is offloaded with `run_in_threadpool`.

⚠️ Two `python` processes are listening on port 8000 (`127.0.0.1` PID 37136, `0.0.0.0` PID 13868, started 04:17 and 06:03). Both report the same version; the duplicate is untidy but not breaking.

---

## 3. Real prediction results (re-verified live just now)

| | CFP | UWF |
|---|---|---|
| file | `EfficientNetB0_CFP_final.pth` | `EfficientNetB0_UWF_final.pt` |
| input size used | 224 ⚠️ assumed | 512 ✅ from checkpoint `config` |
| result | **Moderate** (class 2) | **Mild** (class 1) |
| confidence | 0.703081 | 0.881828 |
| Σ probabilities | 0.999999 | 0.999999 |
| device | cpu | cpu |
| latency | 6268.77 ms (includes lazy checkpoint load) | 7300.46 ms |

CFP distribution: `No DR 0.004808 · Mild 0.018070 · Moderate 0.703081 · Severe 0.273637 · Proliferative DR 0.000403`.
UWF distribution: `No DR 0.023327 · Mild 0.881828 · Moderate 0.088653 · Severe 0.004967 · Proliferative DR 0.001224`.

Each response carries a machine-readable `preprocessing` provenance block (`values_verified_by_checkpoint`, `config_source`).

---

## 4. Grad-CAM status and verified results

✅ `backend/inference/gradcam.py` is committed and served by `POST /gradcam` (target layer `features.8`).

| | CFP | UWF |
|---|---|---|
| explained class | 2 Moderate | 1 Mild |
| native CAM grid | **(7, 7)** | **(16, 16)** |
| returned PNG | 224 × 224 | 512 × 512 |
| `gradients_are_nonzero` | ✅ true | ✅ true |
| `is_predicted_class` | ✅ true | ✅ true |
| latency | 1026.93 ms | 21925.83 ms |

⚠️ UWF Grad-CAM takes ~22 s on CPU (forward + backward at 512×512) — exceeds Render's free-tier timeout, as `render.yaml` itself warns.
❌ Overlay compositing onto the source photograph is still not implemented; the heatmap ships alone, in the model-input frame.

---

## 5. Frontend status and real API connection

✅ React 19 + TypeScript + Tailwind v4 + Vite 8. **No mock data anywhere** — every value rendered comes from a server response.

- `frontend/src/api/client.ts` is the only network code and implements all four calls: `predictImage`, `requestGradCam`, `fetchHealth`, `requestExplanation`.
- `App.tsx` calls `predictImage` then `requestGradCam` as a sequential follow-up sharing one monotonic `requestGeneration` guard, so grade and heatmap cannot describe different images. `target_class` is never sent (the backend explains its own prediction).
- UI: Analyze → `AnalysisResultCard` + `DrGradeScale`; Explainability → `GradCamCard` (renders the `data_uri` directly as `<img src>`); Model → `ModelPage` (static config plus the values the last run reported). The old `XaiPanel` placeholder was removed in `d72b545`, so the earlier UX contradiction is resolved.
- ✅ `npm run typecheck` → exit 0. ✅ `npm run lint` → exit 0 (both run just now, no rebuild).
- `frontend/dist` was last built 3 Oct 05:23; Vite dev server is listening on `:5173` (node PID 23324).
- ⚠️ `fetchHealth` and `requestExplanation` exist in the client but are **not called by any component** — `/explain` has no UI.
- ⚠️ Browser end-to-end verification was **not** performed in this session; frontend correctness rests on typecheck, lint and contract review against the live payloads.

---

## 6. Model / checkpoint status and the CPU-only constraint

✅ Both checkpoints exist on disk under `backend/models/` and load with `strict=True` (0 missing / 0 unexpected).

| | CFP | UWF |
|---|---|---|
| size | 15.65 MB | 46.40 MB |
| architecture | EfficientNet-B0 + `dr_head: Linear(1280,5)` + 7 × `lesion_heads: Linear(1280,1)` | stock `efficientnet_b0(num_classes=5)` |
| params | 4,022,920 | 4,013,953 |

✅ CPU-only, verified in the live venv: Python 3.10.11, `torch 2.14.0+cpu`, `torchvision 0.29.0+cpu`, `torch.version.cuda = None`, `torch.cuda.is_available() = False`. `DEVICE = "cpu"` is pinned in `main.py:99`; `cuda_available` is reported by `/health` for transparency only. Requirements pin the `+cpu` wheels via the PyTorch CPU extra index. The CFP model's 7 auxiliary lesion heads are never read or surfaced.

---

## 7. Architecture / inference files

✅ All eight modules are tracked and coherent: `exceptions.py`, `gradcam.py`, `model_architecture.py`, `model_loader.py`, `predictor.py`, `preprocessing.py` (+ `inspect_models.py`, `verify_architecture.py` CLI helpers). Single 5-class ICDR map in `predictor.DR_CLASSES`, shared by `/predict` and `/gradcam`; preprocessing is decode → EXIF transpose → RGB → bilinear resize → `[0,1]` → ImageNet normalize → `(1,3,S,S)`, with training augmentation deliberately off.

---

## 8. Warnings and caveats

1. ⚠️ **CFP preprocessing is an assumption, not checkpoint data.** `EfficientNetB0_CFP_final.pth` has no `config` key, so `image_size = 224` and the ImageNet mean/std are documented defaults. The API ships this verbatim: `values_verified_by_checkpoint: false`, `normalization_source: "ASSUMED (default, not in checkpoint)"`, plus a `config_source` warning string. If CFP actually trained at 512 (as UWF did), predictions degrade **silently** — EfficientNet is fully convolutional with GAP, so a wrong size produces no error. Override via `CFP_IMAGE_SIZE`.
2. ⚠️ **UWF is verified** from its checkpoint config (512, ImageNet statistics).
3. ⚠️ **No ground truth exists.** One unlabelled test image per modality; no accuracy, QWK, sensitivity or specificity claim is possible. `best_val_qwk` inside the checkpoints is a training-time value, not a measurement of this system.
4. ⚠️ **Label order is an assumption.** Neither checkpoint stores a class-index → grade map; the ICDR order is documented in `predictor.py`, not verifiable from the files.
5. ⚠️ **CFP CAM grid is only 7×7** — coarse for lesion localisation.
6. ⚠️ **`render.yaml` self-declares** that the free plan (0.1 CPU, 512 MB) is likely insufficient for `/predict` and `/gradcam`; `starter` is recommended.
7. ⚠️ Two uvicorn processes are bound to port 8000.

---

## 9. Known repo issues still remaining

| # | Issue | Severity |
|---|---|---|
| 1 | All deployment work is uncommitted: 6 modified files + 7 untracked, including the 62 MB of checkpoints the new `.gitignore` explicitly re-includes. A fresh clone of `main` still has **no weights**, so `/predict` and `/gradcam` would 503 | blocker |
| 2 | `VITE_API_URL` / `CORS_ORIGINS` wiring is uncommitted; a deployed frontend bundle must be **built** with `VITE_API_URL` set or it silently points at `localhost:8000` | blocker for deploy |
| 3 | Frontend has never been exercised in a real browser (largest untested gap) | bug risk |
| 4 | `/explain` is implemented and a key is configured (`gemini_configured: true`), but no component calls it and it was not verified this session | gap |
| 5 | `frontend/dist/index.html` + `retina.svg` are tracked while `dist/assets/` is ignored → the tracked artifact references JS/CSS that a fresh clone does not contain | repo hygiene |
| 6 | `backend/schemas/api.py` still carries a dead `PredictDevResponse` and a docstring claiming prediction schemas are "deliberately absent" | cleanup |
| 7 | Docs are stale: `README.md` still says Flutter and "not created yet"; `CURRENT_PROGRESS.md` and `RETINAGRADE_AI_PROGRESS_REPORT.md` cite HEAD `b093fde`; `RETINAGRADE_AI_CURRENT_STATUS.md` cites HEAD `ab1deeb` and a stale server | docs |
| 8 | No automated tests: `tests/` and `docs/` hold only `.gitkeep`, `pytest` is not installed. `backend/test_prediction.py`, `backend/test_models.py`, `inference/test_model_loading.py` are manual CLI scripts | blocker for release |
| 9 | No CI, no Dockerfile, no auth, no rate limiting | blocker for release |
| 10 | Duplicate uvicorn processes on port 8000 | cleanup |

---

## 10. Completed vs not completed

### ✅ Completed
1. Exact architecture reconstruction for both checkpoints (374/374 and 360/360 tensors, names/shapes/order).
2. Strict CPU-only loading, params and buffers asserted on CPU, lazy per-modality load.
3. Preprocessing with machine-readable verified-vs-assumed provenance.
4. Real end-to-end inference on real images for **both** modalities, full 5-class distribution.
5. Five live endpoints with one error envelope, CORS allow-list, no traceback leaks.
6. Grad-CAM: forward+backward at `features.8`, hook cleanup, non-zero-gradient guard, base64 PNG inside the JSON envelope.
7. Frontend wired to the real API for `/predict` + `/gradcam`, stale-response guard, no mock data, clean typecheck and lint.
8. Deployment scaffolding drafted (`render.yaml`, `runtime.txt`, `.python-version`, `.env.example` files, CPU-only requirements).

### ❌ Not completed
1. Committing any of the current working-tree changes (including the checkpoints).
2. Browser end-to-end verification.
3. Any UI for `/explain` (Gemini prose reports).
4. Grad-CAM overlay compositing onto the photograph.
5. Automated test suite / CI.
6. CFP training-resolution ground truth.
7. Documentation refresh (README + three stale reports).
8. PDF export, screening history, report storage, `GET /models` — all still absent by instruction.

---

## 11. Suggested next steps (not implemented)

1. **Commit the working tree**, checkpoints included — this is the single highest-value action and unblocks a fresh clone.
2. Verify the UI in a real browser against `:8000` / `:5173`: both modalities, plus the error states (bad modality, >25 MB, backend down, Grad-CAM failure).
3. Resolve the CFP input size from the training config and set `CFP_IMAGE_SIZE` (or bake the real value into `preprocessing.py`) to retire the silent-failure risk.
4. Add a `pytest` + `httpx` suite in the empty `tests/`: 5-class contract, Σ probabilities ≈ 1, CPU-only assertion, `strict=True` load, both error envelopes, Grad-CAM non-zero-gradient check.
5. Decide the `frontend/dist` policy: untrack it entirely, or un-ignore `dist/assets/`.
6. Add a `/explain` surface to the UI (call `fetchHealth` first, honour `gemini_not_configured`), or drop the unused client functions.
7. Refresh `README.md` and fold the three stale reports into this one.
8. Before deploying: set `CORS_ORIGINS` on the API and rebuild the frontend with `VITE_API_URL`, and move the backend off Render's free plan.

---

## Summary

The backend is **fully working and currently serving** all five endpoints from real checkpoints on CPU, with real CFP (Moderate, 0.703081) and UWF (Mild, 0.881828) predictions and working Grad-CAM on both modalities, all re-verified live during this report. The React frontend is redesigned, connected to the real API for `/predict` and `/gradcam`, and passes typecheck and lint. What is missing is **not** functionality: it is the commit, the browser test, the CFP size ground truth, `/explain` UI, tests/CI, and a documentation refresh — plus the standing caveat that CFP preprocessing is assumed rather than checkpoint-verified.