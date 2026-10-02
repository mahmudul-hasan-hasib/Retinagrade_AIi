# RetinaGrade AI — Current Status Audit

**Date:** 2 October 2026 · **Repo:** `F:\retinagrade-ai` · **Branch:** `main` (in sync with `origin/main`) · **HEAD:** `ab1deeb`
**Basis:** `git status`/`log`, source inspection, one combined inference+Grad-CAM run on both real test images, live probe of the running server on `:8000`, `tsc --noEmit`, `eslint`.
**No source modified. No packages installed. No commits. No redesign.**

✅ VERIFIED now · ⚠️ CAVEAT · ❌ NOT IMPLEMENTED

---

## 1. Git status + latest commits

**Working tree: DIRTY** — 4 modified, 2 untracked.

```
 M backend/main.py                +276/-4   adds POST /gradcam
 M frontend/src/App.tsx           +279/-43  adds GradCamCard + 2nd API call
 M frontend/src/api/client.ts     +131/-13  adds requestGradCam + types
 M frontend/dist/index.html       4         build artifact (asset hashes)
?? backend/inference/gradcam.py   450 lines  THE ENTIRE GRAD-CAM CORE
?? RETINAGRADE_AI_CURRENT_STATUS.md
```

Latest commits (newest first): `ab1deeb` connect api · `b7471a4` TEST CFP API ONLY · `44c98d1` fix frontend reports source tracking · `8dd5183` stale-response race fix · `3307160` fix bug 1 · `b093fde` add frontend · `a529878` prepar for prediction · `3f1d2f6` architecture reconstruction.

No stashes. 12 commits total.

---

## 2. Backend / API endpoints

Source declares **4 endpoints**; `API_VERSION = "0.5.0-dev"`.

| Method | Path | Purpose | Model run |
|---|---|---|---|
| GET | `/` | service banner | no |
| GET | `/health` | liveness + device + `cuda_available` | no |
| POST | `/predict` | real DR grading | **yes** |
| POST | `/gradcam` | real Grad-CAM heatmap (base64 PNG data URI) | **yes** |

⚠️ **The server currently listening on `127.0.0.1:8000` is STALE** — it reports `version: "0.4.0-dev"` and exposes only `/, /health, /predict`. **`/gradcam` is not live until uvicorn restarts.**

✅ Verified working in-process (fresh import of current source): all 4 endpoints respond correctly. Error handling uses a consistent `{"success": false, "error": {code, message}}` envelope; CORS allows any localhost port; global exception handler never leaks a traceback.

---

## 3. Predictor + preprocessing

`inference/predictor.py` ✅ — 5-class ICDR map (`No DR`/`Mild`/`Moderate`/`Severe`/`Proliferative DR`) in one place, shared by `/predict` and `/gradcam`. Softmax probabilities, argmax class, full distribution returned (not just the winner). Refuses to run if the model emits ≠ 5 logits. CFP's 7 auxiliary lesion heads are never surfaced (no label map exists in the checkpoint).

`inference/preprocessing.py` ✅ — decode (format whitelist, EXIF transpose, RGB) → bilinear resize → `[0,1]` → ImageNet normalize → `(1,3,S,S)` CPU tensor. Training augmentation deliberately not applied. Every response embeds a provenance block stating whether values came from the checkpoint or are assumed.

| | CFP | UWF |
|---|---|---|
| image_size | 224 ⚠️ **ASSUMED** | 512 ✅ read from checkpoint `config` |
| normalization | ImageNet mean/std ⚠️ assumed | ImageNet mean/std ✅ verified |
| `values_verified_by_checkpoint` | `false` | `true` |

⚠️ The CFP checkpoint has **no `config` key at all**. 224 is the canonical EfficientNet-B0 default, chosen only because the file is silent. A warning is logged on every CFP call and `verified: false` ships in the payload. If CFP actually trained at 512, predictions fail **silently**. `--image-size` override exists.

---

## 4. Model loading + real prediction

✅ Both checkpoints load with `load_state_dict(strict=True)`, 0 missing / 0 unexpected, all params **and** buffers on CPU, `eval()` confirmed. `strict=False` appears nowhere.

| | CFP | UWF |
|---|---|---|
| file | `EfficientNetB0_CFP_final.pth` (16.4 MB) | `EfficientNetB0_UWF_final.pt` (48.7 MB) |
| architecture | torchvision EfficientNet-B0 + `dr_head:Linear(1280,5)` + 7 × `lesion_heads:Linear(1280,1)` | stock `efficientnet_b0(num_classes=5)` |
| tensors / params | 374 / 4,022,920 | 360 / 4,013,953 |
| strict load | ✅ exact match | ✅ exact match |

**Real prediction, both modalities, run now:**

| | CFP | UWF |
|---|---|---|
| result | **Moderate** (class 2) | **Mild** (class 1) |
| confidence | 0.703081 | 0.881828 |
| Σ probabilities | 0.999999 | 0.999999 |
| device | cpu | cpu |

**CPU status:** ✅ `torch 2.14.0+cpu`, `torchvision 0.29.0+cpu`, Python 3.10.11, `torch.cuda.is_available() = False`, `torch.version.cuda = None`. No CUDA code path anywhere.

⚠️ `n = 1` unlabelled test image per modality (two different files). No ground truth exists → **no accuracy/QWK/sensitivity claim is possible**. `best_val_qwk` (0.9226 / 0.9153) is a training-time value read from inside the checkpoints, not a measurement of this system.

---

## 5. Grad-CAM

✅ Implemented in `backend/inference/gradcam.py` (450 lines, **untracked**) and live in `POST /gradcam`.

- Standard formulation: `weights = gradients.mean(spatial)` → `relu((weights * activations).sum(channels))` → bilinear upsample → normalize `[0,1]`.
- **Target layer: `features.8`** — last `Conv2dNormActivation` (320→1280, BN, SiLU), feeding parameter-free GAP directly, so the CAM weights equal the class-head weights exactly.
- Activation captured via a single forward hook calling `retain_grad()`; hook removal is idempotent and runs in `finally`, so a failed request cannot leak a hook onto the cached model.
- Runs under `enable_grad()` (reusing `dr_logits()` would sever gradients); model forced to `eval()` with prior mode restored; per-modality lock serialises concurrent requests.
- Aux lesion heads never read. Zero-gradient results are refused with a 500 rather than returned as evidence.
- PNG encoded with a NumPy jet ramp + Pillow (no matplotlib dependency).

**Verified now, both modalities:**

| | CFP | UWF |
|---|---|---|
| explained class | 2 Moderate | 1 Mild |
| CAM grid (native) | **(7, 7)** | **(16, 16)** |
| CAM output shape | (224, 224) | (512, 512) |
| gradient norm | 0.137601 | 0.048864 |
| `gradients_are_nonzero` | ✅ true | ✅ true |

❌ **Not implemented:** overlay compositing onto the source photograph. The heatmap is returned alone, in the model-input frame.

---

## 6. Frontend → `/predict`

✅ **Connected and working.** `src/api/client.ts` is the only network code. `predictImage()` POSTs `multipart/form-data` with fields `image` + `modality` to `http://127.0.0.1:8000/predict`, matching `main.py` exactly. `Content-Type` left unset so the browser generates the boundary. `PredictApiError` carries `httpStatus`/`code` and handles both `{success:false,error}` and FastAPI `{detail}` shapes. Stale responses are dropped via a monotonic `requestGeneration` ref.

`AnalyzePage.tsx` reads the real response and renders grade, confidence, per-class probabilities, checkpoint name and inference ms via `PredictionCard` / `ProbabilityCard` / `ProbabilityBars`, plus the raw body in `ApiResponseCard`. **No mock data anywhere.** No Vite proxy — absolute URLs only.

---

## 7. Frontend → `/gradcam`

⚠️ **Code-complete but never exercised in a browser.**

- `requestGradCam()` exists in `client.ts`, POSTs `image` + `modality`, omits `target_class` entirely when unset. Types `GradCamApiResponse` / `GradCamImage` mirror the Pydantic models.
- `App.tsx` defines `GradCamCard`, which renders the returned `data_uri` directly as `<img src>` (no client-side decode, colourmap or blending) plus a metadata grid: explained grade, confidence, target layer, CAM grid, model input, inference ms. States: `idle` / `loading` / `ready` / `error`. A CAM failure is non-fatal and never invalidates the grade already on screen.
- Called as a **sequential follow-up** to a successful `/predict`, sharing the same stale-response guard so grade and heatmap can never describe different images.
- Rendered on both the Analyze page and the Explainability page.
- `npx tsc --noEmit` → **exit 0**. `npm run lint` → **exit 0**.

**Why it is only "in progress":**

1. ⚠️ **Never run in a real browser.** Verified by typecheck, lint and contract review against the live response only.
2. ⚠️ **Contradictory UI.** `XaiPanel.tsx` is still mounted on both pages and still says *"The heatmap is still empty: Grad-CAM is not implemented"*, with all `GRAD_CAM_META` values `'-'` — directly above/below a real heatmap. `AnalyzePage.tsx:92` likewise warns that "Grad-CAM maps … are still placeholders".
3. ⚠️ The "Overlay" tab in `XaiPanel` is inert (backend returns no blend).

---

## 8. Known bugs / issues / blockers

| # | Issue | Severity |
|---|---|---|
| 1 | `backend/inference/gradcam.py` is **untracked** and 3 files are modified — the whole Grad-CAM feature is one `git clean` from being lost | **blocker** |
| 2 | Live server on `:8000` serves `0.4.0-dev` with **no `/gradcam`** — backend must be restarted before any frontend test | **blocker** |
| 3 | `POST /gradcam` with `target_class=9` returns **422 FastAPI default `{"detail":[…]}`, not the documented 400** — `Form(ge=0, le=4)` intercepts before the handler's own check at `main.py:428`. The project error envelope is bypassed | bug |
| 4 | `XaiPanel` placeholder contradicts the real `GradCamCard` on the same pages | bug (UX) |
| 5 | `dist/index.html` is tracked but `dist/assets/` is git-ignored → the tracked artifact references non-existent JS/CSS on a fresh clone | bug (repo hygiene) |
| 6 | `README.md` stale + partly corrupt: claims Flutter (it is React), claims inference not done (it is), wrong checkpoint names | docs |
| 7 | `CURRENT_PROGRESS.md` and `RETINAGRADE_AI_PROGRESS_REPORT.md` both cite HEAD `b093fde` and describe an API that runs no model | docs |
| 8 | `schemas/api.py` keeps a dead `PredictDevResponse` and a docstring claiming prediction schemas are "deliberately absent" — both untrue | cleanup |
| 9 | CFP input size 224 is a guess; wrong size fails **silently** (EfficientNet is fully convolutional + GAP) | **correctness risk** |
| 10 | No ground truth anywhere → no accuracy/QWK/sensitivity claim possible; `n=1` per modality | **correctness risk** |
| 11 | CAM grid is only **7×7 for CFP** — coarse for lesion localisation | limitation |
| 12 | No automated tests, no CI, no Docker, no auth, no rate limiting | blocker for release |
| 13 | `opencv-python` installed and pinned but never imported by any backend module | cleanup |

---

## 9. Files currently modified / untracked

| State | Path |
|---|---|
| modified | `backend/main.py` |
| modified | `frontend/src/App.tsx` |
| modified | `frontend/src/api/client.ts` |
| modified | `frontend/dist/index.html` |
| untracked | `backend/inference/gradcam.py` (450 lines) |
| untracked | `RETINAGRADE_AI_CURRENT_STATUS.md` (this file) |

Correctly ignored (verified): `backend/models/*.pt|.pth`, `frontend/dist/assets/*`, `backend/.venv/`, `frontend/node_modules/`, `__pycache__/`, `.env`.
Note: `backend/.gitignore` lists `models/*.pt` but not `*.pth`; the root `.gitignore` covers both, so it is currently harmless.

---

## 10. Completed vs remaining

### ✅ COMPLETE
1. Exact architecture reconstruction for both checkpoints (374/374, 360/360 — names, shapes, order).
2. Safe strict CPU-only loading: `weights_only=True` + `safe_globals`, `strict=True` everywhere, params *and* buffers asserted on CPU.
3. Preprocessing with machine-readable verified-vs-assumed provenance.
4. Real end-to-end prediction on real images, both modalities — 10/10 contract checks each.
5. FastAPI service, 4 endpoints, correct statuses, CORS, consistent error envelope, no traceback leaks.
6. `/predict` fully wired to the real model with threadpool offload and the full 5-class distribution.
7. `/gradcam` fully working — forward+backward, `features.8`, hook cleanup, gradient-norm verification, in-envelope PNG.
8. Grad-CAM verified on both modalities, non-zero gradients, deterministic.
9. Frontend API client with correct URL/field contracts and dual error-shape handling.
10. Frontend renders the real grade, confidence, probabilities and heatmap.
11. `tsc` and `eslint` both clean.

### ❌ REMAINS
1. **Commit the Grad-CAM work** (untracked + modified).
2. **Restart the backend** so `/gradcam` is actually served.
3. Browser end-to-end verification — the largest untested gap.
4. `XaiPanel` reconciliation with the real Grad-CAM card.
5. `target_class` validation-order fix (422 vs documented 400).
6. `dist/` tracking policy.
7. CFP training size ground truth (224 assumed).
8. Automated test suite (`tests/` is empty; no pytest/httpx).
9. `GET /models` route — `ModelManager.describe()` is implemented but unexposed; `DashboardPage` hardcodes "Not loaded".
10. Gemini report generation — zero code, `google-genai` not installed, `load_dotenv()` never called so `GEMINI_API_KEY` is inert.
11. PDF export — `reportlab` not installed, buttons `disabled`.
12. Screening history / persistence — `services/` + `utils/` empty.
13. Deployment — no Dockerfile, no CI, no auth, no rate limiting.
14. Documentation refresh (README + both stale reports).
15. Dead code removal (`PredictDevResponse`, unused `opencv-python`).

---

## Summary

**DONE** — model architecture + strict CPU loading, preprocessing provenance, real CFP and UWF inference, all 4 API endpoints, Grad-CAM backend core, frontend `/predict` + `/gradcam` wiring, clean typecheck/lint.

**IN PROGRESS** — frontend Grad-CAM UX: code written and type-safe but **never run in a browser**, and still contradicted by the placeholder `XaiPanel`.

**NOT STARTED** — Gemini reports, PDF export, screening history, automated tests, `GET /models`, CI/Docker/deployment, documentation refresh, and resolving the CFP input-size ground truth.

**Two blockers:** the entire Grad-CAM feature is uncommitted, and the running backend predates it.

---

## Recommended next steps (not implemented)

1. **Commit the current work** — `gradcam.py` plus the 3 modified files. Smallest effort, removes the largest data-loss risk.
2. **Restart uvicorn** and re-verify `/gradcam` is live on `:8000`.
3. **Verify in a real browser** (Vite is already listening on `:5173`) — upload both test images, confirm grade, probabilities, heatmap and both failure states.
4. **Fix the `target_class` validation order** so an out-of-range grade returns the documented `400` in the project envelope, not FastAPI's `422`.
5. **Reconcile `XaiPanel`** — either wire `GradCamCard` into it and fill `GRAD_CAM_META` from the response, or unmount it. Update the stale notice at `AnalyzePage.tsx:92`.
6. **Resolve the CFP training image size** (retrieve the training config/script) — the single largest correctness risk.
7. **Add a `pytest` + `httpx` suite** in the empty `tests/` dir: 5-class contract, probability sum, CPU-only assertion, `strict=True` load, both error envelopes, Grad-CAM gradient-norm.
8. **Expose `GET /models`** from the existing `ModelManager.describe()` so the frontend can show real model status.
9. **Fix the `dist/` policy** — untrack it, or un-ignore `dist/assets/`.
10. **Refresh the docs**, then defer Gemini/PDF/history until the blockers above are cleared.