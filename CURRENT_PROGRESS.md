# RetinaGrade AI — Current Progress Report

**Date:** 2 October 2026
**Repo:** `F:\retinagrade-ai` · **HEAD:** `b093fde "add frontend"` (7 commits) · **Working tree: DIRTY** — uncommitted frontend API work.
**Basis:** direct file inspection + `git status/diff` + live execution of `test_prediction.py` (CFP, UWF) + live HTTP probes of the running API + `npm run typecheck` / `npm run lint`.
**No code was modified in producing this report.**

Legend: ✅ VERIFIED (reproduced now) · ⚠️ ASSUMED/INFERRED · ❌ NOT IMPLEMENTED

---

## 1. Project structure ✅

```
retinagrade-ai/
├── backend/
│   ├── main.py                 206 lines — FastAPI skeleton, 3 endpoints
│   ├── test_models.py / test_prediction.py   CLI verification scripts
│   ├── inference/              ✅ the complete, working part
│   │   ├── model_architecture.py / verify_architecture.py
│   │   ├── model_loader.py / test_model_loading.py
│   │   ├── inspect_models.py, preprocessing.py, predictor.py, exceptions.py
│   ├── models/                 EfficientNetB0_CFP_final.pth, EfficientNetB0_UWF_final.pt (git-ignored)
│   ├── schemas/api.py          52 lines — 4 response schemas
│   ├── services/, utils/       EMPTY (only __init__.py)
│   └── test_images/            cfp/tr000001.jpg, uwf/tr000001.jpg (unlabelled)
├── frontend/                   React 19 + TS 5.9 + Vite 8 + Tailwind 4 — 33 src files
│   └── src/api/client.ts       ✅ NEW, uncommitted — first HTTP call
└── tests/, docs/               .gitkeep only
```

`tests/` holds no suite: the four `test_*.py` files are CLI scripts, not pytest. `pytest`/`httpx` are commented out in `requirements.txt`.

## 2. Model status (CFP / UWF) ✅

Both checkpoints load with `load_state_dict(strict=True)`, zero missing/unexpected tensors, CPU-only, `eval()`.

| | CFP | UWF |
|---|---|---|
| File | `EfficientNetB0_CFP_final.pth` | `EfficientNetB0_UWF_final.pt` |
| Architecture | torchvision EfficientNet-B0 + `dr_head: Linear(1280,5)` + 7 unnamed `lesion_heads: Linear(1280,1)` | stock `efficientnet_b0(num_classes=5)` |
| State-dict tensors | 374 | 360 |
| Strict load | ✅ 0 missing / 0 unexpected | ✅ 0 missing / 0 unexpected |
| Device / mode | `cpu` / `eval()` | `cpu` / `eval()` |
| Training-time `best_val_qwk` (inside file) | 0.9226 | 0.9153 |

Environment: Python 3.10.11, `torch 2.14.0+cpu`, `torchvision 0.29.0+cpu`, `torch.version.cuda = None`, `torch.cuda.is_available() = False`. `strict=False` appears nowhere. The 7 CFP auxiliary heads have **no label map in the file** ⚠️ — count is verified, meaning is not.

## 3. Real prediction test status ✅ (re-run now, both passed)

```
test_prediction.py --image test_images\cfp\tr000001.jpg --modality cfp
test_prediction.py --image test_images\uwf\tr000001.jpg --modality uwf
```

| | CFP | UWF |
|---|---|---|
| Result | **Moderate** (class 2) | **Mild** (class 1) |
| Confidence | 0.703081 | 0.881828 |
| Input size used | 224 ⚠️ **ASSUMED** | 512 ✅ read from checkpoint `config` |
| Σ probabilities | 0.999999 | 0.999999 |
| Output contract | **ALL CHECKS PASSED** | **ALL CHECKS PASSED** |

Caveats that must not be dropped: `n = 1` per modality, the two files are **different images** (SHA-256 and dimensions differ) so the Moderate/Mild difference is not a cross-modality disagreement; there is no ground truth anywhere, so no accuracy metric is possible; CFP's 224 rests on a guess and would fail silently if the model actually trained at 512.

## 4. Preprocessing status ✅ / ⚠️

`inference/preprocessing.py` — decode (format whitelist, EXIF transpose, RGB) → bilinear resize → `[0,1]` → ImageNet normalize → `(1,3,S,S)` CPU tensor. Training augmentation deliberately not applied.

- UWF: size 512 + ImageNet mean/std ✅ read from the checkpoint (`verified: true`).
- CFP: size 224 + ImageNet ⚠️ **ASSUMED** — the CFP checkpoint has no `config` key at all. A warning is logged on every CFP load and `values_verified_by_checkpoint: false` is embedded in the payload.
- `--image-size` override exists and always clears the `verified` flag.

## 5. FastAPI status ✅ (service runs, inference NOT wired)

Live probes against a server already listening on `127.0.0.1:8000` (PID 18512):

| Method | Path | Live result |
|---|---|---|
| GET | `/` | `{"app":"RetinaGrade AI","status":"running"}` |
| GET | `/health` | `{"status":"healthy","device":"cpu","cuda_available":false}` |
| POST | `/predict` (valid cfp) | `{"success":true,"message":"Image received","modality":"CFP"}` |
| POST | `/predict` (modality=mri) | HTTP 400 `unsupported_modality` |

`/openapi.json` lists exactly `/`, `/health`, `/predict`.

❌ **`POST /predict` runs no model.** It performs 5 checks (modality, file present, non-empty, ≤25 MB) and returns a placeholder. `inference.predictor.predict` is never imported by `main.py`. No lifespan/preload hook, no `GET /models`, no prediction response schemas (`schemas/api.py` deliberately omits them). CORS + global exception handler are present.

## 6. Frontend status ✅ (uncommitted work in progress)

- 33 source files; 6 views; no router; all result fields are `null`/placeholder; **no mock prediction data exists anywhere**.
- ✅ `npm run typecheck` (`tsc -b --noEmit`) — **0 errors**.
- ✅ `npm run lint` (`eslint .`) — **0 problems**.
- The Analyze action is no longer a no-op state flip: `App.tsx:handleAnalyze` is now `async` and drives `PredictionStatus = 'idle' | 'analyzing' | 'analyzed' | 'error'`, with a new `ApiResponseCard` rendering the raw JSON body and HTTP status.

## 7. Frontend ↔ backend connection ⚠️ PARTIAL

- ✅ `frontend/src/api/client.ts` exists and is the only network code: `fetch(`${BACKEND_BASE_URL}/predict`)` with `multipart/form-data` (`image` + `modality`), a `PredictApiError` carrying `httpStatus`/`code`, and error handling for both `{success:false,error:{...}}` and FastAPI's `{detail:...}`.
- ✅ Field names and the `BACKEND_BASE_URL = 'http://127.0.0.1:8000'` target match `main.py`, and CORS permits Vite's `127.0.0.1:5173`.
- ⚠️ **Not verified end-to-end in a browser** — the client was verified by typecheck/lint and by contract review against the live API response, not by an actual browser round-trip.
- ⚠️ No Vite dev proxy (`vite.config.ts` is 12 lines, `server` only). Direct absolute-URL calls work; a relative `/api` call would not.
- ❌ Because the endpoint returns no grade, the clinical panels (grade, confidence, probabilities, Grad-CAM, AI report) still stay placeholders. The connection carries a validation receipt, not a prediction.

## 8. What is working now ✅

1. Both checkpoints load strictly on CPU with exact architecture reconstruction.
2. Real end-to-end prediction on real images, both modalities, contract 10/10 PASS.
3. Preprocessing with machine-readable provenance (verified vs assumed).
4. FastAPI service up with 3 endpoints, real status codes, CORS, safe error envelope.
5. A complete, accessible, lint-clean, type-clean UI shell.
6. A working frontend → `POST /predict` HTTP call that surfaces the server's answer verbatim, including errors.
7. No fabricated data anywhere in the stack.

## 9. What is not implemented yet ❌

| # | Item | Evidence |
|---|---|---|
| 1 | **API-side inference** — `/predict` never calls the model | `main.py:151-203`; `predictor` not imported by `main.py` |
| 2 | Prediction response schemas | `schemas/api.py` — deliberately absent |
| 3 | Model preload / `GET /models` | no lifespan hook; `manager.describe()` has no route |
| 4 | Rendering a real grade/confidence in the UI | `AnalyzePage` passes `null` for every result field |
| 5 | Grad-CAM / saliency | zero backend code; `XaiPanel` renders an "not implemented" placeholder |
| 6 | Gemini AI report | `google-genai` not installed; `load_dotenv` never called, so `.env` is inert; button `disabled` |
| 7 | PDF export | `reportlab` not installed; `services/` empty |
| 8 | History / persistence, auth, rate limiting, deployment config | `services/` + `utils/` empty; no Dockerfile/CI |
| 9 | Automated test suite | `tests/` = `.gitkeep`; no pytest, no frontend test tooling, no CI |
| 10 | Dataset ground truth → any accuracy/QWK/sensitivity claim | 2 unlabelled test images |
| 11 | Frontend routing, `loading`/`error` panels beyond the analyze button, history & settings views | no `react-router`; `UnavailablePage` |

⚠️ Two unresolved inference-level unknowns: the CFP input size (224 assumed) and the class-index → ICDR-grade map (documented inference, not stored in either checkpoint).

## 10. Next recommended single step

**Wire `inference.predictor.predict()` into `POST /predict` and return its result.** One step, in this order: add a lifespan hook calling `manager.load_all()` → add `PredictResponse` / `PredictionResultModel` schemas mirroring `PredictionResult.to_dict()` → replace the placeholder body of `predict_validation_only` with a real `predict(...)` call and map `InferenceError` subclasses to HTTP status codes.

This is the only remaining gap between a verified working engine and a usable service: inference, preprocessing, provenance and the HTTP layer all exist and are tested, and the frontend already displays whatever JSON the endpoint returns, so no frontend work is needed before a real grade appears on screen.

Do it **after** resolving the CFP input size question (or ship with the existing `verified: false` provenance flag visible in the response) — a 224 prediction on a 512-trained model fails silently.
