# RetinaGrade AI — Project Progress Report

**Report date:** 1 October 2026
**Repository:** `F:\retinagrade-ai`
**Git:** 7 commits, HEAD = `b093fde "add frontend"`, working tree clean
**Environment:** Windows, `backend\.venv` = CPython 3.10.11, torch 2.14.0+cpu
**Basis of this report:** direct inspection of the repository on disk, plus live execution of the four verification scripts that ship with the project and a live HTTP probe of the running FastAPI service. Every number below was either read out of a file in the repository or produced by a command listed in **Appendix A**. No result in this report is estimated, extrapolated, or carried over from an earlier claim.

---

## Evidence legend

| Marker | Meaning |
|---|---|
| ✅ **VERIFIED** | Reproduced in this session by a command in Appendix A |
| ⚠️ **ASSUMED** | Not stated by any checkpoint; a documented default, flagged in code as unverified |
| ❌ **NOT IMPLEMENTED** | No code in the repository performs this |
| 📄 **DOCUMENTED ONLY** | Described in a comment/README/schema but no executable path exists |

---

## 1. Project structure

```
retinagrade-ai/
├── README.md                              2,412 B  (partly stale — see §12.9)
├── .gitignore
├── RETINAGRADE_AI_PROGRESS_REPORT.md      this file
│
├── backend/                               FastAPI service, CPU-only
│   ├── .venv/                             CPython 3.10.11 virtualenv (untracked)
│   ├── .env                               APP_ENV / API_HOST / API_PORT / GEMINI_API_KEY (untracked)
│   ├── .gitignore
│   ├── requirements.txt                   30 lines, pinned CPU-only wheels
│   ├── main.py                            206 lines — API skeleton, 3 endpoints
│   ├── test_models.py                     164 lines — CLI checkpoint verification
│   ├── test_prediction.py                 305 lines — CLI real-image inference test
│   ├── models/
│   │   ├── .gitkeep                       (checkpoints are git-ignored)
│   │   ├── EfficientNetB0_CFP_final.pth   16,408,231 B  (untracked)
│   │   └── EfficientNetB0_UWF_final.pt    48,652,882 B  (untracked)
│   ├── inference/                         ← the completed part of the project
│   │   ├── __init__.py                    0 B
│   │   ├── exceptions.py                  1,391 B   domain errors
│   │   ├── inspect_models.py              20,146 B  read-only checkpoint inspection
│   │   ├── model_architecture.py          10,574 B  evidence-based architecture definition
│   │   ├── verify_architecture.py         15,125 B  architecture ↔ checkpoint comparison
│   │   ├── model_loader.py                25,187 B  strict CPU weight loading + ModelManager
│   │   ├── test_model_loading.py          12,757 B  weight-loading verification CLI
│   │   ├── preprocessing.py               12,423 B  decode → resize → normalise
│   │   └── predictor.py                    7,590 B  DR grading (NOT called by the API)
│   ├── schemas/
│   │   ├── __init__.py                    0 B
│   │   └── api.py                         52 lines — 5 Pydantic schemas
│   ├── services/__init__.py               0 B — EMPTY placeholder
│   ├── utils/__init__.py                  0 B — EMPTY placeholder
│   └── test_images/
│       ├── cfp/tr000001.jpg               657,724 B — 2077×2077 RGB JPEG
│       └── uwf/tr000001.jpg               451,362 B — 2600×2048 RGB JPEG
│
├── frontend/                              React 19 + TypeScript + Vite + Tailwind SPA
│   ├── package.json / package-lock.json   33 lines / 115,492 B
│   ├── vite.config.ts                     12 lines
│   ├── tsconfig.json / .app.json / .node.json
│   ├── eslint.config.js / index.html / public/retina.svg
│   ├── dist/                              4 files, prior production build
│   └── src/                               32 files (31 .ts/.tsx + 1 .css)
│       ├── App.tsx  main.tsx  types.ts  index.css
│       ├── pages/            AnalyzePage, DashboardPage, ExplainabilityPage,
│       │                     ReportsPage, UnavailablePage
│       ├── components/
│       │   ├── analyze/      ImageDropzone, ImagePreview, ModalitySelector
│       │   ├── layout/       AppShell, Brand, SidebarContent, SidebarNav, TopBar
│       │   ├── results/      PredictionCard, ProbabilityCard, ProbabilityBars
│       │   ├── xai/          XaiPanel
│       │   ├── reports/      AiReportCard
│       │   ├── ui/           Badge, Button, Card, EmptyState, Feedback, Placeholder
│       │   └── icons/        index.tsx
│       ├── data/clinical.ts
│       ├── hooks/useImageSelection.ts
│       └── lib/format.ts
│
├── tests/.gitkeep                         0 B — EMPTY (no automated test suite)
└── docs/.gitkeep                          0 B — EMPTY
```

**Notable structural facts**

- ✅ `backend/services/` and `backend/utils/` contain **only empty `__init__.py` files**. There is no Gemini client, no history store, no PDF writer.
- ✅ `tests/` (repo root) contains only `.gitkeep`. The four `test_*.py` files are **CLI verification scripts, not a pytest suite** — `pytest` is not installed (`requirements.txt:30`, commented out).
- ✅ `docs/` contains only `.gitkeep`.
- ✅ Both checkpoints are present on disk but git-ignored (`.gitignore:26-30`).
- ✅ `frontend/dist/` exists **and is tracked in git** despite the `dist/` rule at `.gitignore:23` (see §12.8).

---

## 2. Tech stack

### 2.1 Backend — pinned in `backend/requirements.txt`, resolved versions read from `backend/.venv/`

| Package | Pinned | Actually installed | Source |
|---|---|---|---|
| Python | 3.10 / 3.11 | **3.10.11** | ✅ venv |
| torch | `2.14.0+cpu` | **2.14.0+cpu** | ✅ |
| torchvision | `0.29.0+cpu` | **0.29.0+cpu** | ✅ |
| fastapi | `0.141.1` | 0.141.1 | `requirements.txt:17` |
| uvicorn[standard] | `0.54.0` | 0.54.0 | ✅ (`uvicorn-0.54.0.dist-info`) |
| python-multipart | `0.0.32` | 0.0.32 | `requirements.txt:19` |
| python-dotenv | `1.2.3` | 1.2.3 | `requirements.txt:20` |
| numpy | `2.2.6` | **2.2.6** | ✅ |
| pillow | `12.3.0` | 12.3.0 | `requirements.txt:24` |
| opencv-python | `5.0.0.93` | 5.0.0.93 | `requirements.txt:25` |

Installed as `uvicorn[standard]` extras: `websockets 16.1.1`, `watchfiles 1.3.0`, `httptools`, `uvloop`, `python-dotenv`, `PyYAML`, `colorama`.

- ✅ `torch 2.14.0+cpu` and `torchvision 0.29.0+cpu` are **CPU-only builds** (`torchvision-0.29.0+cpu.dist-info`). No CUDA wheel is present.
- ✅ **`opencv-python` is installed but never imported.** A grep for `import cv2` across `backend/**/*.py` returns zero matches. Unused dependency.
- ✅ **`python-dotenv` is installed but never called.** A grep for `load_dotenv` returns zero matches, so `backend/.env` is **not actually loaded** at runtime. The `GEMINI_API_KEY` / `GEMINI_MODEL` keys it defines are inert.
- ❌ **Not installed** (`requirements.txt:27-30`, commented out): `google-genai`, `reportlab`, `pytest`, `httpx`.
- ✅ `timm` is **not installed** and is not needed — the checkpoints are torchvision EfficientNet, not timm (see §3).

### 2.2 Frontend — `frontend/package.json`

| Package | Version | Notes |
|---|---|---|
| react / react-dom | `^19.3.0` | the only runtime dependencies |
| typescript | `~5.9.3` | |
| vite | `^8.3.1` | dev server `127.0.0.1:5173` (`vite.config.ts:8-11`) |
| tailwindcss + `@tailwindcss/vite` | `^4.3.3` | Tailwind v4 via plugin, no `tailwind.config.js` |
| eslint + typescript-eslint + plugins | `^10.11.0` / `^8.71.0` | `npm run lint` available |
| `@types/react`, `@types/react-dom` | `^19.3.0` | |

- ✅ **No router** (`react-router` absent from `package.json` and from `package-lock.json`).
- ✅ **No HTTP client** (`axios`, `swr`, `@tanstack/*` all absent).
- ✅ `node_modules` **is installed** — 121 packages present, so `npm run dev` / `build` / `typecheck` / `lint` are runnable.
- ✅ `npm run typecheck` (`tsc -b --noEmit`) **exits clean, no errors** — verified this session.
- ✅ The project description in `package.json:6` self-declares: *"React + TypeScript + Tailwind CSS dashboard (UI only, no model integration)."*

### 2.3 What the models were trained with (recorded inside the UWF checkpoint)

| Field | Value |
|---|---|
| `pytorch_version` | `2.10.0+cu128` |
| `torchvision_version` | `0.25.0+cu128` |
| `cuda_version` | `12.8` |
| `device` | `cuda` |
| `gpu_count` | `2` |
| `amp` | `true` |
| `optimizer` / `scheduler` / `loss` | `AdamW` / `CosineAnnealingLR` / `CrossEntropyLoss(weight=None)` |
| `batch_size` / `epochs` / `early_stopping_patience` | `4` / `20` / `5` |
| `augmentation` | `RandomHorizontalFlip(p=0.5) + RandomRotation(10)` |
| `seed` | `42` |

The training stack was CUDA-based. **None of it is required or installed here** — inference runs on a CPU-only PyTorch build (§5).

---

## 3. CFP model — architecture and weight verification

✅ **VERIFIED by A1, A2, A3, A4.**

### 3.1 Checkpoint file facts (read-only inspection, A1)

| Property | Value |
|---|---|
| File | `backend/models/EfficientNetB0_CFP_final.pth` |
| Size | 16,408,231 bytes |
| Python type after `torch.load` | `builtins.dict` (checkpoint dict, **not** a bare state_dict, **not** a pickled `nn.Module`) |
| Top-level keys | `best_val_qwk`, `epoch`, `history`, `model_state_dict` |
| `config` key | ❌ **ABSENT** — this is the single most consequential fact in the whole project (see §6) |
| Tensors in `model_state_dict` | **374** |
| Total tensor elements | **4,064,985** |
| dtypes | `torch.float32`, `torch.int64` |
| Devices after `map_location='cpu'` | `['cpu']` |
| `epoch` | `12` |
| `best_val_qwk` | `0.9225706931411989` (`np.float64`) |
| `history` | `list`, length 12; entry fields: `epoch, lr, train_dr_loss, train_lesion_loss, train_total_loss, val_Accuracy, val_Macro_F1, val_QWK, val_dr_loss, val_lesion_loss, val_total_loss` |
| File unchanged by inspection | ✅ `True` |

The `history` field names confirm this checkpoint came from a **multi-task** run: separate `train_dr_loss` / `train_lesion_loss` and `val_dr_loss` / `val_lesion_loss` terms.

### 3.2 Reconstructed architecture

Defined once in `backend/inference/model_architecture.py:165` as `CfpEfficientNetB0`:

| Key prefix | Tensors | Module |
|---|---|---|
| `features.*` | 358 | `torchvision.models.efficientnet_b0(weights=None).features` — MBConv stages 0–8 |
| `avgpool` | 0 (parameter-free) | `nn.AdaptiveAvgPool2d(1)` |
| `dr_head.*` | 2 | `nn.Linear(1280, 5)` — bare Linear, **not** a Sequential, so **no dropout** |
| `lesion_heads.0…6.*` | 14 | `nn.ModuleList` of 7 × `nn.Linear(1280, 1)` — binary auxiliary heads |

`forward()` returns `(dr_logits, auxiliary_binary_logits)` — `model_architecture.py:193`.

### 3.3 Why torchvision and not timm — ✅ proven from the keys, not assumed

`model_architecture.py:12-40` records the argument, and A2 re-checks it on the real files:

- `features.0.0.weight` = `(32, 3, 3, 3)` → stem `Conv2d(3, 32, k=3, s=2, p=1)` → **3-channel RGB input**.
- `features.8.0.weight` = `(1280, 320, 1, 1)` → final conv → **1280-d pooled features**.
- Trunk stages present: `[0,1,2,3,4,5,6,7,8]`; 81 4-D conv layers; 64 Squeeze-Excitation tensors.
- 49 `num_batches_tracked` int64 buffers = exactly torchvision B0's BatchNorm count.
- **Zero** `blocks.*`, `conv_head.*` or `bn1.*` keys → **timm is ruled out**.
- A2 reports `timm: not installed (and not required)`.

### 3.4 Architecture ↔ checkpoint comparison (A2)

```
CFP  ->  EfficientNetB0_CFP_final.pth
  total parameters   : 4,022,920
  model tensors      : 374
  checkpoint tensors : 374
  checkpoint 'features.*' tensors : 358 (expected 358)
  names identical      : True
  shapes identical     : True
  key order identical  : True
  missing keys         : 0
  unexpected keys     : 0
  shape mismatches     : 0
  spot-checks:
    OK  features.0.0.weight     model=(32,3,3,3)      checkpoint=(32,3,3,3)
    OK  features.8.0.weight     model=(1280,320,1,1)  checkpoint=(1280,320,1,1)
    OK  dr_head.weight          model=(5,1280)         checkpoint=(5,1280)
    OK  lesion_heads.0.weight   model=(1,1280)         checkpoint=(1,1280)
    OK  lesion_heads.6.weight   model=(1,1280)         checkpoint=(1,1280)
  VERDICT: EXACT MATCH - architecture CAN be reconstructed exactly
```

### 3.5 Weight loading (A3)

| Check | Result |
|---|---|
| Loaded via `ModelManager` | ✅ YES |
| `load_state_dict(strict=True)` | ✅ **succeeded — 374 tensors from `model_state_dict`, 0 missing, 0 unexpected** |
| Head layout | `multi_head` (auto-detected from the keys, `model_loader.py:273`) |
| `model.training` | `False` (eval mode) ✅ |
| Device | `cpu` ✅ |
| Parameters *and* buffers on CPU | ✅ verified — 0 off-CPU |
| `dr_head` | `nn.Linear(1280, 5)` ✅ |
| `lesion_heads` | `nn.ModuleList` with exactly 7 × `nn.Linear(1280, 1)` ✅ |
| Classification outputs | `5` ✅ |
| Zero-tensor forward probe | ✅ returns `(1, 5)` and `(1, 7)` |

**`strict=True` is never relaxed anywhere in the project.** `model_loader.py:428-446` documents and enforces this: a partial load would leave randomly initialised weights in the network and produce confident nonsense.

**Parameter count, reconciled.** Two different totals appear in this repository, and both are correct — they count different things:

| Number | Counted by | Composition |
|---|---|---|
| **4,022,920** | `test_models.py` / `test_model_loading.py` (live run) | `nn.Module.parameters()` only — the 49 BatchNorm `running_mean`/`running_var` buffers are *not* included |
| **4,064,985** | `inspect_models.py` (live run) and the `model_architecture.py:43` docstring | every tensor in the state_dict, i.e. parameters **+ 42,065 BatchNorm buffers** |

✅ Verified by direct summation: `4,022,920 + 42,065 = 4,064,985`. The two figures are not a discrepancy; they are two different definitions. **Report either one, but name which definition you used.**

### 3.6 Auxiliary heads — ⚠️ meaning is UNKNOWN

- ✅ **Count verified:** 7 heads.
- ❌ **Semantics unknown and unknowable from the file.** The CFP checkpoint stores no label map for `lesion_heads.0…6`.
- `model_architecture.py:59-61`: *"their ordering and meaning are unknown. They stay unnamed raw outputs and must never be presented as named clinical findings."*
- `test_model_loading.py:232-235` repeats the warning in its output.
- `predictor.py:158-167` will return them **only** if the caller passes `include_raw_auxiliary=True`, and then only as `aux_head_0 … aux_head_6` with an explicit `auxiliary_note` disclaiming clinical meaning.

---

## 4. UWF model — architecture and weight verification

✅ **VERIFIED by A1, A2, A3, A4.**

### 4.1 Checkpoint file facts (A1)

| Property | Value |
|---|---|
| File | `backend/models/EfficientNetB0_UWF_final.pt` |
| Size | 48,652,882 bytes |
| Python type | `builtins.dict` (checkpoint dict) |
| Top-level keys | `best_val_qwk`, `config`, `epoch`, `history`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict` |
| Tensors in `model_state_dict` | **360** |
| Total tensor elements | **4,056,018** |
| dtypes / devices | `float32` + `int64` / `['cpu']` |
| `epoch` | `7` |
| `best_val_qwk` | `0.9152520275167323` (Python `float`) |
| `history` | `dict`, 9 fields: `epoch, epoch_time_minutes, learning_rate, train_loss, val_loss, val_accuracy, val_macro_f1, val_macro_auc, val_qwk` |
| `config` | `dict`, 27 keys (full dump in §4.5) |
| File unchanged by inspection | ✅ `True` |

This file also carries **optimizer / scheduler / AMP-scaler state** — the CFP file does not. That is why the two files differ in size (48.6 MB vs 16.4 MB) despite near-identical weight counts.

### 4.2 Reconstructed architecture

`model_architecture.py:217` → `create_uwf_model()` returns stock `torchvision.models.efficientnet_b0(weights=None, num_classes=5)`. No custom class needed.

| Key prefix | Tensors | Module |
|---|---|---|
| `features.*` | 358 | identical EfficientNet-B0 trunk (same factory as CFP) |
| `classifier.0` | 0 | `nn.Dropout(p=0.2)` — parameter-free, hence invisible in the state_dict |
| `classifier.1.*` | 2 | `nn.Linear(1280, 5)` |

`classifier` has exactly one tensor-bearing index, `1`; index `0` holds no tensors. That is the stock `nn.Sequential(nn.Dropout(p), nn.Linear(...))` signature, which is why A2 concludes the file *is* `efficientnet_b0(num_classes=5)`. `classifier.1.weight` = `(5, 1280)` confirms `num_classes: 5`.

### 4.3 Architecture ↔ checkpoint comparison (A2)

```
UWF  ->  EfficientNetB0_UWF_final.pt
  total parameters   : 4,013,953
  model tensors      : 360
  checkpoint tensors : 360
  checkpoint 'features.*' tensors : 358 (expected 358)
  names identical      : True
  shapes identical     : True
  key order identical  : True
  missing keys         : 0
  unexpected keys     : 0
  shape mismatches     : 0
  spot-checks:
    OK  features.0.0.weight     model=(32,3,3,3)      checkpoint=(32,3,3,3)
    OK  features.8.0.weight     model=(1280,320,1,1)  checkpoint=(1280,320,1,1)
    OK  classifier.1.weight     model=(5,1280)         checkpoint=(5,1280)
  VERDICT: EXACT MATCH - architecture CAN be reconstructed exactly
```

### 4.4 Weight loading (A3)

| Check | Result |
|---|---|
| Loaded via `ModelManager` | ✅ YES |
| `load_state_dict(strict=True)` | ✅ **succeeded — 360 tensors from `model_state_dict`, 0 missing, 0 unexpected** |
| Head layout | `single_head` (auto-detected) |
| `model.training` | `False` (eval mode) ✅ |
| Device | `cpu` ✅ |
| Parameters *and* buffers on CPU | ✅ verified |
| `classifier` | `nn.Sequential`, length 2, `classifier[0]` is `nn.Dropout(p=0.2)`, `classifier[1]` is `nn.Linear(1280, 5)` ✅ |
| Classification outputs | `5` ✅ |
| Zero-tensor forward probe | ✅ returns a single `(1, 5)` tensor |

Parameter reconciliation (same two definitions as §3.5): **4,013,953** parameters, **+ 42,065** BatchNorm buffers = **4,056,018** state_dict elements. ✅ Verified by direct summation.

### 4.5 The UWF `config` — read from the file, but ⚠️ internally inconsistent

Full 27-key dump reproduced from A4:

```json
{
  "experiment": "E1_CFP_EfficientNetB0",
  "dataset": "CFP",
  "model": "EfficientNet-B0",
  "seed": 42,
  "image_size": 512,
  "num_classes": 5,
  "batch_size": 4,
  "num_workers": 0,
  "epochs": 20,
  "early_stopping_patience": 5,
  "validation_fraction": 0.1,
  "loss": "CrossEntropyLoss(weight=None)",
  "optimizer": "AdamW",
  "learning_rate": 0.0001,
  "weight_decay": 0.0001,
  "scheduler": "CosineAnnealingLR",
  "augmentation": "RandomHorizontalFlip(p=0.5) + RandomRotation(10)",
  "normalization": "ImageNet mean/std",
  "pretrained": "ImageNet",
  "amp": true,
  "data_parallel": false,
  "channels_last": false,
  "device": "cuda",
  "gpu_count": 2,
  "pytorch_version": "2.10.0+cu128",
  "torchvision_version": "0.25.0+cu128",
  "cuda_version": "12.8"
}
```

**⚠️ Two stale fields, and they matter.** This is the UWF model file, yet `config["experiment"] = "E1_CFP_EfficientNetB0"` and `config["dataset"] = "CFP"`. Both are labels the training script wrote and never updated. `preprocessing.py:43-51` flags this explicitly. There is a third inconsistency: `history` contains `val_macro_auc`, which is not a meaningful metric for a 5-class `CrossEntropyLoss` ordinal model — another stale-script artifact.

**Consequence:** `image_size` and `normalization` *are* read from this dict (they are the only preprocessing evidence in either file), but the dict cannot be trusted wholesale. This is exactly why the CFP values are treated as assumptions rather than inferred from the UWF config (§6).

### 4.6 Head-count cross-check

The CFP model has 7 extra binary heads relative to UWF. The two checkpoints' parameter counts differ by exactly `7 × 1,281 = 8,967` (`4,022,920 − 4,013,953 = 8,967`), which is the fingerprint of 7 × `Linear(1280, 1)` with bias. ✅ Consistent and independently corroborating.

---

## 5. CPU-only verification

✅ **VERIFIED by A3, A4, A5.** CPU-only is not a convention here; it is asserted in code and checked by the scripts.

| Evidence | Value | Where |
|---|---|---|
| `torch.__version__` | `2.14.0+cpu` | A3, A4 |
| `torchvision.__version__` | `0.29.0+cpu` | A4 |
| `torch.version.cuda` | **`None`** (CPU-only build) | A3, A4 |
| `torch.cuda.is_available()` | **`False`** | A3, A4, A5 |
| Installed CUDA wheels | none — `torchvision-0.29.0+cpu.dist-info` | A7 |
| `CPU_DEVICE` constant | `torch.device("cpu")` | `model_loader.py:84` |

**Enforcement mechanisms in the code (all confirmed by reading):**

1. `safe_load_checkpoint` uses `torch.load(path, map_location="cpu", weights_only=True)` inside a `torch.serialization.safe_globals([...])` block — `model_loader.py:238-240`. Unsafe unpickling is never used.
2. A checkpoint that turns out to be a **fully pickled `nn.Module`** is **rejected** (`ArchitectureMismatchError`) rather than accepted, because an unverified network cannot be checked — `model_loader.py:247-253`.
3. `model.to(CPU_DEVICE)` then `model.eval()` — `model_loader.py:450-451`.
4. **Every parameter *and* every buffer** is iterated and asserted to be on CPU. `model_loader.py:394-459` and `test_model_loading.py:66-68`. Checking `parameters()` alone would leave all BatchNorm `running_mean`/`running_var` unverified, so buffers are checked explicitly.
5. The structural forward probe is created on `device=CPU_DEVICE` — `model_loader.py:503`.
6. `LoadedModel.dr_logits` / `binary_logits` force `.to(CPU_DEVICE)` inside `torch.no_grad()` — `model_loader.py:197-212`.
7. `test_model_loading.verify_cpu_only()` **asserts** `cuda_available is False` and fails the run if CUDA appears — `test_model_loading.py:259`.
8. `predictor.py:178` hardcodes `device=str(CPU_DEVICE)` in the result payload.
9. `main.py:56` sets `DEVICE = "cpu"` with the comment *"CPU is mandatory. CUDA is reported but never used."*
10. `test_prediction.py:84-88` re-checks `torch.cuda.is_available()` at prediction time and folds it into the contract assertions.

**Train/inference split:** the checkpoints were trained on CUDA 12.8 across 2 GPUs with AMP (`config` in the UWF file). Their tensors are `float32` and deserialise onto the CPU with `map_location="cpu"`. No CUDA package is required or installed.

**A2 explicitly reports `timm: not installed (and not required)`** and `CUDA available: False`.

---

## 6. Preprocessing status

Implemented in `backend/inference/preprocessing.py`. The pipeline is:

```
bytes / PIL / file-like
  → decode_image()      Pillow open, format whitelist, exif_transpose, convert("RGB")
  → TF.resize(S, S)     bilinear, antialias=True
  → TF.to_tensor()      float32 in [0, 1]
  → TF.normalize()      (x - mean) / std
  → unsqueeze(0)        → (1, 3, S, S) CPU tensor
```

- ✅ Accepted formats: `JPEG, PNG, BMP, TIFF, WEBP` (`preprocessing.py:77`).
- ✅ `preprocessing.py:53-55`: training augmentation (`RandomHorizontalFlip(p=0.5) + RandomRotation(10)`) is **deliberately not applied at inference**, and the payload reports `training_augmentation_applied: false`.
- ✅ Decode failures raise `InvalidImageError` with specific messages for empty input, unsupported format, corrupt file, and wrong Python type.

### 6.1 UWF preprocessing — ✅ VERIFIED from the checkpoint

| Property | Value | Provenance |
|---|---|---|
| `image_size` | **512** | ✅ read from `config["image_size"]` at load time (`preprocessing.py:164-183`) |
| `normalization` | **ImageNet mean/std** — mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)` | ✅ `config["normalization"] = "ImageNet mean/std"`, `config["pretrained"] = "ImageNet"` |
| `color_space` | `RGB` | ✅ explicit `convert("RGB")` |
| `verified` flag | **`True`** | ✅ `preprocessing.py:115` |
| Runtime report | `values_verified_by_checkpoint: true`, `normalization_source: "checkpoint config"` | ✅ A5 |

`CHECKPOINT_IMAGE_SIZE = 512` (`preprocessing.py:85`). The live run reports `Config source: read from checkpoint config (image_size=512)`.

### 6.2 CFP preprocessing — ⚠️ ASSUMED / UNVERIFIED

**The CFP checkpoint stores no `config` key at all.** Its top-level keys are exactly `best_val_qwk`, `epoch`, `history`, `model_state_dict` (✅ A1). There is no image size, no normalization, no colour mode — nothing.

| Property | Value used | Status |
|---|---|---|
| `image_size` | **224** | ⚠️ **ASSUMED** |
| `normalization` | ImageNet mean/std | ⚠️ **ASSUMED** (justified only *indirectly*, by the sibling UWF run) |
| `color_space` | `RGB` | ✅ code-level fact |
| `verified` flag | **`False`** | ✅ `preprocessing.py:105` |
| Runtime report | `values_verified_by_checkpoint: false`, `normalization_source: "ASSUMED (default, not in checkpoint)"` | ✅ A5 |

The code says all of this out loud, at three levels:

- `preprocessing.py:21-38` — the `ASSUMPTION` block in the module docstring.
- `preprocessing.py:98-104` — the `source` string stored in `_DEFAULTS["cfp"]`.
- `preprocessing.py:196-201` — a `logger.warning("Preprocessing for cfp is an ASSUMPTION, not checkpoint data: …")` fires on **every** CFP load. It fired in the live CFP run (A5).

The library's own warning, reproduced live:

```
** ASSUMPTION WARNING **
  The CFP checkpoint does not state these values, so they are
  a documented default, not checkpoint data. The sibling UWF checkpoint
  records image_size=512. If this model was trained at a
  different size the logits are still produced but accuracy is not
  trustworthy. Re-run with --image-size to test another size.
```

**Why 224 is a silent-failure risk, in the library's own words** (`preprocessing.py:30-38`): *"the sibling UWF checkpoint says `512`; if the CFP run also trained at 512 then 224 is the wrong size and accuracy will suffer. EfficientNet-B0 is fully convolutional with global average pooling, so any size runs without erroring — a wrong size fails silently, which is exactly why this is reported rather than hidden."*

**Therefore: the CFP confidence figure in §7 is NOT a validated accuracy number.** It is a well-formed model output computed under a guessed input size.

### 6.3 Escape hatch

- `with_image_size(config, size)` (`preprocessing.py:213`) and `test_prediction.py --image-size N` allow an override.
- ✅ An override **always clears `verified`**, so an override can never masquerade as checkpoint data (`preprocessing.py:217-226`).
- `get_config()` also flips `verified` to `True` if a positive `image_size` appears in a checkpoint config, and **downgrades to `False` with a warning** if a checkpoint reports a non-ImageNet normalization (`preprocessing.py:184-191`).

### 6.4 Summary table

| Modality | Image size | Normalization | Provenance | `verified` |
|---|---|---|---|---|
| **UWF** | 512 | ImageNet mean/std | ✅ **read from checkpoint `config`** | **True** |
| **CFP** | 224 | ImageNet mean/std | ⚠️ **ASSUMED — not in checkpoint** | **False** |

---

## 7. Real-image prediction results

✅ **Both results VERIFIED by live execution (A5), reproduced in this session.**

**Command form:**
```
backend\.venv\Scripts\python.exe test_prediction.py --image "test_images\cfp\tr000001.jpg" --modality cfp
backend\.venv\Scripts\python.exe test_prediction.py --image "test_images\uwf\tr000001.jpg" --modality uwf
```

### 7.1 Headline results

| | **CFP** | **UWF** |
|---|---|---|
| Image | `test_images/cfp/tr000001.jpg` | `test_images/uwf/tr000001.jpg` |
| Image dimensions | 2077 × 2077 RGB JPEG | 2600 × 2048 RGB JPEG |
| SHA-256 (first 16) | `092D6D04CB618268` | `5AB0576E2DB01E54` |
| **Predicted label** | **Moderate** | **Mild** |
| `predicted_class` | 2 | 1 |
| `description` | Moderate non-proliferative DR | Mild non-proliferative DR |
| **Confidence** | **0.703081 = 70.3081 %** (≈ 70.31 %) | **0.881828 = 88.1828 %** (≈ 88.18 %) |
| Checkpoint | `EfficientNetB0_CFP_final.pth` | `EfficientNetB0_UWF_final.pt` |
| Model | EfficientNet-B0 | EfficientNet-B0 |
| **Device** | **`cpu`** | **`cpu`** |
| Input size used | 224 (⚠️ assumed) | 512 (✅ from checkpoint) |
| Probability sum | 0.999999 | 0.999999 |
| Contract checks | **10 / 10 PASS** | **10 / 10 PASS** |

### 7.2 Full probability vectors

**CFP** — `tr000001.jpg`, class 2 wins:

| Class | Index | Probability |
|---|---|---|
| No DR | 0 | 0.004808 |
| Mild | 1 | 0.018070 |
| **Moderate** | **2** | **0.703081** ← argmax |
| Severe | 3 | 0.273637 |
| Proliferative DR | 4 | 0.000403 |
| | | **Σ = 0.999999** |

**UWF** — `tr000001.jpg`, class 1 wins:

| Class | Index | Probability |
|---|---|---|
| No DR | 0 | 0.023327 |
| **Mild** | **1** | **0.881828** ← argmax |
| Moderate | 2 | 0.088653 |
| Severe | 3 | 0.004967 |
| Proliferative DR | 4 | 0.001224 |
| | | **Σ = 0.999999** |

### 7.3 Output contract — both models, all checks PASS

`test_prediction.py:110-162` asserts ten things. Both live runs printed `CONTRACT: ALL CHECKS PASSED`:

| # | Assertion | CFP | UWF |
|---|---|---|---|
| 1 | exactly 5 probabilities | PASS | PASS |
| 2 | probability keys match the DR class mapping | PASS | PASS |
| 3 | every probability in [0, 1] | PASS (min 0.000403 / max 0.703081) | PASS (min 0.001224 / max 0.881828) |
| 4 | probabilities sum to ~1 (tolerance 1e-4) | PASS (0.999999) | PASS (0.999999) |
| 5 | `predicted_class` in range | PASS | PASS |
| 6 | `predicted_class` has the highest probability | PASS (argmax = 2) | PASS (argmax = 1) |
| 7 | `confidence` equals the highest probability | PASS | PASS |
| 8 | `label` matches `predicted_class` | PASS | PASS |
| 9 | device is CPU | PASS (`cpu`) | PASS (`cpu`) |
| 10 | CUDA is not used | PASS | PASS |

The Σ = 0.999999 is a rounding artefact, not a bug: `predictor.py:176` rounds each of the 5 values to 6 decimals, so the maximum possible drift from 1.0 is 2.5 × 10⁻⁶. The tolerance is 1e-4 (`test_prediction.py:62`).

### 7.4 ⚠️ How far these results can and cannot be pushed

This section is deliberately conservative. Do not quote §7.1 as a performance claim.

1. **`n = 1` per modality.** These are two single-image runs. They prove the pipeline executes and produces a well-formed distribution. They prove **nothing** about accuracy, sensitivity, specificity, or agreement with any reference standard.
2. **⚠️ The CFP number rests on an assumed input size.** 224 was guessed (§6.2). If the CFP run actually trained at 512 — which its own sibling did — 70.31 % is meaningless. This is the single largest caveat in the project.
3. **The two images are not the same image.** Despite the identical filename `tr000001.jpg`, SHA-256 differs (`092D…` vs `5AB0…`) and the dimensions differ (2077×2077 vs 2600×2048). They are two different source captures. **The Moderate-vs-Mild disagreement is therefore not evidence of a bug and not evidence of anything** — the two models were shown different pictures. It must not be reported as cross-modality disagreement.
4. **No ground truth is available.** Neither image has a label in the repository, so no accuracy, QWK, or confusion matrix can be computed. The checkpoints carry their *own training-time* `val_QWK` (CFP `0.9226`, UWF `0.9153`) — that is the training run's validation score, not a measurement of these images.
5. **The label ordering is an assumption about the checkpoints, not a fact stored in them.** Neither file contains a class-index → grade map. `predictor.py:27-52` documents the mapping as "fixed by the ICDR 5-grade scale" and states the order is "the logit order produced by `dr_head` (CFP) and `classifier.1` (UWF)". That the trained model's index 0 really means "No DR" is a reasonable, well-documented inference — **but it is not verifiable from the files.** A permuted head would produce confident nonsense with a perfect output contract.
6. **Auxiliary heads were not requested.** `include_raw_auxiliary` was not set, so no `auxiliary_raw_outputs` appear in either payload.
7. **`proba` sums are structural, not semantic.** A sum of 1.0 proves the softmax ran. It says nothing about whether the model was right.

---

## 8. FastAPI status and endpoints

`backend/main.py`, 206 lines. `API_VERSION = "0.3.0-dev"` (`main.py:53`).

### 8.1 Endpoints — ✅ exactly three, all reachable

Confirmed by A6 against the live service on `http://127.0.0.1:8000`. `GET /openapi.json` lists precisely `/`, `/health`, `/predict` — no more.

| Method | Path | Runs a model? | Live response (A6) |
|---|---|---|---|
| `GET` | `/` | **NO** | `{"app":"RetinaGrade AI","status":"running"}` |
| `GET` | `/health` | **NO** | `{"status":"healthy","device":"cpu","cuda_available":false}` |
| `POST` | `/predict` | ❌ **NO** | see §8.2 |

Interactive docs are available at `http://127.0.0.1:8000/docs` (FastAPI default; not disabled).

### 8.2 `POST /predict` — ❌ VALIDATION ONLY, no inference

This is the most important thing to be unambiguous about. From the source and confirmed live:

- ✅ A valid CFP upload returned `{"success":true,"message":"Image received","modality":"CFP"}` — **no grade, no confidence, no probabilities.**
- ✅ A valid UWF upload returned `{"success":true,"message":"Image received","modality":"UWF"}` — same.
- ✅ An invalid modality returned HTTP 400: `{"success":false,"error":{"code":"unsupported_modality","message":"Unsupported modality 'mri'. Allowed values: cfp, uwf."}}`

`main.py:151-203` performs exactly five checks and nothing else:

1. `modality` (trimmed, lowercased) is `cfp` or `uwf` → else **400 `unsupported_modality`**
2. an `image` file field was actually sent, with a filename → else **400 `invalid_image`**
3. the file is not empty → else **422 `invalid_image`**
4. the file is ≤ `MAX_UPLOAD_BYTES` = **25 MB** → else **413 `image_too_large`**
5. returns `PredictDevResponse`

`main.py:166`: *"The image bytes are read only to check emptiness and size. They are not decoded, not preprocessed and not passed to any network."*

`main.py:192` logs, on every request: `"POST /predict validated: modality=…, file=…, N bytes. Model NOT run (skeleton)."`

**`inference/predictor.predict()` is never imported by `main.py`.** The working, verified model layer is not wired to the API. `main.py:22-24` states: *"`inference/` is left untouched: the CFP and UWF checkpoints on disk are not modified, read or executed by this file."*

### 8.3 Also absent from the API

| Missing | Evidence |
|---|---|
| Any endpoint returning a prediction | `schemas/api.py:4-7` — prediction schemas `PredictionResultModel`, `PredictResponse`, `ModelsInfoResponse` are *"deliberately absent"* |
| `GET /models` or model-status endpoint | `manager.describe()` exists in `model_loader.py:611` but is **never called by the API** |
| Model preload / lifespan / startup hook | no `@app.on_event`, no lifespan context anywhere in `main.py` |
| `/predict` returning `device`, `checkpoint`, `preprocessing` provenance | `PredictionResult.to_dict()` (`predictor.py:76`) builds it; no consumer |
| Request timing / logging of inference | no such code |
| `GET /preprocessing` | `describe_preprocessing()` (`preprocessing.py:229`) exists, unused |
| Any Grad-CAM endpoint | ❌ no Grad-CAM code exists at all (§11) |
| Any Gemini endpoint | ❌ no Gemini code exists at all (§11) |
| Any history / PDF endpoint | ❌ `services/` is empty (§11) |

### 8.4 Middleware and error handling — ✅ present and verified by reading

- **CORS** (`main.py:91-99`): `allow_origins` = 8 fixed localhost origins (ports 8000/3000/5000/8080 on `localhost` and `127.0.0.1`); `allow_origin_regex` = `^http://(localhost|127\.0\.0\.1)(:\d+)?$` — enabled because `ALLOW_LOCALHOST_ANY_PORT = True`, so a Vite dev server on any port is permitted. `allow_credentials=True`, `allow_methods=["GET","POST","OPTIONS"]`, `allow_headers=["*"]`.
- **Global exception handler** (`main.py:109-117`): logs the traceback server-side, returns a generic **500 `internal_error`** with no traceback leak to the client.
- Error envelope is consistent: `{"success": false, "error": {"code", "message"}}` (`main.py:102-106`).
- ✅ `inference/exceptions.py` exists with domain error classes, but ❌ they are **not mapped to HTTP status codes** — no exception handler translates `ModelNotFoundError` / `ArchitectureMismatchError` / `InvalidImageError` into responses.

### 8.5 API verdict

**The service runs, and it is honestly labelled as a skeleton.** `main.py:1-31` and `main.py:86-88` both state that image prediction is NOT implemented. The gap between §7 (working inference) and §8 (skeleton API) is the single largest remaining piece of work (§12.1).

---

## 9. Frontend status

`frontend/` — 32 source files, React 19 + TypeScript 5.9 + Vite 8 + Tailwind 4.

### 9.1 Headline: ✅ a complete, polished, 100 % static UI shell with zero network I/O

- ✅ **There is not a single network call in the entire frontend.** A grep across `frontend/src/**` for `fetch(`, `axios`, `XMLHttpRequest`, `WebSocket`, `EventSource`, `sendBeacon`, `import.meta.env`, `VITE_` returns **no matches**.
- `BACKEND_BASE_URL = 'http://127.0.0.1:8000'` is declared at `src/data/clinical.ts:6` but is **never used for a request** — it is imported once and rendered as a *text label* in `src/components/layout/SidebarContent.tsx:53`.
- ✅ No Vite dev proxy: `vite.config.ts` is 12 lines with `server: { host: '127.0.0.1', port: 5173 }` and **no `server.proxy` block**. Even a future `fetch('/api/...')` would not reach the backend in dev without adding one.
- ✅ The built bundle `dist/assets/index-D3fiBzDM.js` contains exactly one `fetch(` occurrence, and it is React DOM's internal resource preloader, not application code.

### 9.2 What is real and working

| Area | Detail |
|---|---|
| Shell | `AppShell` responsive sidebar + mobile drawer, scroll lock, Escape-to-close, skip link |
| Navigation | 6 views (`analyze`, `dashboard`, `explainability`, `reports`, `history`, `settings`) via `useState` + ternaries in `App.tsx:49,86-121` |
| File input | `ImageDropzone` — drag-and-drop, click, keyboard; `role="radiogroup"`/`aria-checked` on `ModalitySelector` |
| Client validation | 5 accepted MIME types, **25 MB** max (`clinical.ts:9`), zero-byte reject (`useImageSelection.ts:66-71`) |
| Object-URL hygiene | `URL.revokeObjectURL` on replace / clear / unmount (`useImageSelection.ts:26-34,73-76`) |
| Preview | Local `blob:` URL rendering in `ImagePreview` |
| Design system | Full Tailwind v4 token set (`index.css`), 6 `ui/` primitives, 1 icon module |
| Honesty | Runtime notices on every page state that nothing is connected |
| Build | ✅ `npm run typecheck` → **clean, 0 errors**; `node_modules` installed (121 packages) |

### 9.3 What is placeholder

| Surface | Evidence |
|---|---|
| **The entire "Analyze" action** | `App.tsx:73-75` — `handleAnalyze = () => setStatus('analyzed')`. That is the whole of it. `AnalyzePage.tsx:32-33` says so: *"`onAnalyze` only flips the panels from 'idle' to 'placeholder'. No request leaves the browser."* |
| Predicted grade / confidence / probabilities | `AnalyzePage.tsx:136-147` passes `label={null} confidence={null}` and `probabilities={null} predictedIndex={null} probabilitySum={null}`; components render `—` |
| No fake data anywhere ✅ | `types.ts:45-54` types `AnalysisRecord` fields as `| null` with the comment *"they are never faked"*. **There are no hardcoded grades, confidences or probabilities in the codebase.** Only class *names* are static (`clinical.ts:42-78`), and they are never rendered as a prediction. |
| **Grad-CAM** | `XaiPanel.tsx:26-31` `GRAD_CAM_META` values are all `'-'`; `:95-110` the `gradcam` and `overlay` views render an animated dashed circle captioned *"Grad-CAM is not implemented"*; `:33-38` comments admit *"no activation hook, no gradient capture, no heatmap rendering happens anywhere in this build"*; saliency map, class activation map, attention consensus all `'-'`; the summary is 3 `<Skeleton>` bars. `ExplainabilityPage.tsx` is a 26-line pass-through to `XaiPanel`. |
| **AI report / Gemini** | `AiReportCard.tsx:23-28` `REPORT_META` has `Generator: 'Gemini'` as a **hardcoded display string only**; the body is shimmer skeletons, not fake prose; `:138-143` the "Generate report" and "Export PDF" buttons are **permanently `disabled`**; the badge reads `'Awaiting generation'` even after analyzing. **Zero Gemini API integration.** |
| Dashboard KPIs | 4 tiles all `'-'`; Model Status shows both modalities as `Not loaded` |
| History, Settings | `UnavailablePage` with static roadmap bullet lists |
| Inference timing | `PredictionCard.tsx:87` hardcoded `—` |
| `PredictionStatus` | `types.ts:39` has only `'idle' \| 'analyzed'` — no `'loading'`, no `'error'`, no `'done'` |
| `AnalysisRecord` | declared at `types.ts:45`, **referenced nowhere** |

### 9.4 Structural gaps that will block integration

| # | Gap | Location |
|---|---|---|
| 1 | No HTTP client and no API layer at all | `frontend/src/**` |
| 2 | No `server.proxy` in the Vite config | `vite.config.ts` |
| 3 | No `'loading'` / `'error'` states in the status union | `types.ts:39` |
| 4 | No router — no URLs, no deep-linking, no back/forward, state lost on refresh | `App.tsx:49` |
| 5 | Grad-CAM needs a real image-compositing path; today only the `original` view renders an `<img>` | `XaiPanel.tsx:87-110` |
| 6 | `MODEL_FILE_NAMES` is duplicated in the frontend and must be kept in sync by hand with `model_loader.py::MODEL_SPECS` | `clinical.ts:148-151` |
| 7 | No frontend test tooling of any kind | `package.json:7-13` |
| 8 | `dist/` is tracked in git despite `.gitignore:23` | `git ls-files frontend/dist` → 4 files |

✅ One point of good news on #6: `clinical.ts:148-150` lists `EfficientNetB0_CFP_final.pth` and `EfficientNetB0_UWF_final.pt`, which **matches** `model_loader.py:81-82` exactly.

---

## 10. What is currently working

Everything below was reproduced in this session.

### 10.1 Model layer — ✅ complete and verified

1. **Both checkpoints inspected read-only** without mutation. `inspect_models.py` reports `file unchanged by read: True` for both. Type, top-level keys, tensor count, dtypes, devices, conv shapes, head leaves and class count all extracted directly from the files.
2. **Both architectures reconstructed exactly** from the checkpoint keys, with a documented argument for torchvision-over-timm. `verify_architecture.py` returns `EXACT MATCH` for both: names, shapes and **key order** identical; 0 missing, 0 unexpected, 0 shape mismatches.
3. **Both checkpoints load with `strict=True`** and zero diff — CFP 374 tensors, UWF 360 tensors. `strict=False` appears nowhere in the project.
4. **Head layouts auto-detected** from the state_dict keys (`model_loader.py:273`), not hardcoded per modality, and cross-checked against the registered spec.
5. **Both models in `eval()` mode with every parameter *and* buffer on CPU.**
6. **CPU-only enforcement** on five independent layers (load, placement, forward probe, runtime assertion, payload field) — §5.
7. **A `ModelManager`** that loads each modality exactly once behind a `threading.Lock`, keeps CFP and UWF fully separate, and never silently falls back from one modality to the other (`model_loader.py:554-628`).
8. **Typed domain errors** — `ModelNotFoundError`, `CheckpointLoadError`, `ArchitectureMismatchError`, `UnsupportedModalityError`, `InvalidImageError`, `InferenceFailedError`.

### 10.2 Inference layer — ✅ works end-to-end

9. **Real-image prediction runs on real images for both modalities on CPU.** CFP `tr000001.jpg` → Moderate 70.31 %; UWF `tr000001.jpg` → Mild 88.18 %.
10. **The output contract is machine-asserted, 10/10 PASS on both** — 5 probabilities, keys matching the class map, all in [0,1], Σ ≈ 1 within 1e-4, argmax consistency, confidence = max, label = index, `device == "cpu"`, CUDA unused.
11. **A well-defined, documented 5-class ICDR mapping** with `key` and `label` deliberately identical so they cannot drift apart (`predictor.py:42-42`).
12. **`predict()` refuses to guess**: if the model emits a class count other than 5, it raises rather than mapping anyway (`predictor.py:132-136`).
13. **Preprocessing provenance is carried in the output.** Every payload embeds a `preprocessing` block with `normalization_source`, `config_source` and `values_verified_by_checkpoint` — the CFP assumption is machine-visible in the JSON, not just in a comment.
14. **Unnamed auxiliary heads are quarantined** — returned only on explicit request, always as `aux_head_N`, always with a disclaimer.
15. **Decode robustness** — format whitelist, EXIF transpose, RGB conversion, specific errors for empty/corrupt/unsupported input.
16. **An override path for the assumed CFP size** (`--image-size`) that always clears the `verified` flag.

### 10.3 Service layer

17. **The FastAPI app starts and serves.** `/`, `/health`, `/predict` all respond; `/openapi.json` and `/docs` work.
18. **Real request validation with correct status codes** — 400 / 422 / 413, all reproduced live.
19. **CORS configured** for a localhost dev workflow including any Vite port.
20. **A global exception handler** that never leaks a traceback to a client.

### 10.4 Frontend

21. **A complete, responsive, accessible UI shell** — 32 files, 6 views, sidebar + drawer, skip link, ARIA-correct controls.
22. **A working, validated file-selection flow** with drag-and-drop, click and keyboard paths, MIME + 25 MB + zero-byte rules, and correct object-URL lifecycle.
23. **Clean TypeScript** — `tsc -b --noEmit` exits with 0 errors.
24. **A production build exists** in `frontend/dist/` and is newer than the newest source file.
25. **Radical honesty in the UI** — every placeholder is labelled as one, and **no mock prediction data exists anywhere in the codebase.**

---

## 11. What is NOT implemented

| # | Feature | Status | Proof |
|---|---|---|---|
| 1 | **API-side inference** — `POST /predict` running a model | ❌ **NOT IMPLEMENTED** | `main.py:151-203` validates only; returns `{"success":true,"message":"Image received"}`; `predictor` never imported by `main.py` |
| 2 | **Any prediction response schema** | ❌ **NOT IMPLEMENTED** | `schemas/api.py:4-7` — deliberately absent; only `RootResponse`, `HealthResponse`, `PredictDevResponse`, `ErrorResponse` exist |
| 3 | **Grad-CAM / saliency / class activation maps** | ❌ **NOT IMPLEMENTED** | Zero backend code. A repo-wide grep for `gradcam|grad-cam|grad_cam|GradCAM` finds only frontend placeholder strings. `XaiPanel.tsx:33-38` admits no activation hook, no gradient capture, no heatmap |
| 4 | **Gemini report generation** | ❌ **NOT IMPLEMENTED** | Zero backend code. `google-genai` not installed (`requirements.txt:28`); `GEMINI_API_KEY` in `.env` is never loaded (`load_dotenv` never called); "Gemini" in the frontend is a display string at `AiReportCard.tsx:24` |
| 5 | **PDF export** | ❌ **NOT IMPLEMENTED** | `reportlab` not installed (`requirements.txt:29`); `services/` is empty; the frontend "Export PDF" button is `disabled` (`AiReportCard.tsx:138-143`) |
| 6 | **Screening history / persistence** | ❌ **NOT IMPLEMENTED** | `services/__init__.py` is 0 bytes; no database, no store, no file writer; the frontend History nav item is `implemented: false` (`clinical.ts:138`) and routes to `UnavailablePage` |
| 7 | **Frontend ↔ backend integration** | ❌ **NOT IMPLEMENTED** | Zero network calls in `frontend/src`; no HTTP client dependency; no Vite proxy; `BACKEND_BASE_URL` is a text label only |
| 8 | **Model preload / warm-up in the API** | ❌ **NOT IMPLEMENTED** | No startup or lifespan hook in `main.py`; `manager` and `manager.describe()` are never called from the API |
| 9 | **`GET /models` model-status endpoint** | ❌ **NOT IMPLEMENTED** | `LoadedModel.describe()` and `ModelManager.describe()` exist in `model_loader.py` but have no HTTP route |
| 10 | **Automated test suite** | ❌ **NOT IMPLEMENTED** | `pytest`/`httpx` commented out (`requirements.txt:30`); `tests/` holds only `.gitkeep`; the four `test_*.py` files are CLI scripts, not a suite; no CI config; frontend has no test tooling |
| 11 | **Training code, dataset pipeline, evaluation harness** | ❌ **OUT OF SCOPE / ABSENT** | No such files exist. The repo consumes checkpoints only. **No accuracy, QWK, sensitivity or specificity has been measured here** |
| 12 | **Dataset ground-truth labels** | ❌ **ABSENT** | `test_images/` holds 2 unlabelled JPEGs. Accuracy metrics are therefore impossible from this repository |
| 13 | **Frontend routing** | ❌ **NOT IMPLEMENTED** | No `react-router`; `useState` + ternaries (`App.tsx:49,86-121`) |
| 14 | **Frontend loading / error states** | ❌ **NOT IMPLEMENTED** | `PredictionStatus = 'idle' \| 'analyzed'` only (`types.ts:39`) |
| 15 | **Settings view** | ❌ **NOT IMPLEMENTED** | `clinical.ts:140` `implemented: false`; renders `UnavailablePage` |
| 16 | **The 7 CFP auxiliary lesion heads** | ⚠️ **MEANING UNKNOWN** | Count verified (7); no label map in the checkpoint. Cannot be named as clinical findings — `model_architecture.py:59-61` |
| 17 | **A verified CFP input size** | ⚠️ **UNVERIFIED** | The CFP checkpoint has no `config` key at all. 224 is an assumption; the sibling model used 512 |
| 18 | **A verified class-index → grade map** | ⚠️ **INFERRED** | Neither checkpoint stores a label map. The ICDR ordering in `predictor.py:46-52` is a documented, reasonable inference, not file-proven |
| 19 | **Auth, rate limiting, request IDs, structured logging** | ❌ **NOT IMPLEMENTED** | No code in `main.py` |
| 20 | **Deployment config** | ❌ **NOT IMPLEMENTED** | No Dockerfile, no `docker-compose`, no CI workflow, no reverse-proxy or process-manager config |

---

## 12. Next development steps

Ordered by dependency. Steps 1–3 are the critical path from "model layer works" to "product works".

### 12.1 🔴 P0 — Wire the working model layer into the API

**Why first:** the hardest, riskiest part of this project is already done and verified (§7). The API is a validated-request shell that throws the result away. This step converts a working engine into a working service.

1. Add a FastAPI lifespan/startup hook that calls `manager.load_all()` once and logs per-modality status.
2. Add real prediction response schemas to `schemas/api.py` — `PredictResponse`, `PredictionResultModel`, `ModelsInfoResponse` — mirroring `PredictionResult.to_dict()` (`predictor.py:76-103`) field for field.
3. Replace the placeholder body of `POST /predict` with a call to `inference.predictor.predict(...)`.
4. **Map `inference/exceptions.py` to HTTP status codes** via exception handlers: `UnsupportedModalityError` → 400, `InvalidImageError` → 422, `ModelNotFoundError` → 503, `ArchitectureMismatchError` / `CheckpointLoadError` → 500 with a server-side-only detail.
5. Add `GET /models` returning `manager.describe()`, and `GET /preprocessing` returning `describe_preprocessing()`.
6. **Pass through the `preprocessing` provenance block in the response** so an API consumer sees `values_verified_by_checkpoint: false` for CFP instead of having to trust a docstring.
7. Add an integration test that POSTs a real `test_images/cfp/tr000001.jpg` and asserts the response equals the §7.1 contract. Requires installing `pytest` + `httpx` (`requirements.txt:30`).

### 12.2 🔴 P0 — Resolve the CFP preprocessing assumption

**Why:** the CFP confidence figure is currently uninterpretable (§6.2, §7.4.2). This is a correctness issue, not a polish item.

1. **Recover the true CFP training resolution from an external source** — the training script or its logs. `EfficientNetB0_CFP_final.pth` cannot tell us: it has no `config`.
2. Sweep the size empirically: run `test_prediction.py --image-size {224, 256, 320, 384, 448, 512}` over a labelled set and pick the size that maximises QWK against ground truth. Record the result.
3. Once a size is confirmed by evidence, set `DEFAULT_IMAGE_SIZE` and flip `_DEFAULTS["cfp"]["verified"]` to `True` with a `source` string that names the evidence. **Do not flip the flag without evidence** — the flag is the mechanism that keeps this honest.
4. Independently confirm the CFP normalization is ImageNet (the current justification is only "the sibling UWF run used it").
5. **Fix the stale UWF `config` fields** — `experiment` and `dataset` both say CFP. Ask the training owner to correct them, or document them as known-bad in `model_loader.py::_extract_metadata` so a future reader is not misled.

### 12.3 🟠 P1 — Obtain ground truth and measure performance

**Why:** `n = 1` per modality (§7.4.1) proves nothing. No accuracy claim in this project is currently supportable.

1. Source labelled CFP and UWF validation sets with ICDR ground truth. **This is an external dependency and the main schedule risk.**
2. Add a dataset loader and an evaluation script computing per-modality: accuracy, macro-F1, quadratic weighted kappa (the checkpoints were selected on QWK), a confusion matrix, and per-class sensitivity/specificity.
3. Compare measured QWK against the training-time values (CFP `0.9226`, UWF `0.9153`). A large shortfall would point at the preprocessing pipeline.
4. Report results **separately per modality**. Never compare CFP and UWF predictions on different images.

### 12.4 🟠 P1 — Connect the frontend

1. Add an HTTP client. Decide between `fetch` and a small wrapper; keep it in one module.
2. Add `server.proxy` in `vite.config.ts` to forward `/api` → `http://127.0.0.1:8000`, and use relative URLs in the frontend. The backend CORS already permits any localhost port, but a proxy removes the class of problem entirely.
3. **Extend `PredictionStatus`** to `'idle' | 'loading' | 'analyzed' | 'error'` (`types.ts:39`) *before* wiring the call.
4. Replace `handleAnalyze` (`App.tsx:73-75`) with a real `POST /predict` using `FormData` with `image` and `modality`.
5. Wire `PredictionCard`, `ProbabilityCard` and `ProbabilityBars` to the real response. These components are already built and already accept the right props — they are currently handed `null`.
6. Add a `GET /models` call so the Dashboard "Model Status" tiles stop reading `Not loaded`.
7. Add abort/timeouts and surface backend error envelopes in the UI.
8. **Keep `MODEL_FILE_NAMES` in sync** with `model_loader.py::MODEL_SPECS` (`clinical.ts:146-151`) — preferably generate it rather than duplicating it.
9. Add a router. Manual `useState` navigation loses state on refresh and supports no deep links.

### 12.5 🟡 P2 — Grad-CAM explainability

1. Backend first, before the UI. Nothing exists today.
2. Choose and document the target layer — for EfficientNet-B0 the conventional choice is the last MBConv block before pooling (`features.8`), not the classifier.
3. Implement forward + backward hooks, weight activations by the DR-class gradient, ReLU, upsample, and a colormap overlay composited against the **original** image.
4. The model runs in `eval()` under `torch.no_grad()` (`model_loader.py:451`, `predictor.py:199`). Grad-CAM **requires gradients**, so a separate `requires_grad_(True)` path on a cloned model is needed. **This is the main technical trap in this step** — do not remove `no_grad()` from the production prediction path.
5. Choose a target resolution and **re-apply the correct per-modality preprocessing to the overlay base image** (512 for UWF, the resolved CFP size). Render the heatmap at the source resolution, not the network input resolution.
6. Add `GET /predict/gradcam` or extend `POST /predict` with `include_gradcam`.
7. Frontend: give `XaiPanel` a real overlay path. Today only the `original` view renders an `<img>` (`XaiPanel.tsx:87-110`).

### 12.6 🟡 P2 — Gemini report generation

1. Install `google-genai` (uncomment `requirements.txt:28`).
2. **Call `load_dotenv()`** so `backend/.env` is actually read — this is currently missing project-wide (§2.1), and `GEMINI_API_KEY` is inert without it.
3. Implement the client in `services/` (the package exists and is empty).
4. **Prompt design is the safety-critical part.** The report must be grounded in the actual prediction — grade, confidence, the full probability vector, the verified preprocessing provenance. It must state model limitations and must not present the model output as a diagnosis.
5. ⚠️ **Carry the `verified` flag into the prompt and the output.** A report generated from an assumed CFP input size must say so. A polished clinical-sounding paragraph built on a 224-vs-512 guess is the worst failure mode this project has.
6. **Do not name the auxiliary lesion heads** (§3.6) — a generative model will happily invent plausible-sounding lesion names for 7 unlabeled logits.
7. Add `POST /report`.
8. Frontend: enable the currently-`disabled` buttons in `AiReportCard`.

### 12.7 🟡 P2 — Screening history and PDF export

1. Install `reportlab` (`requirements.txt:29`).
2. Choose and justify a store. Nothing exists; `services/` is empty.
3. Define a record schema. `AnalysisRecord` at `frontend/src/types.ts:45` is a reasonable starting shape but is currently unused and frontend-only.
4. Persist: image reference, modality, label, confidence, full probability vector, checkpoint name, **preprocessing provenance including `verified`**, model version, timestamp, device.
5. Add `GET /history` and `POST /report/pdf`.
6. Frontend: replace `UnavailablePage` for the History nav item.
7. Add patient-data handling policy — PHI in filenames, retention, access control. Decide before storing anything.

### 12.8 🟢 P3 — Repository hygiene

1. **Untrack `frontend/dist/`.** ✅ Confirmed: `git ls-files frontend/dist` returns 4 files while `.gitignore:23` lists `dist/`. The rule was added after the files were staged, and `.gitignore` does not apply retroactively to tracked files. Fix with `git rm -r --cached frontend/dist`.
2. Drop or use `opencv-python`. It is installed, pinned at `5.0.0.93`, and **never imported** — pure install weight.
3. Actually use `python-dotenv`, or drop it (§12.6.2).
4. Add `pytest` + `httpx` and convert the four CLI scripts into a real suite. Keep the scripts — they are excellent diagnostic tools — but add assertions that can run unattended.
5. Add a CI workflow running `tsc -b --noEmit`, `eslint .`, and the Python verification scripts.
6. Add `docs/`. It contains only `.gitkeep`.
7. Reconcile the two parameter-count figures used across the repo (`4,022,920` vs `4,064,985`). Both are correct; label them explicitly so future readers do not read them as a bug (§3.5).

### 12.9 🟢 P3 — Documentation

1. **Fix `README.md`.** It is stale in ways that actively mislead:
   - *"Frontend: Flutter (desktop + mobile) - not created yet"* and *"`frontend/`: # Flutter app (not created yet)"* — the frontend is React + TypeScript + Vite + Tailwind and exists with 32 source files.
   - *"Models: EfficientNet - `efficient_cfp.pt` … `efficient_uwf.pt`"* and *"place `efficient_cfp.pt` and `efficient_uwf.pt` in `backend/models/` manually"* — the real filenames are `EfficientNetB0_CFP_final.pth` and `EfficientNetB0_UWF_final.pt`. The frontend already caught this drift (`clinical.ts:146`).
   - `- [ ] Model loading & inference (CFP / UWF)` — both are now complete and verified (§3, §4, §7).
   - `backend/services/` and `backend/utils/` are described as if populated; both are empty.
2. Add a document recording the **ground-truth provenance of the checkpoints** — who trained them, from which dataset, with which labels, and the true CFP image size. This is the missing input for §12.2 and §12.3, and it is not derivable from the repository.

---

## Appendix A — Commands executed for this report

Every quantitative claim above traces to one of these. None modified any source file.

| ID | Command (from `backend/` unless noted) | Purpose |
|---|---|---|
| **A1** | `.\.venv\Scripts\python.exe inference\inspect_models.py` | Read-only checkpoint inspection: type, top-level keys, tensor counts, dtypes, devices, conv shapes, head leaves, class counts, `file unchanged by read` |
| **A2** | `.\.venv\Scripts\python.exe inference\verify_architecture.py` | Architecture ↔ checkpoint comparison; `EXACT MATCH` verdicts; timm ruled out; key spot-checks |
| **A3** | `.\.venv\Scripts\python.exe -m inference.test_model_loading` | `strict=True` weight loading, eval mode, CPU placement of parameters **and** buffers, CPU-only assertion |
| **A4** | `.\.venv\Scripts\python.exe test_models.py` | Per-modality checkpoint facts, effective preprocessing, CUDA check, `RESULT: CFP=OK UWF=OK` |
| **A5** | `.\.venv\Scripts\python.exe test_prediction.py --image "test_images\cfp\tr000001.jpg" --modality cfp`<br>`.\.venv\Scripts\python.exe test_prediction.py --image "test_images\uwf\tr000001.jpg" --modality uwf` | **The §7 prediction results**, plus the 10-assertion output contract |
| **A6** | `uvicorn main:app --host 127.0.0.1 --port 8000`, then `GET /`, `GET /health`, `GET /openapi.json`, `POST /predict` (cfp, uwf, and an invalid `mri` modality) | §8 — live endpoint behaviour and the validation-only payload |
| **A7** | `.\.venv\Scripts\python.exe -c "…"` (temp script, since deleted) | Dumped the full 27-key UWF `config`, the CFP `history[0]` fields, and reconciled the parameter counts by direct summation |
| **A8** | Pillow open of both test images; `Get-FileHash -Algorithm SHA256` on both | Image dimensions, modes and SHA-256, proving the two `tr000001.jpg` files are **different images** |
| **A9** | `npm run typecheck` (in `frontend/`) | TypeScript build check — exits clean |
| **A10** | Grep of `frontend/src/**` for `fetch(`, `axios`, `XMLHttpRequest`, `WebSocket`, `import.meta.env` | Proves **zero** network calls |
| **A11** | Grep of `backend/**/*.py` for `import cv2`, `load_dotenv`, `gradcam\|gemini\|genai\|reportlab` | Proves opencv and dotenv are unused, and that no Grad-CAM / Gemini / PDF code exists |
| **A12** | `git log --oneline`, `git status --short`, `git ls-files frontend/dist`, `Test-Path frontend\node_modules` | 7 commits, HEAD `b093fde`, clean tree, `dist/` tracked, `node_modules` installed |

**Note on A6:** a uvicorn instance was already listening on `127.0.0.1:8000` when the probe began, so the probe's own launch attempt failed to bind (`Errno 10048`) and the responses came from the already-running instance of the same `main:app`. That instance was left running; the two temporary log files created during the probe were deleted and `git status` is clean.

---

## Appendix B — One-line status per subsystem

| Subsystem | Status |
|---|---|
| Checkpoint inspection | ✅ Complete, verified, read-only |
| CFP architecture reconstruction | ✅ EXACT MATCH, `strict=True`, 0 diff |
| UWF architecture reconstruction | ✅ EXACT MATCH, `strict=True`, 0 diff |
| CPU-only enforcement | ✅ Verified on 5 independent layers |
| Preprocessing pipeline | ✅ Implemented · **CFP params ⚠️ ASSUMED, UWF ✅ verified from checkpoint** |
| Real-image inference (CLI) | ✅ Both modalities, CPU, 10/10 contract checks pass |
| FastAPI service | ✅ Runs · ❌ **no endpoint runs a model** |
| Frontend UI shell | ✅ Complete, typed, `tsc` clean |
| Frontend ↔ backend wiring | ❌ Zero network calls |
| Grad-CAM | ❌ Not implemented (neither side) |
| Gemini reports | ❌ Not implemented (neither side) |
| History / PDF | ❌ Not implemented (neither side) |
| Automated tests / CI | ❌ None |
| Ground truth / accuracy measurement | ❌ Not possible — no labels in repo |
| Documentation | ⚠️ `README.md` materially stale |

---

## Closing assessment

The **model layer is finished, rigorously verified, and the best-engineered part of this repository.** Both checkpoints were reconstructed exactly from their own keys, load with `strict=True` and zero diff, sit in `eval()` mode entirely on the CPU, and produce well-formed 5-class distributions on real images — Moderate 70.31 % for CFP, Mild 88.18 % for UWF, both on `tr000001.jpg`, both on CPU, both with Σ ≈ 1.0 and all ten contract assertions passing. The discipline is genuine: no `strict=False` anywhere, no `timm`, no CUDA, no fabricated numbers, and the one genuinely unknown quantity — the CFP input size — is labelled as an assumption in the code, in the log output, and in the JSON payload.

**The product layer does not exist yet.** The API validates an upload and throws the result away; the frontend is a polished shell that never makes a request; Grad-CAM, Gemini, history and PDF are absent from both sides. The frontend's discipline of rendering honest placeholders rather than fake predictions deserves credit, and it should be preserved when the wiring lands.

**The critical path is short and unusually clear:** (1) wire `predictor.predict()` into `POST /predict`; (2) determine the true CFP input size from an external source; (3) obtain labelled data so any accuracy claim becomes possible. Everything else in §12 is downstream of those three.
