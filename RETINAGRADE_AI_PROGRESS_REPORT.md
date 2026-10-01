# RetinaGrade AI — Project Progress Report

**Report date:** 1 October 2026
**Repository:** `F:\retinagrade-ai` (Git, 6 commits, HEAD = `a529878 "prepar for prediction"`)
**Environment:** Windows, `backend\.venv` = CPython 3.10.11
**Basis of this report:** direct inspection of the repository plus live execution of the existing verification scripts on the current machine. Every quantitative statement below is either quoted from a file in the repository or was produced by a command listed in Appendix A. Nothing is estimated, projected or carried over from an unverified claim.

---

## 1. Project Overview

RetinaGrade AI is a diabetic-retinopathy (DR) screening application built around two pretrained retinal-imaging models that are supplied as checkpoint files rather than trained inside this repository. The system is designed around two imaging modalities:

| Modality | Full name | Capture |
|---|---|---|
| **CFP** | Colour Fundus Photography | Centred circular view of the posterior pole |
| **UWF** | Ultra-Wide Field | Wide single capture covering the retina including the periphery |

Both models are **torchvision EfficientNet-B0** networks that output a **5-class DR grade** on the ICDR scale (`No DR`, `Mild`, `Moderate`, `Severe`, `Proliferative DR`), as defined in `backend/inference/predictor.py:46` and mirrored in the frontend at `frontend/src/data/clinical.ts:42`.

**Architectural intent (documented, not yet delivered end-to-end):** upload a retinal capture → server-side CPU inference → DR grade, confidence and per-class probabilities → Grad-CAM explainability overlay → optional Gemini-generated short report → screening history with PDF export.

**Current stage in one sentence:** the model layer is finished and verified (architecture reconstructed, both checkpoints loading strictly onto the CPU), while the API and the web client are deliberately non-functional placeholders that never execute a network.

**Scope boundary:** the repository contains model *consumption* code only. There is no training code, no dataset pipeline and no evaluation harness in this repository (see §4 and §12).

---

## 2. Current Project Structure

```
retinagrade-ai/
├── README.md                      (2,412 B — partly outdated, see §11)
├── .gitignore                     (VCS, venv, *.pt/*.pth, secrets; no node_modules rule)
├── RETINAGRADE_AI_PROGRESS_REPORT.md
│
├── backend/                       FastAPI service, CPU-only
│   ├── .venv/                     CPython 3.10.11 virtualenv (not tracked)
│   ├── .env                       APP_ENV/HOST/PORT + empty GEMINI_API_KEY
│   ├── .gitignore
│   ├── requirements.txt           pinned CPU-only wheels
│   ├── main.py                    206 lines — API skeleton, 3 endpoints
│   ├── test_models.py             CLI checkpoint verification
│   ├── test_prediction.py         CLI real-image inference test
│   ├── models/
│   │   ├── .gitkeep               (checkpoints are git-ignored)
│   │   ├── EfficientNetB0_CFP_final.pth   16,408,231 B
│   │   └── EfficientNetB0_UWF_final.pt    48,652,882 B
│   ├── inference/                 ← the completed part of the project
│   │   ├── __init__.py            (empty)
│   │   ├── inspect_models.py      20,146 B  read-only checkpoint inspection
│   │   ├── model_architecture.py  10,574 B  evidence-based architecture definition
│   │   ├── verify_architecture.py 15,125 B  architecture ↔ checkpoint comparison
│   │   ├── model_loader.py        25,187 B  strict CPU weight loading
│   │   ├── test_model_loading.py  12,757 B  weight-loading verification CLI
│   │   ├── preprocessing.py       12,423 B  decode → resize → normalise
│   │   ├── predictor.py            7,590 B  DR grading (not called by the API)
│   │   └── exceptions.py           1,391 B  domain errors → HTTP mapping
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── api.py                 52 lines — 4 Pydantic schemas
│   ├── services/__init__.py       EMPTY (placeholder for Gemini / history / PDF)
│   └── utils/__init__.py          EMPTY (placeholder)
│
├── frontend/                      React + TypeScript + Vite + Tailwind SPA
│   ├── package.json / package-lock.json / vite.config.ts
│   ├── tsconfig.json / tsconfig.app.json / tsconfig.node.json
│   ├── eslint.config.js / index.html / public/retina.svg
│   ├── dist/                      prior production build (untracked)
│   │   ├── index.html (962 B)
│   │   ├── retina.svg (378 B)
│   │   ├── assets/index-D3fiBzDM.js  (268,520 B)
│   │   └── assets/index-DWvLyAx9.css (35,183 B)
│   └── src/                       32 files (25 .ts/.tsx + 1 .css)
│       ├── App.tsx, main.tsx, types.ts, index.css
│       ├── pages/                 AnalyzePage, DashboardPage,
│       │                          ExplainabilityPage, ReportsPage, UnavailablePage
│       ├── components/
│       │   ├── analyze/           ImageDropzone, ImagePreview, ModalitySelector
│       │   ├── layout/            AppShell, Brand, SidebarContent, SidebarNav, TopBar
│       │   ├── results/           PredictionCard, ProbabilityCard, ProbabilityBars
│       │   ├── reports/           AiReportCard
│       │   ├── xai/               XaiPanel
│       │   ├── ui/                Badge, Button, Card, EmptyState, Feedback, Placeholder
│       │   └── icons/index.tsx
│       ├── data/clinical.ts       modality + DR-class definitions
│       ├── hooks/useImageSelection.ts
│       └── lib/format.ts
│
├── tests/                         EMPTY (.gitkeep only) — no automated suite
└── docs/                          EMPTY (.gitkeep only)
```

**Version-control state (verified with `git status` / `git ls-files`):**

- 24 tracked files; all tracked content is backend + top-level files.
- **The entire `frontend/` application is untracked** (`src/`, configs, `package.json`, `package-lock.json`, `index.html`, `public/`, `dist/`). It exists on disk but is not in Git.
- `backend/main.py` and `backend/schemas/api.py` carry **uncommitted modifications**. Inspecting that diff shows it *removes* an inference-wired API (`0.2.0`, with `GET /models`, a lifespan model preload, an `InferenceError` handler and a real `POST /predict` calling `predictor.predict`) and replaces it with the current `0.3.0-dev` validation-only skeleton. The working tree is the skeleton described throughout this report; the removed code exists only in commit `759dc60` and is not part of the current source of truth.
- `frontend/node_modules/` and `frontend/dist/` are untracked **and not covered by `.gitignore`** (hygiene gap).

---

## 3. Technology Stack

### 3.1 Backend — pinned in `backend/requirements.txt`, verified installed

| Component | Pinned | Installed & verified |
|---|---|---|
| Python | 3.10 / 3.11 | **3.10.11** |
| PyTorch | `torch==2.14.0+cpu` | **2.14.0+cpu** |
| TorchVision | `torchvision==0.29.0+cpu` | **0.29.0+cpu** |
| FastAPI | `fastapi==0.141.1` | **0.141.1** |
| Uvicorn | `uvicorn[standard]==0.54.0` | **0.54.0** |
| Starlette | (transitive) | 1.7.0 |
| Pydantic | (transitive) | 2.13.5 |
| python-multipart | `0.0.32` | **0.0.32** |
| python-dotenv | `1.2.3` | **1.2.3** |
| NumPy | `2.2.6` | **2.2.6** |
| Pillow | `12.3.0` | **12.3.0** |
| OpenCV | `opencv-python==5.0.0.93` | **5.0.0.93** |

Notes verified during inspection:

- `requirements.txt:10` adds `--extra-index-url https://download.pytorch.org/whl/cpu` so the `+cpu` builds are resolved instead of the CUDA defaults. This is the mechanism that makes CPU-only operation real rather than aspirational.
- **`timm` is not installed and is not required.** `verify_architecture.py` prints "timm not installed (and not required)"; the trunk was identified as torchvision's from the checkpoint keys themselves (`model_architecture.py:34-40`).
- Deliberately **not** installed: `google-genai` (Gemini), `reportlab` (PDF), `pytest`, `httpx` — all listed as comments in `requirements.txt:27-30`.
- No CUDA package is present anywhere in the virtualenv (`pip list` contains no `nvidia-*` or `triton` entries).

### 3.2 Frontend — `frontend/package.json`

React `^19.3.0` + React DOM `^19.3.0`; Vite `^8.3.1` with `@vitejs/plugin-react` `^6.1.1`; TypeScript `~5.9.3`; Tailwind CSS `^4.3.3` via `@tailwindcss/vite`; ESLint `^10.11.0` with `typescript-eslint`, `eslint-plugin-react-hooks`, `eslint-plugin-react-refresh`. Toolchain present on the machine: **Node v24.14.0, npm 11.12.0**. Vite dev server is pinned to `127.0.0.1:5173` (`vite.config.ts:9-11`).

**Correction to the project's own documentation:** `README.md:5` and the layout block at `README.md:53` describe the frontend as *"Flutter (desktop + mobile) — not created yet"*. That is stale. The delivered client is a **React web SPA**, not Flutter. `.gitignore:26-36` still carries the Flutter ignore rules, and `frontend/src/data/clinical.ts:146` acknowledges the mismatch in a code comment. No Flutter project (`pubspec.yaml`, `.dart_tool/`) exists.

### 3.3 Notable dependency decision

The runtime dependency list contains **no HTTP client, no database, no ORM and no authentication library**. There is no persistence layer and no user model anywhere in the repository.

---

## 4. MMRDR Dataset Information

**Verified finding: this repository contains no MMRDR dataset artifacts and no dataset code.**

Concretely, all of the following are absent:

- No dataset directory, no image archives, no CSV/JSON label files, no split manifests, no download scripts.
- No dataset loader, augmentation pipeline, sampler or `Dataset` subclass anywhere in `backend/`.
- No evaluation, cross-validation or metrics script. `backend/main.py:20` lists "dataset evaluation or accuracy calculation" among the things deliberately *not* implemented.
- `docs/` and `tests/` contain only `.gitkeep`.
- A full-text search of the working tree (backend, docs, tests and frontend sources) returns **no occurrence of the string `MMRDR`**, and no occurrence of any other public-dataset identifier (`Messidor`, `IDDr`, `DeepDR`, `APTOS`, `Kaggle`).
- The only two dataset-adjacent strings anywhere in the repository are inside the **UWF checkpoint's own `config` dictionary** and are quoted below.

### 4.1 Dataset-related evidence available (from the checkpoint, not from this project)

The UWF checkpoint stores a 27-key training `config`; the fields that bear on data are reproduced verbatim (`inference/preprocessing.py:8-19, 45-51` and Appendix A.5):

| Key | Value | Note |
|---|---|---|
| `dataset` | `'CFP'` | **Stale label** — written into the UWF model file by the CFP training script |
| `experiment` | `'E1_CFP_EfficientNetB0'` | **Stale label** — same reason |
| `validation_fraction` | `0.1` | A 10 % validation split was used at training time |
| `batch_size` | `4` | |
| `num_workers` | `0` | |
| `epochs` | `20` | Training stopped at epoch 7 (UWF) / 12 (CFP) |
| `early_stopping_patience` | `5` | |
| `seed` | `42` | |
| `image_size` | `512` | Verified, read from the checkpoint at load time |
| `normalization` | `'ImageNet mean/std'` | Verified |
| `pretrained` | `'ImageNet'` | Verified |
| `augmentation` | `RandomHorizontalFlip(p=0.5) + RandomRotation(10)` | **Training-only**; `preprocessing.py:53-56` forbids applying it at inference |
| `amp` / `data_parallel` / `channels_last` | `True` / `False` / `False` | |
| `device` / `gpu_count` | `cuda` / `2` | Original training used 2 GPUs |
| `pytorch_version` / `torchvision_version` / `cuda_version` | `2.10.0+cu128` / `0.25.0+cu128` / `12.8` | Not required for inference here |

The CFP checkpoint has **no `config` key at all**; its top-level keys are exactly `epoch`, `model_state_dict`, `best_val_qwk`, `history` (verified in Appendix A.5).

### 4.2 Consequences and open risks

1. **The 5-class label map is a project assumption, not checkpoint data.** Neither file stores a class→label mapping. The ICDR ordering used throughout the code is defined in `predictor.py:46-52` and mirrored in `frontend/src/data/clinical.ts:42`; if the training label encoder used a different index order, every reported grade is mislabelled while remaining numerically valid. This is the single highest-impact unresolved risk in the project and it cannot be closed without the training script or dataset metadata.
2. **No dataset is available to re-verify the models.** Accuracy, per-class metrics, confusion matrices and any comparison against MMRDR ground truth are currently unobtainable from this repository.
3. **The UWF `config` is internally inconsistent** (`dataset='CFP'` in a UWF file). `preprocessing.py:43-51` documents this and treats it as the reason CFP preprocessing defaults are treated as assumptions rather than inferred.

---

## 5. CFP and UWF Model Details

All facts in this section were read out of the checkpoint files by `inference/inspect_models.py` (`torch.load(map_location='cpu', weights_only=True)`); the tool explicitly reports `file unchanged by read: True` for both files.

### 5.1 Common properties

| Property | CFP | UWF |
|---|---|---|
| File | `EfficientNetB0_CFP_final.pth` | `EfficientNetB0_UWF_final.pt` |
| Size | 16,408,231 B | 48,652,882 B |
| Python type | `dict` (checkpoint dict, **not** a bare state_dict, **not** a pickled `nn.Module`) | `dict` (same) |
| Weights key | `model_state_dict` | `model_state_dict` |
| Tensors | **374** | **360** |
| dtypes | `float32`, `int64` | `float32`, `int64` |
| Device after `map_location="cpu"` | `cpu` | `cpu` |
| Trunk key prefix | `features` (358 tensors) | `features` (358 tensors) |
| Trunk stages present | `0 … 8` (9 MBConv stages) | `0 … 8` |
| Stem conv | `features.0.0.weight` = `(32, 3, 3, 3)` → **3-channel RGB input** | identical |
| Widest conv | `features.6.1.block.2.fc1.weight` = `(48, 1152, 1, 1)` | identical |
| Final conv | `features.8.0.weight` = `(1280, 320, 1, 1)` → **1280-d output** | identical |
| 4-D conv layers | 81 | 81 |
| `num_batches_tracked` count | 49 (= torchvision B0's BatchNorm count) | 49 |

**Both files share the identical EfficientNet-B0 convolutional trunk.** The 16.4 MB vs 48.7 MB size difference is explained by the UWF file also carrying `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `history` and `config`; it is not a different architecture.

### 5.2 CFP — multi-head network

| Key prefix | Tensors | Meaning |
|---|---|---|
| `features` | 358 | torchvision EfficientNet-B0 trunk, unchanged |
| `dr_head` | 2 | `weight (5, 1280)`, `bias (5,)` → **5-class DR grade**, 1280→5 |
| `lesion_heads` | 14 | `lesion_heads.0 … lesion_heads.6`, each `Linear(1280, 1)` → 7 binary auxiliary outputs |

- Head layout tag: `multi_head`.
- Tensor-bearing indices are contiguous from 0 → `nn.ModuleList`.
- Other top-level metadata: `epoch = 12`; `best_val_qwk = 0.9225706931411989` (`np.float64`); `history` = list of 12 epoch records.
- **`lesion_heads` are deliberately unnamed.** No label map for them exists in the file, so their ordering and clinical meaning are *unknown*. `model_architecture.py:59-61` and `model_loader.py` treat them as raw numbers that must never be presented as clinical findings.
- Training-history fields present: `epoch, lr, train_total_loss, train_dr_loss, train_lesion_loss, val_total_loss, val_dr_loss, val_lesion_loss, val_QWK, val_Macro_F1, val_Accuracy`. Final recorded epoch (12): `val_QWK 0.9087346719076914`, `val_Macro_F1 0.7233143910957684`, `val_Accuracy 0.8337078651685393`. The stored `best_val_qwk = 0.92257` is higher than the last epoch, i.e. the best checkpoint was not the final epoch.
- The multi-task structure (separate DR loss and lesion loss) is confirmed by the presence of `train_dr_loss`/`train_lesion_loss` in the history.

### 5.3 UWF — single-head network

| Key prefix | Tensors | Meaning |
|---|---|---|
| `features` | 358 | the same EfficientNet-B0 trunk |
| `classifier` | 2 | `classifier.1.weight (5, 1280)`, `classifier.1.bias (5,)` → **5-class DR grade**, 1280→5 |

- Head layout tag: `single_head`. The whole file is exactly `torchvision.models.efficientnet_b0(num_classes=5)`.
- `classifier` has exactly **one tensor-bearing index, `1`**. Index 0 holds no tensors, so slot 0 is a parameter-free module → the stock `nn.Sequential(nn.Dropout(p), nn.Linear(...))` layout. Inspection prints: *"slot 0 is nn.Dropout and holds no tensors"*; the verification run prints `classifier = Sequential(Dropout(p=0.2), Linear(1280, 5))`.
- Other top-level metadata: `epoch = 7`; `best_qwk = 0.9152520275167323`; `history` = dict with 9 keys covering **epochs 1–7**; `config` = 27 keys (see §4.1).
- Training-history fields present: `epoch, train_loss, val_loss, val_qwk, val_macro_f1, val_accuracy, val_macro_auc, learning_rate, epoch_time_minutes`. Per-epoch `val_qwk`: 0.9083, 0.8970, 0.9070, 0.9055, 0.9039, 0.9146, **0.9153** (epoch 7 = best).

### 5.4 Honest framing of the recorded metrics

Every number in §5.2/§5.3 (QWK, macro-F1, accuracy, macro-AUC, epoch counts) is **training-time validation metadata stored inside the checkpoint files by the original training run**. This project has not reproduced, re-measured or confirmed any of them. They are reported here only as provenance of the supplied weights.

### 5.5 What the checkpoints do *not* store, and how the project handles it

Documented at `model_architecture.py:81-97`:

- `num_classes` is read from the head's `out_features` (5) and is **never widened or narrowed** to make a load succeed.
- BatchNorm `eps`/`momentum` and the dropout probability are Python hyperparameters that no `state_dict` can hold; torchvision defaults are used (`DROPOUT_PROB = 0.2`). Dropout is the identity in `eval()`, so it cannot change an inference output.
- No pooling parameters are stored; `AdaptiveAvgPool2d(1)` is added because a 1280-d head can only consume pooled features, and it adds no keys.
- The CFP checkpoint stores **no preprocessing config**, so CFP input size (224) and normalisation are documented **assumptions**, flagged as such at runtime by a warning log. UWF's 512 px is checkpoint-verified (`verified=True`).

---

## 6. Model Architecture Reconstruction

Implemented in **`backend/inference/model_architecture.py` (224 lines)**, which is the single source of truth for construction. Loading and I/O live in separate modules, so the architecture can be verified in isolation.

### 6.1 Method: the architecture was derived from the checkpoints, not guessed

The module documents the evidence chain used to identify the backbone as torchvision (and to rule out `timm`):

| Observation in the state_dict | What it establishes |
|---|---|
| `features.0.0.weight` = `(32, 3, 3, 3)` | stem `Conv2d(3, 32, k=3, s=2, p=1)` → 3-channel RGB |
| `features.0.1.*` | `BatchNorm2d(32)` directly after the stem |
| `features.0 … features.8` | 9 top-level MBConv stages |
| `features.<i>.0.block.<j>.*` | MBConv / ConvBNActivation blocks |
| `features.<i>.1.block.<j>.*` | fused stages (`features.1 … features.7`) |
| 64 SE tensors at `features.*.block.2.fc1/.fc2` | Squeeze-Excitation |
| `num_batches_tracked` present 49 times | exactly torchvision B0's BatchNorm count |
| `features.8.0.weight` = `(1280, 320, 1, 1)` | final conv → 1280-d pooled features |
| 358/358 trunk keys match `efficientnet_b0(weights=None).features` in **name, shape and order** | origin = `torchvision.models.efficientnet_b0` |
| zero `blocks.*`, `conv_head.*`, `bn1.*` keys | `timm` is **excluded** |

Constants declared: `INPUT_CHANNELS = 3`, `BACKBONE_CHANNELS = 1280`, `NUM_CLASSES = 5`, `NUM_BINARY_HEADS = 7`, `EXPECTED_TRUNK_TENSORS = 358`, `DROPOUT_PROB = 0.2`.

### 6.2 Constructed modules

- **`create_efficientnet_b0_trunk()`** — `efficientnet_b0(weights=None).features`. `weights=None` is mandatory so no pretrained download is attempted and the checkpoint stays the only source of parameters. **Both modalities are built from this one factory**, so the backbone is defined exactly once.
- **`CfpEfficientNetB0`** (CFP): `features` + `avgpool = AdaptiveAvgPool2d(1)` + `dr_head = Linear(1280, 5)` + `lesion_heads = ModuleList([Linear(1280, 1)] * 7)`. `forward` returns `(dr_logits, auxiliary_binary_logits)`.
- **`create_uwf_model()`** (UWF): `efficientnet_b0(weights=None, num_classes=5)` — trunk plus stock `classifier = Sequential(Dropout, Linear(1280, 5))`.

The two construction functions return **separate instances**; nothing is shared or aliased between CFP and UWF (`model_architecture.py:100-106`).

### 6.3 Verification result (executed, exit code 0)

`python -m inference.verify_architecture --strict-load` reports for **both** checkpoints:

```
names identical      : True
shapes identical     : True
key order identical  : True
missing keys         : 0
unexpected keys      : 0
shape mismatches     : 0
strict load_state_dict(strict=True)  ->  All keys matched successfully.
VERDICT: EXACT MATCH - architecture CAN be reconstructed exactly
```

| Label | Class | Params | Ckpt tensors | Model tensors | Verdict |
|---|---|---|---|---|---|
| CFP | `CfpEfficientNetB0` | 4,022,920 | 374 | 374 | **EXACT MATCH** |
| UWF | `EfficientNet` | 4,013,953 | 360 | 360 | **EXACT MATCH** |

Spot-checks confirmed identical model-vs-checkpoint shapes for `features.0.0.weight`, `features.8.0.weight`, `dr_head.weight`, `lesion_heads.0/6.weight` and `classifier.1.weight`.

> **Counting note (avoids an apparent contradiction):** the parameter figures above are `sum(p.numel() for p in model.parameters())` = **4,022,920** (CFP) and **4,013,953** (UWF). The inspection tool separately reports *"total parameters 4,064,985 / 4,056,018"*, which is the element count of **all** state_dict tensors including BatchNorm buffers. Both numbers are correct; they measure different things. The 8,967-parameter CFP-minus-UWF delta equals the 7 auxiliary heads (`7 × (1280 + 1)`).

---

## 7. Model Loading Verification

Implemented in **`backend/inference/model_loader.py` (648 lines)**, verified by **`backend/inference/test_model_loading.py`**.

### 7.1 Loading policy (deliberately unforgiving)

1. `torch.load(path, map_location="cpu", weights_only=True)` under a narrow `safe_globals` allow-list (NumPy scalars/dtypes + `TorchVersion`, required because the files store `np.float64` and a `TorchVersion`). Unsafe unpickling is never used.
2. A fully pickled `nn.Module` is **refused** (`ArchitectureMismatchError`) because its architecture could not be verified.
3. Weights are taken from `model_state_dict`.
4. Architecture is built with `weights=None`; the checkpoint is the only parameter source.
5. **`load_state_dict(state_dict, strict=True)`.** `strict=False` is never used anywhere in the project. On failure, loading aborts and the original error is re-raised verbatim; the architecture is never adjusted to force a load. Rationale recorded in code: a partial load would leave randomly initialised weights inside the network and produce confident nonsense.
6. `model.to(torch.device("cpu"))`, then `model.eval()`, then **every parameter *and* buffer** is asserted to be on the CPU. Checking only `parameters()` would leave BatchNorm `running_mean`/`running_var` unverified (`model_loader.py:394-403`).
7. One zero-filled `(1, 3, 64, 64)` CPU probe confirms the real output dimensions — a *structural shape check, not prediction*. No image is read.
8. `ModelManager` loads each checkpoint exactly once under a lock, is re-entrant and idempotent, and keeps CFP and UWF **completely separate**: a failure on one modality never triggers a silent fallback to the other.

### 7.2 Executed verification result (`python -m inference.test_model_loading`, exit code 0)

```
                      CFP                         UWF
Loaded                YES                         YES
Device                cpu                         cpu
Eval mode             True                        True
Parameters                4,022,920        4,013,953
Classification outputs            5                5
Lesion heads                      7              n/a
```

Per-modality detail:

| Check | CFP | UWF |
|---|---|---|
| Checkpoint present | YES | YES |
| Object type | `nn.Module` | `nn.Module` |
| Strict load | `strict=True`, **374 tensors from `model_state_dict`, 0 missing, 0 unexpected** | `strict=True`, **360 tensors from `model_state_dict`, 0 missing, 0 unexpected** |
| Device (params **and** buffers) | cpu | cpu |
| Parameters not on CPU | **0** | **0** |
| Eval mode | True | True |
| Classification outputs | 5 | 5 |
| Head | `dr_head = nn.Linear(1280, 5)` | `classifier = Sequential(Dropout(p=0.2), Linear(1280, 5))` |
| Auxiliary binary heads | 7 | n/a |
| Checkpoint metadata | `epoch=12`, `best_val_qwk=0.9225706931411989` | `epoch=7`, `best_val_qwk=0.9152520275167323`, `config` present |

Terminal report of the script:

```
- CFP weights loaded successfully: YES
- UWF weights loaded successfully: YES
- CFP device: cpu        - UWF device: cpu
- CFP output classes: 5  - UWF output classes: 5
- CFP lesion heads count: 7
- Eval mode status: CFP=True, UWF=True
- CUDA availability: False
- Any error: none
```

Both checkpoints load into the reconstructed architectures with `strict=True`, are in eval mode, and every parameter and buffer is on the CPU. **No image was read and no prediction was made.**

### 7.3 Preprocessing layer — implemented and available, but never exercised on a real image

`inference/preprocessing.py` (312 lines) is complete: EXIF transpose, RGB conversion, format allow-list (JPEG/PNG/BMP/TIFF/WEBP), bilinear antialiased resize, ImageNet normalisation, output `(1, 3, S, S)`. It refuses training-time augmentation. Effective configuration as reported at runtime:

| Modality | Size | Mean / Std | Source | Verified? |
|---|---|---|---|---|
| CFP | **224** | ImageNet | ASSUMED — checkpoint has no `config` | **No** (logged as a warning) |
| UWF | **512** | ImageNet | read from `EfficientNetB0_UWF_final.pt` `config` | Yes |

### 7.4 Prediction function — implemented but **unverified**

`inference/predictor.py` implements `predict()` end-to-end: modality resolution → model load → preprocessing → CPU forward → softmax → argmax/label/confidence → full 5-class probability dictionary, with the ICDR mapping and correct error handling.

It has, however, **never been executed on a real image**:

- `backend/test_images/` does not exist; the repository contains no retinal image of any kind.
- `python test_prediction.py` (no `--image`) exits with the message *"INFERENCE NOT TESTED … No image was invented or synthesised, so no prediction is reported."*
- The script's own module docstring states it implements *"no API, no Grad-CAM and no metrics"*.
- `main.py` contains no import of `predictor` at all, so the API cannot and does not call it.

**Therefore no prediction result, accuracy figure or per-class output exists for this project.** The 10 contract assertions written into `test_prediction.py:96-166` are unexecuted.

---

## 8. CPU-Only Setup and Verification

### 8.1 Mechanism

- `requirements.txt` pins `torch==2.14.0+cpu` and `torchvision==0.29.0+cpu` and adds the PyTorch CPU wheel index so the `+cpu` builds are resolved rather than the CUDA defaults.
- `README.md` states Python 3.10/3.11 and *"No NVIDIA GPU — everything runs on CPU, no CUDA packages are installed."* Confirmed: `pip list` contains no `nvidia-*` or `triton` packages.
- Device selection is hard-coded, not configurable: `DEVICE = "cpu"` (`main.py:56`), `CPU_DEVICE = torch.device("cpu")` (`model_loader.py:84`).

### 8.2 Verified on this machine

| Check | Result |
|---|---|
| `torch.cuda.is_available()` | **`False`** |
| `torch.version.cuda` | **`None`** (no CUDA build present) |
| Installed torch / torchvision | `2.14.0+cpu` / `0.29.0+cpu` |
| Parameters not on CPU after load | **0** for both models |
| Buffers on CPU after load | asserted for both models (BatchNorm statistics included) |
| Device used by the API | `cpu` (hard-coded, never selected at runtime) |
| Live `GET /health` response | `{"status":"healthy","device":"cpu","cuda_available":false}` |

### 8.3 Training-side vs inference-side device

The checkpoints record `device = cuda`, `gpu_count = 2`, `pytorch_version = 2.10.0+cu128`, `torchvision_version = 0.25.0+cu128`, `cuda_version = 12.8`. Those CUDA builds are **not required and not installed**; the tensors are `float32` and load with `map_location="cpu"` (`model_architecture.py:94-97`).

### 8.4 Enforced, not merely documented

CPU-only operation is defended in code, not just in prose: `map_location="cpu"` at load; an explicit `.to(CPU_DEVICE)`; a per-tensor device assertion that raises `ArchitectureMismatchError` naming the offending parameter; `torch.no_grad()` around every forward; and a docstring in `main.py:22-24` noting that because the API loads no model, *"this app starts instantly and cannot touch CUDA."*

---

## 9. FastAPI Backend Status

**Status: a working but deliberately non-inferential skeleton.** It starts, serves three endpoints and refuses to touch the networks.

| Item | State |
|---|---|
| App | `backend/main.py`, 206 lines, `API_VERSION = "0.3.0-dev"` |
| Title / OpenAPI version | `RetinaGrade AI API` / `0.3.0-dev` (read from live `/openapi.json`) |
| Models loaded at startup | **None.** There is no `lifespan` hook; `inference/` is not imported |
| CORS | `CORSMiddleware`, allow-listed localhost origins + `^http://(localhost\|127\.0\.0\.1)(:\d+)?$` regex, methods `GET, POST, OPTIONS`, headers `*` |
| Error handling | global `Exception` handler → HTTP 500 with a generic message; tracebacks are logged server-side only and never returned |
| Pydantic schemas | `RootResponse`, `HealthResponse`, `PredictDevResponse`, `ErrorBody`/`ErrorResponse` (`schemas/api.py`) |
| Upload cap | `MAX_UPLOAD_BYTES = 25 MB` (matches the frontend constant) |
| Prediction schemas | **Absent by design** — `schemas/api.py:3-7` explains that leaving them would imply inference that does not exist |
| `services/` and `utils/` | Empty `__init__.py` only |

### 9.1 Live verification (uvicorn started, requests issued, then stopped)

```
GET  /            -> 200 {"app":"RetinaGrade AI","status":"running"}
GET  /health      -> 200 {"status":"healthy","device":"cpu","cuda_available":false}
GET  /openapi.json-> paths: /, /health, /predict   info.version: 0.3.0-dev
```

Startup log: `Waiting for application startup. / Application startup complete. / Uvicorn running on http://127.0.0.1:8124`. No model was loaded and no checkpoint file was read by the API process.

### 9.2 Important version-control caveat

`git diff` shows `backend/main.py` and `backend/schemas/api.py` are **modified in the working tree but not committed**, and the modification *removes* a previously written inference-enabled API:

| Capability | In commit `759dc60` | In the current working tree |
|---|---|---|
| API version | `0.2.0` | `0.3.0-dev` |
| `GET /models` (per-modality status, class list, preprocessing) | present | **removed** |
| Lifespan startup model preload via `manager.describe()` | present | **removed** |
| `InferenceError` → HTTP handler | present | **removed** |
| `POST /predict` calling `predictor.predict` | present | **replaced by validation-only stub** |
| `PredictResponse`, `PredictionResultModel`, `ModelsInfoResponse` schemas | present | **removed** |

Any statement about the API must therefore be read against the *working tree*, which is what this report describes. The inference-wired version exists only in Git history and would need to be reinstated (not merely retyped) before a prediction endpoint exists.

---

## 10. Current API Endpoints

Exactly **three** endpoints, confirmed against the live OpenAPI document. **None of them runs a neural network.**

| Method | Path | Purpose | Model run? |
|---|---|---|---|
| `GET` | `/` | Service banner (`app`, `status`) | **NO** |
| `GET` | `/health` | Liveness + device facts (`status`, `device`, `cuda_available`) | **NO** |
| `POST` | `/predict` | **Validates the request only**, returns a temporary placeholder | **NO** |

`POST /predict` accepts `image: UploadFile` (file field) and `modality: str` (form field) and performs exactly five validation steps (`main.py:155-166`):

1. `modality` ∈ {`cfp`, `uwf`}, case-insensitive, whitespace-trimmed
2. an `image` file field was actually sent
3. the file has a filename
4. the file is not empty
5. the file is within the 25 MB limit

The image bytes are read **only** to test emptiness and size. They are not decoded, not preprocessed and not passed to any network.

### 10.1 Live behaviour of `POST /predict` (verified with curl)

| Request | Response |
|---|---|
| valid modality `CFP` + **19 bytes of plain text (not an image at all)** | **HTTP 200** `{"success":true,"message":"Image received","modality":"CFP"}` |
| modality `oct` | **HTTP 400** `{"success":false,"error":{"code":"unsupported_modality","message":"Unsupported modality 'oct'. Allowed values: cfp, uwf."}}` |

The 200 response for non-image bytes is direct proof that the endpoint performs no decoding and no inference. Declared error mappings: `400` invalid modality, `422` missing/empty image, `413` image too large, plus the global `500` handler.

### 10.2 Endpoints that do not exist

`GET /models` (per-modality load status), any prediction-returning `POST /predict` contract, Grad-CAM endpoints, report-generation endpoints, history/CRUD endpoints, authentication. `ModelManager.describe()` and `describe_preprocessing()` exist and are ready to back a `/models` endpoint, but nothing calls them today.

---

## 11. Frontend Status

**Status: a complete-looking, lint-clean, but entirely non-functional UI shell.** It is a static representation of the intended workflow, with every result field deliberately left empty.

### 11.1 What exists

- **Stack:** React 19 + TypeScript 5.9 + Vite 8 + Tailwind CSS 4; SPA with client-side view switching in `App.tsx` (no router library — `activeItem` is React state).
- **Views:** `AnalyzePage` (primary workflow), `DashboardPage`, `ExplainabilityPage`, `ReportsPage`, plus `UnavailablePage` for **Screening History** and **Settings**.
- **Shared state in `App.tsx`:** modality (`cfp`/`uwf`), selected file + local preview URL (`useImageSelection`), and `status: 'idle' | 'analyzed'`. Changing modality or image resets the result panels; state lives above the pages so switching to Explainability/Reports preserves the same capture.
- **Reusable UI:** Card/Badge/Button/EmptyState/Placeholder/Skeleton/Notice/ProgressBar and a local icon set.
- **Production build:** `frontend/dist/` exists with `index.html`, `retina.svg`, a 268,520 B JS bundle and a 35,183 B CSS bundle, and `node_modules/.tmp/*.tsbuildinfo` artifacts are present — i.e. `tsc -b` and `vite build` have been run successfully at some point.

### 11.2 What is deliberately inert

- **No API integration of any kind.** A full-text search of `frontend/src` for `fetch(`, `axios`, `XMLHttpRequest` returns **no matches**. `BACKEND_BASE_URL = 'http://127.0.0.1:8000'` exists as a constant in `data/clinical.ts:6` but is never used to make a request. The Analyze button simply flips `status` to `'analyzed'`.
- **Every result field is `null`.** `types.ts:41-43` states `label`, `confidence` and `probabilities` are `null` until real inference is wired up and "are never faked".
- `PredictionCard` renders `—` for grade and confidence and the fixed caption *"Inference is not connected in this build."*
- `XaiPanel` has Original/Grad-CAM/Overlay tabs, but the Grad-CAM and Overlay tabs render an explicitly empty state: *"The heatmap is still empty: Grad-CAM is not implemented."* All Grad-CAM metadata fields (`Target layer`, `Alpha`, `Max activation`, `Upsample`) are `-`.
- `AiReportCard` shows a Gemini generator panel whose "Generate report" and "Export PDF" buttons are hard-coded `disabled`, with the notice *"Report generation is not connected."*
- `DashboardPage` shows four statistics with no data source and a "Recent Studies" empty state, under the banner *"All figures on this page are placeholders."*
- Image handling is **local only** (drag-and-drop, `URL.createObjectURL` preview, client-side type/size validation mirroring the 25 MB cap). "Analyze Image" is `disabled` until a file is chosen.

### 11.3 Verified quality state

`npm run lint` (`eslint .`) → **exit code 0, no warnings or errors.**

### 11.4 Status gaps

- **The whole frontend is untracked in Git.** It is on disk but nothing is committed — a single `git clean` with the wrong flags would destroy it.
- `frontend/node_modules/` and `frontend/dist/` are untracked **and not git-ignored**; `node_modules` should be added to `.gitignore`.
- The README and `.gitignore` still describe a Flutter client that does not exist (§3.2).
- `ProbabilityBars.tsx` and `UnavailablePage` routing exist; the app has no router, no error boundary and no state-management layer.
- No frontend unit or component tests, and no test runner in `package.json`.

---

## 12. What Is NOT Implemented Yet

Each item below was confirmed absent from the working tree. Nothing in this list is functional today.

### 12.1 Inference

- **No endpoint performs image prediction.** `POST /predict` validates and returns a placeholder. `predictor.predict` is implemented but unreachable from the API.
- **End-to-end inference has never been run.** No retinal image exists in the repository, so `test_prediction.py` reports `INFERENCE NOT TESTED`. There is no verified prediction, no verified softmax output, and the ten output-contract assertions are unexecuted.
- **No accuracy, QWK, F1, AUC or confusion matrix has been produced by this project** — no evaluation script, no ground truth, no dataset.
- **The CFP preprocessing input size is an unverified assumption** (224 vs the sibling's 512). Because EfficientNet-B0 is fully convolutional with global pooling, a wrong size fails *silently* — logits are still produced but accuracy would be untrustworthy. `preprocessing.py:31-38` and `test_prediction.py:197-204` both flag this explicitly, and `--image-size` exists to probe it.
- **The 5-class label ordering is a project assumption**, not checkpoint data (§4.2).

### 12.2 Explainability

- **No Grad-CAM.** No activation hooks, no gradient capture, no target-layer selection, no heatmap rendering, no saliency or class-activation maps — in the backend or the frontend. `XaiPanel` is an empty shell.

### 12.3 Reporting and persistence

- **No Gemini integration.** `google-genai` is not in `requirements.txt` and not installed; `GEMINI_API_KEY=` is empty in `.env`; `backend/services/` is empty. No prompt, no generated text, no `REPORT_META` values.
- **No screening history and no database.** No storage layer, no ORM, no migration, no persistence of results.
- **No PDF export.** `reportlab` is not installed; the "Export PDF" button is permanently disabled.

### 12.4 Backend and integration gaps

- `GET /models` removed and not reinstated; no endpoint reports model load status, DR class list or effective preprocessing.
- No `InferenceError` → HTTP mapping (domain exceptions define `code`/`http_status` but nothing consumes them).
- No request logging, no rate limiting, no authentication, no multi-user support, no CORS beyond localhost.
- No real image decoding in the upload path, so content-type/format rejection happens only in the unused `preprocessing` layer.

### 12.5 Quality, testing and documentation

- **No automated test suite.** `pytest` and `httpx` are not installed, there is no `tests/*.py`, no test config, and `tests/` contains only `.gitkeep`. The verification scripts are hand-run CLIs with exit codes, not a test runner.
- **No CI configuration** of any kind (no `.github/`, no pipeline).
- **No linting or type-checking for Python** — no `ruff`, `flake8`, `mypy` or `black` config anywhere.
- `docs/` is empty; no architecture document, no dataset description, no API contract document, no setup guide beyond the outdated README.
- **README is stale** in three ways: Flutter frontend, obsolete checkpoint names (`efficient_cfp.pt` / `efficient_uwf.pt` instead of `EfficientNetB0_CFP_final.pth` / `EfficientNetB0_UWF_final.pt`, as `clinical.ts:146` notes), and a status checklist that still marks model loading as unfinished.
- **Uncommitted work at rest:** the modified `main.py`/`schemas/api.py` and the entire untracked frontend are not preserved in Git history.

---

## 13. Current Development Workflow

### 13.1 Backend

```powershell
cd backend
.\.venv\Scripts\python.exe -m pip install -r requirements.txt     # CPU wheels via --extra-index-url
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
# interactive OpenAPI docs at http://127.0.0.1:8000/docs
```

### 13.2 Verification scripts (all read-only; none writes a file)

| Command (from `backend/`) | Purpose | Verified result |
|---|---|---|
| `python -m inference.inspect_models` | Read-only checkpoint forensics: types, key prefixes, shapes, head layout, class count | exit 0; `file unchanged by read: True` for both |
| `python -m inference.verify_architecture` | Architecture ↔ checkpoint name/shape/order comparison, no weights loaded | exit 0 |
| `python -m inference.verify_architecture --strict-load` | Adds the real `strict=True` load as final proof | exit 0; *"All keys matched successfully"* for both |
| `python -m inference.test_model_loading` | `ModelManager` end-to-end: CPU placement, eval mode, strict load, head widths | **exit 0**, *"Any error: none"* |
| `python test_models.py` | Human-readable model report; `--image` optional | exit 0; `RESULT: CFP=OK  UWF=OK`; *"No --image supplied: image inference NOT tested."* |
| `python test_prediction.py --image <path> --modality cfp\|uwf [--image-size N] [--json]` | Real-image inference + 10 output-contract assertions | **cannot pass yet** — no image in repo; exits non-zero with `INFERENCE NOT TESTED` |

### 13.3 Frontend

```powershell
cd frontend
npm install
npm run dev        # vite dev server on 127.0.0.1:5173
npm run lint       # eslint .            -> verified exit 0
npm run typecheck  # tsc -b --noEmit
npm run build      # tsc -b && vite build
npm run preview
```

### 13.4 Working practices observed in the codebase

- **Evidence-first engineering:** every module docstring cites the checkpoint evidence for its constants (`model_architecture.py`, `preprocessing.py`, `model_loader.py`), and assumptions are labelled as assumptions at the point of use.
- **Fail loudly, never silently:** `strict=True` is never relaxed; architecture is never bent to fit a checkpoint; auxiliary heads are never named; stale config fields are documented rather than trusted.
- **CPU-only is enforced in code**, not merely documented (§8.4).
- **Nothing invents a result:** missing images, unverified preprocessing and absent datasets produce explicit "NOT TESTED" / warning messages instead of plausible-looking output.
- **Layered separation:** architecture → loading → preprocessing → prediction → API, with each verification concern in its own CLI script.

### 13.5 Git workflow as it actually stands

6 commits: `first commit` → `set backend and cpu only pytourch set up` → `Model Checkpoint Inspect` → `architecture reconstruction` ×2 → `prepar for prediction`. Current state: 24 tracked files; `backend/main.py` and `backend/schemas/api.py` modified but uncommitted; the entire `frontend/` untracked; model checkpoints correctly git-ignored (`backend/models/*` with `!backend/models/.gitkeep`).

---

## 14. Next Planned Steps

Ordered by dependency. Items marked *(inferred)* are implied by existing placeholders — `services/__init__.py`, `GEMINI_API_KEY` in `.env`, the README checklist and the disabled UI buttons — not by any specification document.

### Immediate — close the correctness gaps

1. **Resolve the CFP input size.** Run `test_prediction.py --image-size 224` and `--image-size 512` on the same real CFP image and compare output stability; then confirm against the original training script. Until then, CFP preprocessing stays flagged `verified=False`.
2. **Confirm the 5-class label ordering** from the training script or dataset metadata. Highest priority: a wrong order silently produces clinically inverted output.
3. **Obtain a small set of real, consented CFP and UWF test images** (no synthetic images) and run `test_prediction.py` to close the inference-verification gap. Do not report predictions before this passes.
4. **Commit the current working tree**, or explicitly decide to reinstate the inference-wired API from commit `759dc60`. Leaving the frontend untracked is the single largest data-loss risk in the repository.

### Near term — wire the product together

5. **Reinstate real inference on the API** *(inferred)*: add `POST /predict` calling `predictor.predict`, map `InferenceError` subclasses to their `http_status`/`code`, restore the lifespan model preload, restore the prediction response schemas, and re-add `GET /models` using the existing `ModelManager.describe()` and `describe_preprocessing()`.
6. **Connect the frontend to the API** *(inferred)*: an API client module using the already-declared `BACKEND_BASE_URL`, upload with progress, render real grade / confidence / probabilities into the existing `null` fields, and replace the "Model Status: Not loaded" badges with live status.
7. **Add a Python test suite** *(inferred)*: `pytest` + `httpx`, a `TestClient` suite over the three endpoints, plus regression tests for `predictor.verify_result`'s ten output-contract assertions using a fixed real image.

### Medium term — feature completion

8. **Grad-CAM** *(inferred)*: activation/gradient hooks on the final EfficientNet stage (`features.8`), per-class heatmaps, and endpoints returning overlay images to the existing `XaiPanel` tabs.
9. **Gemini report generation** *(inferred)*: install `google-genai`, add a service under `backend/services/`, prompt from the prediction payload only (never from raw auxiliary head outputs), and populate `AiReportCard`'s `REPORT_META` fields.
10. **Screening history and PDF export** *(inferred)*: persistence layer plus `reportlab`, replacing the `UnavailablePage` for History and enabling the Export button.
11. **Dataset and evaluation work for MMRDR** *(inferred from the project brief)*: locate and document the actual dataset, record the true label encoding, and build an evaluation harness so that QWK/macro-F1 can be reproduced on this hardware rather than quoted from checkpoint metadata.

### Housekeeping

12. Update `README.md`: React frontend (not Flutter), real checkpoint filenames, current status checklist, and a link to this report.
13. Extend `.gitignore` with `node_modules/`, `dist/` and coverage output; add a `frontend/dist` build-artifact decision.
14. Add Python quality tooling (`ruff`, `mypy`) and CI running the backend verification scripts plus `npm run lint` / `typecheck` / `build`.
15. Populate `docs/` with the architecture note, the API contract and the dataset description.

---

## Appendix A — Verification Evidence

All commands were run from `F:\retinagrade-ai\backend` (frontend lint from `F:\retinagrade-ai\frontend`) using `.\.venv\Scripts\python.exe` / `npm`. **No source file was modified in the course of preparing this report.**

| # | Command | Result |
|---|---|---|
| A.1 | Recursive file inventory (`Get-ChildItem -Recurse`, excluding `.git`, `node_modules`, `dist`, `__pycache__`) | 69 project files; 2 checkpoints (16,408,231 B / 48,652,882 B) |
| A.2 | `git log --oneline`, `git status --short`, `git ls-files` | 6 commits; 24 tracked files; `main.py` + `schemas/api.py` modified; `frontend/` untracked |
| A.3 | `git diff backend/main.py backend/schemas/api.py` | Confirms the uncommitted rollback from the inference-wired `0.2.0` API to the `0.3.0-dev` skeleton |
| A.4 | `python -c "import ...; print(torch.__version__, torch.cuda.is_available())"` | Python 3.10.11, torch 2.14.0+cpu, torchvision 0.29.0+cpu, fastapi 0.141.1, pillow 12.3.0, numpy 2.2.6, cv2 5.0.0, **`cuda_available = False`** |
| A.5 | `python -m inference.inspect_models` | exit 0 — full checkpoint forensics for both files (§5, §4.1); `file unchanged by read: True` |
| A.6 | `python -m inference.verify_architecture --strict-load` | exit 0 — EXACT MATCH for both; 0 missing / 0 unexpected / 0 shape mismatches; strict load OK |
| A.7 | `python -m inference.test_model_loading` | **exit 0** — both loaded, cpu, eval, `strict=True`, 0 missing/unexpected; *"Any error: none"* |
| A.8 | `python test_models.py` | exit 0 — `RESULT: CFP=OK  UWF=OK`; `Parameters: 4,022,920 / 4,013,953`; `epoch=12/7`; `best_qwk=0.92257 / 0.91525`; *"image inference NOT tested"* |
| A.9 | `python -m pip list` | No CUDA packages, no `google-genai`, no `reportlab`, no `pytest`, no `httpx` |
| A.10 | `torch.load(...)` dump of both checkpoints' metadata | CFP top-level keys / 12-row history; UWF 27-key `config`, 9-key `history` for epochs 1–7 (§4.1, §5) |
| A.11 | `uvicorn main:app` started; `GET /`, `GET /health`, `GET /openapi.json` | 200 / 200 / 200; paths `/`, `/health`, `/predict`; version `0.3.0-dev`; no model loaded |
| A.12 | `curl -F image=@<19-byte text file> -F modality=CFP` → `POST /predict` | **HTTP 200** `{"success":true,"message":"Image received","modality":"CFP"}` → proof that no decoding or inference occurs |
| A.13 | `curl -F image=@<file> -F modality=oct` → `POST /predict` | **HTTP 400** `unsupported_modality` |
| A.14 | `npm run lint` (frontend) | **exit 0** — no errors, no warnings |
| A.15 | Full-text search for `MMRDR`, `Messidor`, `IDDr`, `DeepDR`, `APTOS`, `Kaggle`, `dataset used` | **No matches** anywhere in the project |
| A.16 | Full-text search of `frontend/src` for `fetch(`, `axios`, `XMLHttpRequest`, `EventSource` | **No matches** → no API integration |
| A.17 | Inventory of `frontend/dist`, `*.tsbuildinfo`, `tests/`, `docs/`, `backend/services/`, `backend/utils/` | Prior production build present; `tests/`, `docs/`, `services/`, `utils/` contain only `.gitkeep` / empty `__init__.py` |

### Verified-fact summary

- **CFP:** EfficientNet-B0 trunk (358 `features.*` tensors) + `dr_head` = `nn.Linear(1280 → 5)` + 7 unnamed binary heads `nn.Linear(1280 → 1)`; 374 checkpoint tensors; 4,022,920 parameters; `epoch 12`, `best_val_qwk 0.9225706931411989`; loads with `strict=True`, 0 missing / 0 unexpected keys; on CPU, eval mode.
- **UWF:** torchvision `efficientnet_b0(num_classes=5)` — trunk + `classifier = Sequential(Dropout(0.2), Linear(1280 → 5))`; 360 checkpoint tensors; 4,013,953 parameters; `epoch 7`, `best_qwk 0.9152520275167323`, `image_size 512`, `normalization 'ImageNet mean/std'`; loads with `strict=True`, 0 missing / 0 unexpected keys; on CPU, eval mode.
- **CPU:** `torch.cuda.is_available() = False`, `torch.version.cuda = None`, torch `2.14.0+cpu` / torchvision `0.29.0+cpu`; 0 parameters off CPU in either model; API hard-coded to `device="cpu"`; live `/health` reports `cuda_available: false`.
- **Strict weight loading:** successful for **both** models, 0 missing and 0 unexpected keys each, verified twice — once by `verify_architecture --strict-load` and once by `test_model_loading`.
- **Not verified:** any prediction, Grad-CAM, Gemini output, dataset metrics or working UI result.

---

*Prepared for project review. All statements are traceable to the repository contents or to the executed commands in Appendix A. Where evidence was absent, the report records the absence rather than filling the gap.*