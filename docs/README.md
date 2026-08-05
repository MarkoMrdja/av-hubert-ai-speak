# AV-HuBERT × AI-SPEAK — project workspace

Master's DL project: analyse, fine-tune, and evaluate AV-HuBERT for visual speech
recognition, first reproducing on an LRS2 subset, then on AI-SPEAK.

## Start here
- **`LEARNING_GUIDE.md`** — study path through the model, papers, and code while
  waiting for the dataset. Read this first.
- **`ARCHITECTURE_TO_CODE.md`** — maps every presentation concept to the exact
  file/function in this code version (the core of task 4).
- **`DATASET_CHECKLIST.md`** — how to confirm you have the correct LRS2.
- **`IZVESTAJ_skelet.md`** — final report skeleton (task 6); Metode section
  pre-filled from the verified architecture, `[POPUNITI]` marks experiment-dependent gaps.
- **`RUNPOD_GUIDE.md`** — step-by-step guide to renting + training on a RunPod GPU
  (the plan: LRS2 training runs here, ~$2–5 total).

## Code version (cite this in the uputstvo)
- `av_hubert` commit `258fb50e155134eec2c4b49c2ae8de267075fd18`
- `fairseq` submodule commit `afc77bdf4bb51453ce76f1572ef2ee6ddcda8eeb`

## Environment
Conda env `avhubert` (Python 3.9). Activate:
```bash
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh && conda activate avhubert
```
Exact pins: `av_hubert/requirements-frozen-macos-arm64.txt`. On the GPU (Linux)
machine you'll rebuild with a CUDA torch build but the same version pins.

## Dataset: LRS3 (switched from LRS2)
We use **LRS3** (the dataset the AV-HuBERT paper actually benchmarks), via the
verified HF mirror `TheNHz/ellipsis-lrs3-raw`. BBC/LRS2 never responded; professor
approved LRS3. Data at `data/lrs3_raw/` — trainval (raw mp4+txt) + test (parquet).

## Workflow
1. `workspace/scripts/download_checkpoint.sh`      (done — Base ckpt + dlib models)
2. `workspace/scripts/download_lrs3.sh`            (done — trainval + test + valid ids)
3. `workspace/scripts/preprocess_lrs3_subset.sh`   (landmarks→crop→audio→manifest; small subset)
4. `workspace/scripts/run_finetune_lrs2.sh`        (on RunPod — see RUNPOD_GUIDE.md)
5. `workspace/scripts/evaluate_lrs2.sh`            (WER on test)

## Approach (per professor, 2026-08)
- LRS3 = prove the code works (train a few iterations) + inference to inspect model.
- The important part = **adaptation to Serbian (AI-SPEAK)** via **LoRA** (`workspace/scripts/lora.py`),
  decoder-first then encoder. LoRA verified on the real 103M model (0.28% trainable).

## Legacy (LRS2 — superseded)
`preprocess_lrs2.sh`, `lrs2_*.py`, `DATASET_CHECKLIST.md` were for the LRS2 plan.
Kept for reference; the LRS3 path above replaces them.

## Hands-on now (no data needed)
```bash
python workspace/scripts/inspect_model.py   # print the real model vs your slides
```

## Layout
```
av_hubert/     pinned upstream (model + fairseq submodule)
workspace/
  configs/     single-GPU fine-tune config (+ rationale in header)
  scripts/     download / preprocess / train / eval / inspect
  checkpoints/ pretrained Base + dlib models
  experiments/ runs land here
docs/          guides (this folder)
data/lrs2/     put the dataset here
```
