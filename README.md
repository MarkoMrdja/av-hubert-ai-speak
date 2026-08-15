# AV-HuBERT × AI-SPEAK — Visual Speech Recognition

Master's project (Mašinsko učenje, FTN Novi Sad): analyse, fine-tune, and adapt
Meta's **AV-HuBERT** for visual speech recognition (lip reading) — reproduced on
an LRS3 subset, and adapted to Serbian on the AI-SPEAK corpus.

This repository holds the **code, configs, scripts, and documents** only. Datasets
(LRS3, AI-SPEAK) and model checkpoints are **not** included — they are large and/or
licensed (AI-SPEAK is CC BY-NC-SA 4.0, LRS3 CC BY 4.0).

## Code version used
- Upstream [facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert)
  is included as a **git submodule** at `av_hubert/`, pinned to commit
  `258fb50e155134eec2c4b49c2ae8de267075fd18` (its nested fairseq submodule at
  `afc77bdf4bb51453ce76f1572ef2ee6ddcda8eeb`).
- Our modification (LoRA support) lives in `our_code_changes/` (patch + modified
  file) and is applied on top of the clean submodule — we do **not** vendor
  upstream's files.

## Setup
```bash
git clone --recursive https://github.com/<you>/av-hubert-ai-speak.git
cd av-hubert-ai-speak
bash our_code_changes/apply_patch.sh      # apply our LoRA change onto upstream
```
Already cloned without `--recursive`? Run `git submodule update --init --recursive`
first. Env pins: `env/requirements-frozen-macos-arm64.txt`.

## Layout
```
av_hubert/          pinned upstream submodule (model + fairseq)
our_code_changes/   our LoRA patch + apply_patch.sh (applied on top of upstream)
env/                frozen dependency pins
workspace/
  configs/          LoRA + fine-tune configs (LRS3 + AI-SPEAK)
  scripts/
    common/         lora.py, inspect_model.py, download_*, run_finetune, evaluate
    lrs3/           LRS3 subset prep (make_subset, parquet test, manifest)
    aispeak/        Serbian prep + LoRA run
    infra/          RunPod setup + sync
  experiments/      run logs + decode results (checkpoints excluded)
docs/               learning guide, architecture↔code map, RunPod/AI-SPEAK guides
deliverables/       Uputstvo (task 4) + Izveštaj (task 6), Serbian .docx + generators
```

## Reproduce
See `docs/ARCHITECTURE_TO_CODE.md` (concept→code map) and
`deliverables/Uputstvo_AV-HuBERT.docx` for setup, preprocessing, training, and
evaluation. The pipeline itself is the `workspace/scripts/{common,lrs3,aispeak}`
drivers, each with a commented CONFIG block.

## Results (LRS3 subset — "does the code work" scale, not paper-scale)
| Run | Trainable | Test WER |
|---|---|---|
| Full fine-tune | 161M (100%) | 74.4% |
| LoRA (decoder q/v) | 295K (0.18%) | 93.6% |

High vs the paper's 26–32% by design: 3,000 clips / 3,000 steps vs 433h / 30k steps.
The goal was to validate the pipeline and the LoRA path before the Serbian adaptation.
