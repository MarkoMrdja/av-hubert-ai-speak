# AV-HuBERT × AI-SPEAK — Visual Speech Recognition

Master's project (Mašinsko učenje, FTN Novi Sad): analyse, fine-tune, and adapt
Meta's **AV-HuBERT** for visual speech recognition (lip reading) — reproduced on
an LRS3 subset, and adapted to Serbian on the AI-SPEAK corpus.

This repository holds the **code, configs, scripts, and documents** only. Datasets
(LRS3, AI-SPEAK) and model checkpoints are **not** included — they are large and/or
licensed (AI-SPEAK is CC BY-NC-SA 4.0, LRS3 CC BY 4.0).

## Code version used
- [facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert)
  (commit `258fb50e`) and its `fairseq` dependency (commit `afc77bdf`) are
  **vendored** under `av_hubert/` — committed directly (MIT-licensed), see
  `av_hubert/PROVENANCE.md`.
- Our only modification (LoRA support) is committed directly in
  `av_hubert/avhubert/hubert_asr.py`. See the diff with `git log -p`.

## Setup
```bash
git clone https://github.com/<you>/av-hubert-ai-speak.git
cd av-hubert-ai-speak
bash workspace/scripts/infra/setup_env.sh .   # conda env + deps (any CUDA GPU machine)
```
Env pins: `env/requirements.txt` (Linux/CUDA) · `env/requirements_mac.txt` (macOS/CPU).

## Layout
```
av_hubert/          vendored upstream (av_hubert + fairseq) incl. our LoRA change
env/                requirements.txt (CUDA) + requirements_mac.txt + verify_env.py
workspace/
  configs/          LoRA + fine-tune configs (LRS3 + AI-SPEAK)
  scripts/
    common/         lora.py, inspect_model.py, download_*, run_finetune, evaluate
    lrs3/           LRS3 subset prep (make_subset, parquet test, manifest)
    aispeak/        Serbian prep + LoRA run
    infra/          env setup + remote sync
  experiments/      run logs + decode results (checkpoints excluded)
docs/               architecture↔code map, results, report skeleton
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
