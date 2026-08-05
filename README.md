# AV-HuBERT × AI-SPEAK — Visual Speech Recognition

Master's project (Mašinsko učenje, FTN Novi Sad): analyse, fine-tune, and adapt
Meta's **AV-HuBERT** for visual speech recognition (lip reading) — reproduced on
an LRS3 subset, and adapted to Serbian on the AI-SPEAK corpus.

This repository holds the **code, configs, scripts, and documents** only. Datasets
(LRS3, AI-SPEAK) and model checkpoints are **not** included — they are large and/or
licensed (AI-SPEAK is CC BY-NC-SA 4.0, LRS3 CC BY 4.0).

## Code version used
- Upstream: [facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert)
  commit `258fb50e155134eec2c4b49c2ae8de267075fd18`
- fairseq submodule commit `afc77bdf4bb51453ce76f1572ef2ee6ddcda8eeb`
- Our modification (LoRA support) is in `our_code_changes/` (patch + modified file).

## Layout
```
workspace/
  configs/     training configs (full fine-tune + LoRA, single-GPU)
  scripts/     preprocess / download / train / eval / setup / lora
  experiments/ run logs + decode results (checkpoints excluded)
our_code_changes/  our LoRA modification to hubert_asr.py (patch + file)
docs/          learning guide, architecture↔code map, dataset checklists, guides
deliverables/  Uputstvo (task 4) + Izveštaj (task 6), Serbian .docx + generators
```

## Reproduce
See `docs/` (LEARNING_GUIDE.md, RUNPOD_GUIDE.md) and
`deliverables/Uputstvo_AV-HuBERT.docx` for full setup, preprocessing, training,
and evaluation instructions.

## Results (LRS3 subset — "does the code work" scale, not paper-scale)
| Run | Trainable | Test WER |
|---|---|---|
| Full fine-tune | 161M (100%) | 74.4% |
| LoRA (decoder q/v) | 295K (0.18%) | 93.6% |

High vs the paper's 26–32% by design: 3,000 clips / 3,000 steps vs 433h / 30k steps.
The goal was to validate the pipeline and the LoRA path before the Serbian adaptation.
