# AV-HuBERT — Learning Guide

A study path for understanding the model, the paper, and this codebase while you
wait for the LRS2 dataset. Work top to bottom; each part builds on the last.
Est. total: 2–4 focused sessions. Check the boxes as you go.

Everything references the pinned code you actually have:
- `av_hubert/` commit `258fb50e` · `fairseq/` submodule `afc77bdf`
- Papers: [AV-HuBERT](https://arxiv.org/abs/2201.02184) (main) ·
  [Robust AV-SR](https://arxiv.org/abs/2201.01763) (noise/AVSR)

---

## Part 0 — The one-paragraph mental model (read first)

AV-HuBERT learns to "read lips" in two phases. **Pretraining** (done for you, on
1800h of unlabeled LRS3+VoxCeleb2) teaches a shared audio-visual encoder to
predict *self-generated cluster labels* for masked frames — no transcripts
involved. **Fine-tuning** (what YOU do, on a small labeled LRS2 subset) attaches a
Transformer decoder and teaches it to emit the actual words. You download the
pretrained encoder as a checkpoint and only train the recognition part. That's
the "1000× less labeled data" story from your slides made concrete.

---

## Part 1 — Concepts & the papers  ☐

Goal: be able to explain the objective function at the whiteboard.

1. Re-read your own slides 4–10 (problem, HuBERT, architecture). You already made
   these — they're your best summary.
2. **Main paper, Sections 3–4** (method + objective). Focus on:
   - The **masked cluster prediction** loss. Write it out by hand:
     `L = Σ_masked CE(pred, cluster_id) [+ α Σ_unmasked ...]`
   - Why targets are **discrete cluster IDs** from k-means, not raw features.
   - The **iterative refinement**: cluster → train → re-cluster from better
     features → repeat (5 iterations).
   - **Modality dropout**: randomly zeroing audio OR video during pretraining so
     the model can't lean on just one stream. This is the key trick that makes the
     *visual* encoder strong.
3. **Robust paper**: skim for the **noise augmentation** idea and the AVSR setup
   (audio+video fine-tuning). You only need this for the optional AVSR experiment.

**Self-check:** Can you answer — (a) What do the cluster indices mean? (b) Why
mask? (c) Why is fine-tuning still needed if pretraining already "learned speech"?
(Answers are in `ARCHITECTURE_TO_CODE.md` and our chat history.)

---

## Part 2 — The code, in reading order  ☐

Open these in your editor alongside `docs/ARCHITECTURE_TO_CODE.md`. Don't try to
understand every line — trace the **data path** and find where each slide concept
lives. Suggested order (all under `av_hubert/avhubert/`):

1. `hubert_dataset.py` — how a training example is loaded: reads the `.tsv`
   manifest, loads mouth-ROI frames + audio, applies masking. *Find:* where video
   and audio tensors are produced.
2. `resnet.py` — the modified ResNet-18 visual frontend (slide "Ekstraktori
   obeležja"). *Find:* the class that maps a stack of mouth images → feature seq.
3. `hubert.py` — the core `AVHubertModel`. THE central file. *Find:*
   - `forward_features` — the two frontends + **channel-concat fusion** (`f^av`).
   - the masking application.
   - the Transformer encoder call.
   - the logits-vs-cluster-embeddings computation (pretraining head).
4. `hubert_criterion.py` — the **pretraining loss** (masked/unmasked CE). Short
   file; match every line to your equation.
5. `hubert_asr.py` — the **fine-tuning models**: `AVHubertSeq2Seq` (what you use)
   and `AVHubertCtc`. *Find:* how the pretrained encoder is wrapped and a
   `TransformerDecoder` is attached.
6. `decoder.py` — the Transformer decoder that emits subword tokens.
7. `hubert_pretraining.py` — the fairseq "task": ties dataset + model + criterion
   together, defines labels/vocab. You won't run pretraining, but this is where
   config fields like `is_s2s`, `labels`, `modalities` are interpreted.

**Self-check:** Point to the exact function for each box in your architecture
slide. If you can, task 4 is essentially done.

---

## Part 3 — The training framework (fairseq + Hydra)  ☐

Goal: understand what `fairseq-hydra-train` actually does with a YAML.

1. Read `workspace/configs/lrs2_base_vsr_1gpu.yaml` top to bottom — every block is
   commented. Map each section to a concept:
   - `task:` → which dataset/model/criterion + data format flags
   - `model:` → `av_hubert_seq2seq`, `w2v_path` (the pretrained checkpoint),
     `freeze_finetune_updates` (encoder frozen early)
   - `criterion:` → `label_smoothed_cross_entropy` (fine-tune loss, NOT the
     pretraining cluster loss)
   - `optimization:` → `max_update`, `lr`, `update_freq` (grad accumulation)
2. Compare it to the original `av_hubert/avhubert/conf/finetune/base_vox_30h.yaml`.
   The diff (8 GPU → 1 GPU, shorter run) is exactly what you'll justify in the
   report. See the header comment in our config for the rationale.
3. Understand the launch command in `workspace/scripts/run_finetune_lrs2.sh`:
   why `common.user_dir` points at `avhubert/`, and how `???` fields get filled on
   the command line.

**Self-check:** Explain what `update_freq: [8]` does and why we use it on a 4GB GPU.

---

## Part 4 — Hands-on now (no dataset needed)  ☐

1. **Load the pretrained model** and inspect it live. See
   `workspace/scripts/inspect_model.py` (run it once the checkpoint finishes
   downloading). You'll print the module tree and parameter count and match them to
   your slides.
2. **Read a config with fresh eyes** and try changing a value mentally: "if I set
   `modalities: [audio, video]`, what changes?" (→ AVSR instead of VSR).
3. Skim the **preprocessing scripts** you'll run later:
   `workspace/scripts/preprocess_lrs2.sh` and the two `lrs2_*.py`. Understand the 4
   stages (prepare → landmark+crop → audio+count → manifest+vocab).

---

## Part 5 — When the dataset arrives (the run plan)  ☐

1. Put LRS2 at `data/lrs2/` (must contain `mvlrs_v1/main/...mp4+.txt` and the split
   lists). See `docs/DATASET_CHECKLIST.md` to verify it's the right dataset.
2. `bash workspace/scripts/preprocess_lrs2.sh` (edit paths first).
3. `bash workspace/scripts/run_finetune_lrs2.sh` on the RTX 3050 machine.
   Watch VRAM; if OOM, drop `max_tokens` 1000→500→300 (see config comment).
4. `bash workspace/scripts/evaluate_lrs2.sh` → read WER from `decode.log`.
5. Log every run (config, checkpoint, subset size, WER, screenshots) for the
   report's Results section.

---

## Where things are

```
av_hubert/                 the pinned upstream repo (model + fairseq submodule)
workspace/
  configs/                 our single-GPU fine-tune config
  scripts/                 preprocess / download / train / eval + model inspector
  checkpoints/             pretrained Base ckpt + dlib models (downloaded)
  experiments/             training runs land here
docs/
  LEARNING_GUIDE.md        <- you are here
  ARCHITECTURE_TO_CODE.md  slide concept -> exact file/function (task 4 core)
  DATASET_CHECKLIST.md     how to confirm you have the right LRS2
data/lrs2/                 put the dataset here
```
