# AI-SPEAK (Serbian) adaptation — guide

The final task: adapt the AV-HuBERT VSR model to Serbian using the AI-SPEAK corpus,
via LoRA (parameter-efficient), on the NTP-122 machine where the licensed data lives.

## Dataset facts (from the AI-SPEAK spec)
- 30 speakers (15F/15M), ~80 Serbian + 80 English utterances each. License CC BY-NC-SA 4.0.
- Layout: `spkXX/` with an `.xlsx` (metadata), `alignment/`, `ser/`, `eng/`.
  Each language folder has `video_a_anonymized` (frontal), `video_{r,l}_anonymized`, `audio`.
- **Transcripts are in the Excel** (`transcript` column; `language` = ser/eng; `common` flag).
- **Video is already lip-only + anonymized** (lips visible, surrounding face pixelized,
  rest black). Frontal camera = 100 fps; audio = 22.05 kHz mono WAV.

## What our scripts do (differences from LRS3)
- `aispeak_prepare.py` reads each `spkXX/*.xlsx`, filters `language == ser`, and for each
  utterance resamples the **frontal** clip to **96×96 grayscale @25fps** and audio to
  **16 kHz**. **No dlib face-detection / mouth-crop** — the video is already the lip region,
  so we just scale it. Ids are `spkXX/<name>`; split is **speaker-disjoint** (unseen
  speakers in valid/test) via `--valid-speakers` / `--test-speakers`.
- `aispeak_build_manifest.py` builds `{train,valid,test}.{tsv,wrd}`, `dict.wrd.txt`, and a
  **new Serbian SentencePiece** vocab (character_coverage handles ć/č/š/ž/đ).
- `preprocess_aispeak.sh` chains prepare → count_frames → manifest.

## Dependency
`pip install openpyxl` (Excel reader) — add to the env on NTP-122.

## Run order (on NTP-122)
```bash
# 1. preprocess (edit paths + speaker splits in the script first)
bash workspace/scripts/preprocess_aispeak.sh
# 2. LoRA fine-tune: decoder first, then encoder+decoder ablation
bash workspace/scripts/run_aispeak_lora.sh decoder
bash workspace/scripts/run_aispeak_lora.sh encdec
# 3. evaluate WER on the Serbian test split (infer_s2s.py, gen_subset=test)
```

## IMPORTANT caveat to verify on-site (partial correction expected)
The AI-SPEAK faces are **pixelized**. Two things to check on the first few clips before a long run:
1. **ROI framing** — open a produced `prepared_ser/video/spkXX/<name>.mp4` and confirm the
   lips are centered and fill the 96×96 frame. AI-SPEAK frames may include black masking
   around the lips; if the lips are small/off-center, we may need a crop (e.g. center-crop
   before scaling) rather than a plain resize. Adjust the `-vf` filter in `aispeak_prepare.py`.
2. **Filename/extension** — the script assumes `<name>.mp4` / `<name>.wav` matching the Excel
   `name` column. If AI-SPEAK uses different extensions or a subfolder, tweak `vsrc/asrc` lookup.

These are exactly the "correct only partially on-site" adjustments the project task anticipates.

## Why LoRA here (recap)
Small Serbian corpus → full fine-tuning would overfit. LoRA freezes the pretrained
(English) encoder and trains ~0.2% adapters. Decoder-first adapts the language/output side;
then `scope=all` (encdec) lets the visual representations shift toward Serbian speakers —
a clean ablation for the report.
