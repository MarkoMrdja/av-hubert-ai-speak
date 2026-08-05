#!/usr/bin/env bash
# End-to-end LRS2 preprocessing driver. Runs the four stages that turn raw LRS2
# (mp4 + .txt transcripts) into the fairseq data dir AV-HuBERT fine-tunes on.
#
# Stages (each wraps a repo script or our LRS2 adapters):
#   1. lrs2_prepare.py     -> file.list, label.list, split.list
#   2. detect_landmark.py  -> 68-pt landmarks per clip     (repo script, dlib)
#      align_mouth.py       -> 96x96 mouth-ROI mp4s          (repo script)
#   3. extract 16kHz wav from each mp4                       (ffmpeg)
#      count_frames.py      -> nframes.video / nframes.audio (repo script)
#   4. lrs2_manifest.py    -> {train,valid,test}.{tsv,wrd}, dict, spm tokenizer
#
# EDIT the paths in the CONFIG block, then: bash preprocess_lrs2.sh
set -euo pipefail

# ============================ CONFIG (edit me) =============================
LRS2_ROOT=/Users/markomrdja/Repos/av-hubert-ai-speak/data/lrs2   # contains mvlrs_v1/ + split lists
SPLITS="train val test"        # add "pretrain" to include the big split
VOCAB_SIZE=1000
SUBSET_HOURS=10                 # cap train set (small-GPU friendly); "" for all
# ===========================================================================

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PREP="$HERE/../../av_hubert/avhubert/preparation"
CKPT="$HERE/../checkpoints"
FFMPEG="$(command -v ffmpeg)"

PREP="$LRS2_ROOT/prepared"
LANDMARK="$LRS2_ROOT/landmark"
VIDEO="$LRS2_ROOT/video"        # mouth ROIs
AUDIO="$LRS2_ROOT/audio"
OUT="$LRS2_ROOT/${SUBSET_HOURS:-all}h_data"

echo "== Stage 1: file.list / label.list =="
python "$HERE/lrs2_prepare.py" --lrs2 "$LRS2_ROOT" --out "$PREP" --splits $SPLITS

echo "== Stage 2a: landmark detection (dlib) =="
python "$REPO_PREP/detect_landmark.py" \
  --root "$LRS2_ROOT/mvlrs_v1" --landmark "$LANDMARK" --manifest "$PREP/file.list" \
  --cnn_detector "$CKPT/mmod_human_face_detector.dat" \
  --face_detector "$CKPT/shape_predictor_68_face_landmarks.dat" \
  --ffmpeg "$FFMPEG" --rank 0 --nshard 1

echo "== Stage 2b: mouth ROI crop =="
python "$REPO_PREP/align_mouth.py" \
  --video-direc "$LRS2_ROOT/mvlrs_v1" --landmark "$LANDMARK" --filename-path "$PREP/file.list" \
  --save-direc "$VIDEO" --mean-face "$CKPT/20words_mean_face.npy" \
  --ffmpeg "$FFMPEG" --rank 0 --nshard 1

echo "== Stage 3a: extract 16kHz wav =="
mkdir -p "$AUDIO"
while IFS= read -r fid; do
  mkdir -p "$AUDIO/$(dirname "$fid")"
  "$FFMPEG" -y -loglevel error -i "$LRS2_ROOT/mvlrs_v1/$fid.mp4" \
    -ar 16000 -ac 1 "$AUDIO/$fid.wav" </dev/null || echo "warn: audio failed for $fid"
done < "$PREP/file.list"

echo "== Stage 3b: count frames =="
python "$REPO_PREP/count_frames.py" --root "$LRS2_ROOT" --manifest "$PREP/file.list" --nshard 1 --rank 0
cat "$LRS2_ROOT/nframes.audio.0" > "$PREP/nframes.audio"
cat "$LRS2_ROOT/nframes.video.0" > "$PREP/nframes.video"

echo "== Stage 4: manifests + tokenizer =="
SUBSET_ARG=""
[ -n "${SUBSET_HOURS}" ] && SUBSET_ARG="--subset-hours $SUBSET_HOURS"
python "$HERE/lrs2_manifest.py" \
  --prepared "$PREP" --video-dir "$VIDEO" --audio-dir "$AUDIO" \
  --out "$OUT" --vocab-size "$VOCAB_SIZE" $SUBSET_ARG

echo ""
echo "DONE. Fine-tune data dir: $OUT"
echo "  data       = $OUT"
echo "  label_dir  = $OUT"
echo "  tokenizer  = $OUT/spm_unigram${VOCAB_SIZE}.model"
