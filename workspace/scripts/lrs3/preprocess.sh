#!/usr/bin/env bash
# Preprocess a SMALL subset of LRS3 trainval end-to-end into AV-HuBERT format.
# Purpose: the "prove the code works" run (professor: pokrenite obuku nekoliko
# iteracija). A few hundred clips is plenty — we are NOT reproducing the full paper.
#
# Chain (trainval = raw 224x224 face mp4 + .txt):
#   0. build subset file.list / label.list from trainval/
#   1. detect_landmark.py  -> 68-pt landmarks (dlib)         [repo script]
#   2. align_mouth.py      -> 96x96 mouth ROI mp4            [repo script]
#   3. ffmpeg              -> 16 kHz wav per clip
#   4. count_frames.py     -> nframes.video / nframes.audio  [repo script]
#   5. merge in the parquet TEST split (already-cropped ROIs)
#   6. lrs3_manifest-style tsv/wrd/dict/spm build
#
# Edit N_TRAIN if you want more/fewer clips, then: bash preprocess_lrs3_subset.sh
set -euo pipefail

# ============================ CONFIG ======================================
LRS3=/Users/markomrdja/Repos/av-hubert-ai-speak/data/lrs3_raw
N_TRAIN=300            # number of trainval clips in the subset
N_VALID=40            # held-out valid clips (from trainval, disjoint)
VOCAB_SIZE=1000
# ===========================================================================

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PREP="$HERE/../../../av_hubert/avhubert/preparation"
CKPT="$HERE/../../checkpoints"
FFMPEG="$(command -v ffmpeg)"
PREP="$LRS3/prepared_subset"
mkdir -p "$PREP"

echo "== Step 0: build subset file.list / label.list =="
python "$HERE/make_subset.py" --lrs3 "$LRS3" --out "$PREP" \
  --n-train "$N_TRAIN" --n-valid "$N_VALID"

echo "== Step 1: landmark detection (dlib) =="
python "$REPO_PREP/detect_landmark.py" \
  --root "$LRS3" --landmark "$LRS3/landmark" --manifest "$PREP/file.list" \
  --cnn_detector "$CKPT/mmod_human_face_detector.dat" \
  --face_predictor "$CKPT/shape_predictor_68_face_landmarks.dat" \
  --ffmpeg "$FFMPEG" --rank 0 --nshard 1

echo "== Step 2: mouth ROI crop (96x96) =="
python "$REPO_PREP/align_mouth.py" \
  --video-direc "$LRS3" --landmark-direc "$LRS3/landmark" --filename-path "$PREP/file.list" \
  --save-direc "$LRS3/video" --mean-face "$CKPT/20words_mean_face.npy" \
  --ffmpeg "$FFMPEG" --rank 0 --nshard 1

echo "== Step 3: extract 16kHz wav =="
mkdir -p "$LRS3/audio"
while IFS= read -r fid; do
  mkdir -p "$LRS3/audio/$(dirname "$fid")"
  "$FFMPEG" -y -loglevel error -i "$LRS3/$fid.mp4" -ar 16000 -ac 1 "$LRS3/audio/$fid.wav" </dev/null \
    || echo "warn: audio failed $fid"
done < "$PREP/file.list"

echo "== Step 4: count frames =="
python "$REPO_PREP/count_frames.py" --root "$LRS3" --manifest "$PREP/file.list" --nshard 1 --rank 0
cat "$LRS3/nframes.audio.0" > "$PREP/nframes.audio"
cat "$LRS3/nframes.video.0" > "$PREP/nframes.video"

echo "== Step 5: convert parquet TEST split (already 96x96 ROIs) =="
python "$HERE/test_from_parquet.py" \
  --parquet-dir "$LRS3/test-mattymchen/data" --out "$LRS3" --fps 25 --sr 16000

echo "== Step 6: build manifests + tokenizer =="
python "$HERE/build_manifest.py" \
  --lrs3 "$LRS3" --prepared "$PREP" --out "$LRS3/subset_data" --vocab-size "$VOCAB_SIZE"

echo ""
echo "DONE. Fine-tune data dir: $LRS3/subset_data"
