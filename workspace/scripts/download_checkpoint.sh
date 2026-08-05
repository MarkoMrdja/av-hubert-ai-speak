#!/usr/bin/env bash
# Download a pretrained AV-HuBERT checkpoint + the dlib models needed for
# mouth-ROI cropping. Checkpoints are ungated direct downloads from Meta's dl.fbaipublicfiles.com.
#
# We fetch the BASE model pretrained on LRS3 + VoxCeleb2 (English) — the right
# starting point for VSR fine-tuning on a small GPU. Swap the URL for the Large
# or noise-augmented variants if you need them (see the model zoo:
#   http://facebookresearch.github.io/av_hubert ).
#
# Usage:  bash download_checkpoint.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="$HERE/../checkpoints"
mkdir -p "$CKPT_DIR"

# --- Pretrained BASE, LRS3 + VoxCeleb2 (self-supervised, no finetuning) ---
BASE_PT_URL="https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/base_vox_iter5.pt"

echo ">> downloading pretrained Base (LRS3+VoxCeleb2) checkpoint ..."
curl -L --fail -o "$CKPT_DIR/base_vox_iter5.pt" "$BASE_PT_URL"

# --- dlib face + landmark models for preparation/detect_landmark.py ---
echo ">> downloading dlib face detector + 68-landmark predictor ..."
curl -L --fail -o "$CKPT_DIR/mmod_human_face_detector.dat.bz2" \
  "http://dlib.net/files/mmod_human_face_detector.dat.bz2"
curl -L --fail -o "$CKPT_DIR/shape_predictor_68_face_landmarks.dat.bz2" \
  "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
bunzip2 -kf "$CKPT_DIR/mmod_human_face_detector.dat.bz2"
bunzip2 -kf "$CKPT_DIR/shape_predictor_68_face_landmarks.dat.bz2"

# --- mean face landmarks (used by align_mouth.py); source per repo help text ---
echo ">> downloading 20-word mean face ..."
curl -L --fail -o "$CKPT_DIR/20words_mean_face.npy" \
  "https://raw.githubusercontent.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/master/preprocessing/20words_mean_face.npy"

echo ""
echo "done. checkpoints/ now contains:"
ls -la "$CKPT_DIR"
