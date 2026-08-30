#!/usr/bin/env bash
# Download the AI-SPEAK corpus (Serbian + English, video + audio) from Kaggle.
#
# Dataset: tijananosek/ai-speak-database-video-and-audio  (CC BY-NC-SA 4.0)
#   https://www.kaggle.com/datasets/tijananosek/ai-speak-database-video-and-audio
# Extracts to a directory containing per-speaker folders spk01 .. spk30 — this is
# the AISPEAK_ROOT the aispeak/preprocess.sh CONFIG block points at.
#
# Prereqs:
#   pip install kaggle                    (into the avhubert env)
#   Kaggle account + API token: on kaggle.com -> Account -> "Create New API Token"
#   -> place the downloaded kaggle.json at ~/.kaggle/kaggle.json (chmod 600).
#
# Usage:  bash download_aispeak.sh [dest_dir]
set -euo pipefail

SLUG="tijananosek/ai-speak-database-video-and-audio"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$HERE/../../../data/aispeak_raw}"
mkdir -p "$DEST"

echo "== downloading AI-SPEAK from Kaggle ($SLUG) =="
echo "   dest: $DEST"

# --unzip extracts in place; the archive expands to spk01 .. spk30 speaker folders.
kaggle datasets download -d "$SLUG" -p "$DEST" --unzip

echo ""
echo "downloaded + extracted."
echo "  speakers: $DEST/spk01 .. $DEST/spk30"
echo "  Next: set AISPEAK_ROOT=$DEST in aispeak/preprocess.sh, then run it."
