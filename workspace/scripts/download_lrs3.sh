#!/usr/bin/env bash
# Download LRS3 from the verified HuggingFace mirror TheNHz/ellipsis-lrs3-raw.
#
# We fetch ONLY what we need for fine-tuning + evaluation:
#   * trainval.tar.gz   (31,982 clips, COMPLETE)  -> fine-tuning data
#   * test parquet       (1,321 utts, COMPLETE)   -> evaluation
#   * lrs3-valid.id      (Meta av_hubert valid ids) -> validation split
# We SKIP the 100 pretrain.*.tar (huge, 87% complete, and we use the pretrained
# checkpoint instead of pretraining).
#
# Prereqs:
#   pip install "huggingface_hub[cli]"     (into the avhubert env)
#   hf auth login                          (free HF account + Read token)
#   Visit https://huggingface.co/datasets/TheNHz/ellipsis-lrs3-raw and click
#   "Agree and access" once (gated: auto).
#
# Usage:  bash download_lrs3.sh [dest_dir]
set -euo pipefail

REPO="TheNHz/ellipsis-lrs3-raw"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$HERE/../../data/lrs3_raw}"
mkdir -p "$DEST"

echo "== downloading LRS3 (trainval + test + valid ids) from $REPO =="
echo "   dest: $DEST"

# New hf CLI (huggingface_hub >= 1.x). Positional: <repo> [files/patterns...].
# --include patterns keep the huge pretrain tars out.
hf download "$REPO" --repo-type dataset --local-dir "$DEST" \
  --include \
    "ainncy/trainval.tar.gz" \
    "ainncy/.manifest.tsv" \
    "test-mattymchen/data/*.parquet" \
    "test-mattymchen/README.md" \
    "verification/lrs3-valid.id.avhubert" \
    "PROVENANCE.txt" "README.md"

echo ""
echo "== extracting trainval =="
tar -xzf "$DEST/ainncy/trainval.tar.gz" -C "$DEST"    # -> $DEST/trainval/<id>/<clip>.mp4 + .txt

echo ""
echo "downloaded + extracted. Next: preprocess with the LRS3 pipeline."
echo "  trainval videos: $DEST/trainval/"
echo "  test parquet:    $DEST/test-mattymchen/data/"
echo "  valid ids:       $DEST/verification/lrs3-valid.id.avhubert"
