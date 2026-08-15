#!/usr/bin/env bash
# Apply our LoRA modification on top of the pinned upstream av_hubert submodule.
#
# av_hubert/ is a git SUBMODULE recorded at a clean upstream commit. Our change to
# avhubert/hubert_asr.py (LoRA support) is kept here as a patch so provenance stays
# clear. Run this once after `git clone --recursive` (or `git submodule update
# --init --recursive`) to make the model runnable.
#
# Idempotent: skips if the patch is already applied.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMODULE="$HERE/../av_hubert"
PATCH="$HERE/lora_hubert_asr.patch"

if git -C "$SUBMODULE" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "LoRA patch already applied — nothing to do."
  exit 0
fi

git -C "$SUBMODULE" apply --check "$PATCH"
git -C "$SUBMODULE" apply "$PATCH"
echo "Applied LoRA patch to av_hubert/avhubert/hubert_asr.py"
