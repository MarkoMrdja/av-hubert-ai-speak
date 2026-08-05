#!/usr/bin/env bash
# Decode the LRS2 test set with a fine-tuned AV-HuBERT checkpoint and report WER.
# Wraps avhubert/infer_s2s.py (seq2seq beam-search decoding).
#
# EDIT the CONFIG block, then: bash evaluate_lrs2.sh
set -euo pipefail

# ============================ CONFIG (edit me) =============================
DATA=/Users/markomrdja/Repos/av-hubert-ai-speak/data/lrs2/10h_data
FT_CKPT=/Users/markomrdja/Repos/av-hubert-ai-speak/workspace/experiments/lrs2_base_vsr/checkpoints/checkpoint_best.pt
RESULTS=/Users/markomrdja/Repos/av-hubert-ai-speak/workspace/experiments/lrs2_base_vsr/decode
GEN_SUBSET=test          # test | valid
MODALITIES=video         # video (VSR) | audio,video (AVSR)
# ===========================================================================

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AVHUBERT="$HERE/../../av_hubert/avhubert"
cd "$AVHUBERT"
mkdir -p "$RESULTS"

python -B infer_s2s.py \
  --config-dir ./conf --config-name s2s_decode \
  common.user_dir="$(pwd)" \
  dataset.gen_subset="$GEN_SUBSET" \
  override.modalities="[$MODALITIES]" \
  override.data="$DATA" override.label_dir="$DATA" \
  common_eval.path="$FT_CKPT" \
  common_eval.results_path="$RESULTS"

echo ""
echo "decode complete. WER + hypotheses in: $RESULTS"
echo "  cat $RESULTS/decode.log        # summary incl. WER"
