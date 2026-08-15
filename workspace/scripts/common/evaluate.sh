#!/usr/bin/env bash
# Decode a test set with a fine-tuned AV-HuBERT checkpoint and report WER.
# Wraps avhubert/infer_s2s.py (seq2seq beam-search decoding). Used for both the
# LRS3 subset runs and the AI-SPEAK (Serbian) LoRA runs — just point CONFIG at the
# matching data dir + experiment.
#
# EDIT the CONFIG block, then: bash evaluate.sh
set -euo pipefail

# ============================ CONFIG (edit me) =============================
PROJECT=${PROJECT:-/workspace/av-hubert-ai-speak}
DATA=${DATA:-$PROJECT/data/lrs3_raw/subset_data}
FT_CKPT=${FT_CKPT:-$PROJECT/workspace/experiments/lrs3_lora/checkpoints/checkpoint_best.pt}
RESULTS=${RESULTS:-$PROJECT/workspace/experiments/lrs3_lora/decode}
GEN_SUBSET=${GEN_SUBSET:-test}      # test | valid
MODALITIES=${MODALITIES:-video}     # video (VSR) | audio,video (AVSR)
# ===========================================================================

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AVHUBERT="$HERE/../../../av_hubert/avhubert"
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
