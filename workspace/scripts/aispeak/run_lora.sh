#!/usr/bin/env bash
# LoRA fine-tune AV-HuBERT on AI-SPEAK (Serbian). Per professor's plan: run
# decoder-scope first, then encdec (encoder+decoder) as an ablation.
#
# Starts from the pretrained BASE checkpoint (English LRS3+VoxCeleb2). The visual/
# acoustic encoder transfers cross-lingually; LoRA adapts it (and the decoder) to
# Serbian with very few trainable params, resisting overfitting on the small corpus.
#
# Usage:
#   bash run_aispeak_lora.sh decoder    # LoRA on decoder only  (do this first)
#   bash run_aispeak_lora.sh encdec     # LoRA on encoder+decoder (ablation)
set -euo pipefail

VARIANT="${1:-decoder}"
case "$VARIANT" in
  decoder) CONFIG=aispeak_lora_decoder ;;
  encdec)  CONFIG=aispeak_lora_encdec ;;
  *) echo "usage: run_aispeak_lora.sh [decoder|encdec]"; exit 1 ;;
esac

# ============================ CONFIG (edit me) =============================
PROJECT=${PROJECT:-/workspace/av-hubert-ai-speak}
DATA=${DATA:-/path/to/ai-speak/ser_data}          # from preprocess_aispeak.sh
CKPT=${CKPT:-$PROJECT/workspace/checkpoints/base_vox_iter5.pt}
EXP=${EXP:-$PROJECT/workspace/experiments/aispeak_lora_$VARIANT}
# ===========================================================================

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AVHUBERT="$HERE/../../../av_hubert/avhubert"
CONF_DIR="$HERE/../../configs"
TOKENIZER="$DATA/$(ls "$DATA" | grep -m1 '^spm_unigram.*\.model$')"

echo "config: $CONFIG   data: $DATA"
mkdir -p "$EXP"
cd "$AVHUBERT"

fairseq-hydra-train --config-dir "$CONF_DIR" --config-name "$CONFIG" \
  task.data="$DATA" task.label_dir="$DATA" \
  task.tokenizer_bpe_model="$TOKENIZER" \
  model.w2v_path="$CKPT" \
  common.user_dir="$(pwd)" \
  hydra.run.dir="$EXP" 2>&1 | tee "$EXP/train.log"

echo ""
echo "done. checkpoints + logs in: $EXP"
echo "then evaluate with common/evaluate.sh (point DATA/FT_CKPT at this run): gen_subset=test on $DATA"
