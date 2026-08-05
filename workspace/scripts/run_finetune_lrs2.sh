#!/usr/bin/env bash
# Launch VSR fine-tuning of a pretrained AV-HuBERT Base model on the LRS2 subset.
# Run on the machine WITH the NVIDIA GPU (the rented RunPod box, not the Mac).
#
# Prereqs: preprocess_lrs2.sh has produced $DATA, and download_checkpoint.sh has
# fetched the Base checkpoint.
#
# CONFIG variant is chosen with the second arg (default: runpod):
#   bash run_finetune_lrs2.sh            # -> lrs2_base_vsr_runpod  (24GB rented GPU)
#   bash run_finetune_lrs2.sh laptop     # -> lrs2_base_vsr_1gpu    (4GB RTX 3050)
#
# Edit the CONFIG block paths for the machine you're on.
set -euo pipefail

# ============================ CONFIG (edit me) =============================
# On RunPod the repo is typically cloned under /workspace; adjust to your paths.
PROJECT=${PROJECT:-/workspace/av-hubert-ai-speak}
DATA=${DATA:-$PROJECT/data/lrs2/10h_data}                       # from preprocess
CKPT=${CKPT:-$PROJECT/workspace/checkpoints/base_vox_iter5.pt}
EXP=${EXP:-$PROJECT/workspace/experiments/lrs2_base_vsr}
# ===========================================================================

VARIANT="${1:-runpod}"
case "$VARIANT" in
  runpod) CONFIG_NAME=lrs2_base_vsr_runpod ;;   # full fine-tune, 24GB GPU
  laptop) CONFIG_NAME=lrs2_base_vsr_1gpu ;;     # full fine-tune, 4GB fallback
  lora)   CONFIG_NAME=lrs3_base_vsr_lora ;;     # LoRA (decoder-scope) on 24GB GPU
  *)      echo "unknown variant '$VARIANT' (use: runpod | laptop | lora)"; exit 1 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AVHUBERT="$HERE/../../av_hubert/avhubert"
CONF_DIR="$HERE/../configs"
TOKENIZER="$DATA/$(ls "$DATA" | grep -m1 '^spm_unigram.*\.model$')"

echo "config: $CONFIG_NAME   data: $DATA"
mkdir -p "$EXP"
cd "$AVHUBERT"    # common.user_dir=`pwd` must point at the avhubert module

fairseq-hydra-train \
  --config-dir "$CONF_DIR" --config-name "$CONFIG_NAME" \
  task.data="$DATA" task.label_dir="$DATA" \
  task.tokenizer_bpe_model="$TOKENIZER" \
  model.w2v_path="$CKPT" \
  hydra.run.dir="$EXP" \
  common.user_dir="$(pwd)"

echo ""
echo "training finished. checkpoints + logs in: $EXP"
echo "tensorboard --logdir $EXP/tblog"
