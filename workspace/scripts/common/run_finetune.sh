#!/usr/bin/env bash
# Launch fine-tuning of a pretrained AV-HuBERT Base model on a prepared data dir.
# Run on the machine WITH the NVIDIA GPU (any CUDA box, not a CPU-only Mac).
#
# Prereqs: a preprocess step has produced $DATA (tsv/wrd/dict/spm), and
# download_checkpoint.sh has fetched the Base checkpoint.
#
# CONFIG variant is chosen with the first arg (default: lora):
#   bash run_finetune.sh              # -> lrs3_base_vsr_lora    (LoRA, 24GB GPU)
#   bash run_finetune.sh aispeak      # -> aispeak_lora_decoder  (Serbian LoRA)
#   bash run_finetune.sh aispeak-enc  # -> aispeak_lora_encdec   (Serbian LoRA enc+dec)
#   bash run_finetune.sh full         # -> lrs3_base_vsr_lora w/ lora disabled (full FT)*
#
# * The full fine-tune runs used the LoRA config with LoRA turned off via override;
#   see workspace/experiments/lrs3_full_ft/.hydra/overrides.yaml for the exact flags.
#
# Edit the CONFIG block paths for the machine you're on.
set -euo pipefail

# ============================ CONFIG (edit me) =============================
# Adjust these paths to wherever you cloned the repo on the GPU machine.
PROJECT=${PROJECT:-/workspace/av-hubert-ai-speak}
DATA=${DATA:-$PROJECT/data/lrs3_raw/subset_data}                # from preprocess
CKPT=${CKPT:-$PROJECT/workspace/checkpoints/base_vox_iter5.pt}
EXP=${EXP:-$PROJECT/workspace/experiments/lrs3_lora}
# ===========================================================================

VARIANT="${1:-lora}"
case "$VARIANT" in
  lora)        CONFIG_NAME=lrs3_base_vsr_lora ;;    # LoRA (decoder-scope)
  aispeak)     CONFIG_NAME=aispeak_lora_decoder ;;  # Serbian LoRA, decoder scope
  aispeak-enc) CONFIG_NAME=aispeak_lora_encdec ;;   # Serbian LoRA, encoder+decoder
  full)        CONFIG_NAME=lrs3_base_vsr_lora ;;    # full FT (disable lora via override)
  *)           echo "unknown variant '$VARIANT' (use: lora | aispeak | aispeak-enc | full)"; exit 1 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AVHUBERT="$HERE/../../../av_hubert/avhubert"
CONF_DIR="$HERE/../../configs"
TOKENIZER="$DATA/$(ls "$DATA" | grep -m1 '^spm_unigram.*\.model$')"

echo "config: $CONFIG_NAME   data: $DATA"
mkdir -p "$EXP"
cd "$AVHUBERT"    # common.user_dir=`pwd` must point at the avhubert module

# Optional Hydra overrides via env vars (LORA_R, MAX_UPDATE, PATIENCE).
# Empty -> config defaults. PATIENCE early-stops after N validations w/o improvement.
EXTRA=()
[ -n "${LORA_R:-}" ]     && EXTRA+=("model.lora_r=$LORA_R")
[ -n "${MAX_UPDATE:-}" ] && EXTRA+=("optimization.max_update=$MAX_UPDATE")
[ -n "${PATIENCE:-}" ]   && EXTRA+=("checkpoint.patience=$PATIENCE")

fairseq-hydra-train \
  --config-dir "$CONF_DIR" --config-name "$CONFIG_NAME" \
  task.data="$DATA" task.label_dir="$DATA" \
  task.tokenizer_bpe_model="$TOKENIZER" \
  model.w2v_path="$CKPT" \
  hydra.run.dir="$EXP" \
  common.user_dir="$(pwd)" \
  "${EXTRA[@]}"

echo ""
echo "training finished. checkpoints + logs in: $EXP"
echo "tensorboard --logdir $EXP/tblog"
