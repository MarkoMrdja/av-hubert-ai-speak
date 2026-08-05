#!/usr/bin/env bash
# End-to-end AI-SPEAK (Serbian) preprocessing driver. Prepares a fairseq data dir
# for LoRA fine-tuning of AV-HuBERT on Serbian.
#
# AI-SPEAK video is ALREADY lip-only/anonymized, so unlike LRS3 there is NO dlib
# landmark/mouth-crop step — aispeak_prepare.py just resamples the frontal clip to
# 96x96 @25fps grayscale and the audio to 16 kHz.
#
# EDIT the CONFIG block (paths, speaker splits), then: bash preprocess_aispeak.sh
set -euo pipefail

# ============================ CONFIG (edit me) =============================
AISPEAK_ROOT=/path/to/ai-speak            # contains spk01 .. spk30
LANGUAGE=ser                              # ser (Serbian) | eng
VALID_SPEAKERS="spk27 spk28"              # held-out (unseen-speaker) validation
TEST_SPEAKERS="spk29 spk30"               # held-out test
VOCAB_SIZE=500
LIMIT_PER_SPEAKER=""                      # e.g. 20 for a quick dry run; "" = all
# ===========================================================================

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PREP="$HERE/../../av_hubert/avhubert/preparation"
FFMPEG="$(command -v ffmpeg)"
PREP="$AISPEAK_ROOT/prepared_${LANGUAGE}"
OUT="$AISPEAK_ROOT/${LANGUAGE}_data"

LIMIT_ARG=""
[ -n "$LIMIT_PER_SPEAKER" ] && LIMIT_ARG="--limit-per-speaker $LIMIT_PER_SPEAKER"

echo "== Step 1: read Excel, resample video/audio, build lists =="
python "$HERE/aispeak_prepare.py" --root "$AISPEAK_ROOT" --out "$PREP" \
  --language "$LANGUAGE" \
  --valid-speakers $VALID_SPEAKERS --test-speakers $TEST_SPEAKERS \
  --ffmpeg "$FFMPEG" $LIMIT_ARG

echo "== Step 2: count frames =="
# count_frames.py expects <root>/video/<id>.mp4 + <root>/audio/<id>.wav and a manifest.
python "$REPO_PREP/count_frames.py" --root "$PREP" --manifest "$PREP/file.list" --nshard 1 --rank 0
cat "$PREP/nframes.audio.0" > "$PREP/nframes.audio"
cat "$PREP/nframes.video.0" > "$PREP/nframes.video"

echo "== Step 3: build manifests + Serbian tokenizer =="
python "$HERE/aispeak_build_manifest.py" --prepared "$PREP" --out "$OUT" --vocab-size "$VOCAB_SIZE"

echo ""
echo "DONE. Fine-tune data dir: $OUT"
echo "  data       = $OUT"
echo "  label_dir  = $OUT"
echo "  tokenizer  = $OUT/spm_unigram${VOCAB_SIZE}.model"
echo ""
echo "NOTE: verify a produced ROI looks right (lips centered) before a long run:"
echo "  ffprobe \$OUT/../${LANGUAGE}_data/../prepared_${LANGUAGE}/video/<spk>/<name>.mp4"
