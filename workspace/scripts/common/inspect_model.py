#!/usr/bin/env python3
"""
Load the pretrained AV-HuBERT Base checkpoint and print its structure, so you can
match the real model to your architecture slides (task 4) — no dataset needed.

Usage (from anywhere, with the avhubert conda env active):
    python inspect_model.py [path/to/checkpoint.pt]
Defaults to workspace/checkpoints/base_vox_iter5.pt.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AVHUBERT = HERE.parent.parent / "av_hubert" / "avhubert"
sys.path.insert(0, str(AVHUBERT))  # so 'hubert', 'hubert_pretraining' import

import fairseq  # noqa: E402
import hubert, hubert_pretraining, hubert_asr  # noqa: E402,F401  (register model/task)


def human(n):
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


def main():
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "checkpoints" / "base_vox_iter5.pt"
    print(f"loading: {ckpt}\n")
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([str(ckpt)])
    model = models[0]

    print("=" * 70)
    print("TOP-LEVEL MODULES (match these to your architecture slide)")
    print("=" * 70)
    for name, child in model.named_children():
        n_params = sum(p.numel() for p in child.parameters())
        print(f"  {name:30s} {type(child).__name__:28s} {human(n_params):>8s} params")

    total = sum(p.numel() for p in model.parameters())
    print("-" * 70)
    print(f"  TOTAL PARAMETERS: {human(total)} ({total:,})   # Base ~103M, Large ~325M")

    print("\n" + "=" * 70)
    print("KEY CONFIG (encoder depth/width, modalities, label rate)")
    print("=" * 70)
    mc = cfg.model
    for k in ["encoder_layers", "encoder_embed_dim", "encoder_ffn_embed_dim",
              "encoder_attention_heads", "label_rate", "modality_dropout",
              "audio_dropout", "masking_type"]:
        if hasattr(mc, k):
            print(f"  {k:28s} = {getattr(mc, k)}")

    print("\n" + "=" * 70)
    print("FULL MODULE TREE (first ~60 lines) — find ResNet + Transformer here")
    print("=" * 70)
    for i, line in enumerate(str(model).splitlines()):
        if i > 60:
            print("  ... (truncated)")
            break
        print(line)


if __name__ == "__main__":
    main()
