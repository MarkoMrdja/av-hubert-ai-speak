#!/usr/bin/env python3
"""Verify the AV-HuBERT training environment is correctly installed and the GPU
actually runs our pinned torch. Exits non-zero (fails loudly) on any problem.

Run from the avhubert module dir so the local modules import, e.g.:
    cd av_hubert/avhubert && python .../env/verify_env.py
"""
import sys

import fairseq
import torch
# local avhubert modules (must be importable from CWD / user_dir)
import hubert, hubert_asr, hubert_criterion, hubert_pretraining, resnet, decoder  # noqa: F401

print("fairseq", fairseq.__version__)
print("torch", torch.__version__)

if not torch.cuda.is_available():
    print("FAIL: CUDA not available to torch. Host driver too old, or wrong wheel.")
    sys.exit(1)

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"GPU: {name} | compute capability sm_{cap[0]}{cap[1]}")

# Real GPU op — proves the driver actually runs our CUDA 11.7 kernels, not just
# that CUDA is 'available'. This is where an incompatible GPU/torch would crash.
try:
    x = torch.randn(1024, 1024, device="cuda")
    y = (x @ x).sum().item()
    torch.cuda.synchronize()
    print(f"GPU matmul OK (sum={y:.1f}) -> torch 1.13 genuinely works on this GPU")
except Exception as e:
    print(f"FAIL: GPU op errored: {e}")
    print("Fallback: pip install torch==2.0.1+cu117 torchaudio==0.13.1 "
          "torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117")
    sys.exit(1)

# LoRA modification present?
if not (hasattr(hubert_asr, "LoRALinear") and hasattr(hubert_asr, "_apply_lora")):
    print("FAIL: LoRA modification (LoRALinear/_apply_lora) missing from hubert_asr.")
    sys.exit(1)

print("avhubert modules import OK (incl. our LoRA change)")
print("ALL CHECKS PASSED — ready to train.")
