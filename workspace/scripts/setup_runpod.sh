#!/usr/bin/env bash
# One-shot environment setup on a freshly-rented RunPod (Linux + CUDA) GPU box.
# Recreates the same pinned AV-HuBERT env we built on the Mac, but with the CUDA
# build of torch 1.13.1 instead of the CPU build.
#
# RECOMMENDED RunPod template: a PyTorch or CUDA image on the CUDA 11.x line
# (e.g. "runpod/pytorch:2.1.0-py3.10-cuda11.8..." works — we install our own torch
# into a fresh conda env, so the base image's torch/python doesn't matter much;
# what matters is the HOST CUDA DRIVER being >= 11.7). Pick a 24GB GPU (RTX 3090/4090).
#
# Usage on the rented box (SSH in, then):
#   git clone --recursive https://github.com/facebookresearch/av_hubert.git   # or your fork
#   bash setup_runpod.sh /workspace/av-hubert-ai-speak
# If you instead scp your whole project folder up, just point the arg at it.
set -euo pipefail

PROJECT="${1:-/workspace/av-hubert-ai-speak}"
echo "== project dir: $PROJECT =="
[ -d "$PROJECT/av_hubert/fairseq" ] || {
  echo "ERROR: $PROJECT/av_hubert/fairseq not found."
  echo "Clone with:  git clone --recursive https://github.com/facebookresearch/av_hubert.git $PROJECT/av_hubert"
  echo "and pin:     cd $PROJECT/av_hubert && git checkout 258fb50e && git submodule update --init --recursive"
  exit 1
}

# --- 1. Miniconda (if absent) ---------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "== installing miniconda =="
  curl -L -o /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
fi
source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"

# --- 2. conda env (conda-forge only; new conda blocks defaults on ToS) -----
echo "== creating conda env 'avhubert' (python 3.9) =="
conda create -y -n avhubert --override-channels -c conda-forge python=3.9
conda activate avhubert

# --- 3. PyTorch 1.13.1 CUDA 11.7 build -------------------------------------
# This is the CUDA counterpart of the Mac CPU build. cu117 wheels work on any
# host NVIDIA driver >= 11.7 (drivers are backward compatible).
echo "== installing torch 1.13.1 (cu117) =="
pip install "numpy==1.23.5" \
  torch==1.13.1+cu117 torchaudio==0.13.1 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117

# --- 4. Repo + prep deps (same pins as the frozen Mac env) -----------------
echo "== installing repo + preprocessing deps =="
pip install "opencv-python==4.9.0.80" sentencepiece editdistance \
  scikit-image==0.19.3 python_speech_features pydub tqdm \
  "Cython<3" "setuptools<70"
# dlib for landmark detection (compiles; cmake must be present)
command -v cmake >/dev/null 2>&1 || (apt-get update && apt-get install -y cmake) || \
  conda install -y -c conda-forge cmake
pip install dlib

# --- 5. fairseq editable + the omegaconf/hydra pins ------------------------
echo "== installing fairseq (editable) =="
cd "$PROJECT/av_hubert/fairseq"
pip install --no-build-isolation -e .
# fairseq's loose bounds pull omegaconf/hydra too old -> ImportError: II.
# Their legacy metadata needs pip<24.1.
pip install "pip<24.1"
pip install "omegaconf==2.0.6" "hydra-core==1.0.7"

# --- 6. verify (fail loudly if the GPU can't actually run our torch) --------
echo "== verifying =="
cd "$PROJECT/av_hubert/avhubert"
python - <<'EOF'
import sys
import fairseq, torch
import hubert, hubert_asr, hubert_criterion, hubert_pretraining, resnet, decoder
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
    print("Fallback: pip install torch==2.0.1+cu117 torchaudio==0.13.1 torchvision==0.15.2+cu117 "
          "--extra-index-url https://download.pytorch.org/whl/cu117")
    sys.exit(1)
print("avhubert modules import OK")
print("ALL CHECKS PASSED — ready to train.")
EOF

echo ""
echo "Activate later with:"
echo "  source \$(conda info --base)/etc/profile.d/conda.sh && conda activate avhubert"
