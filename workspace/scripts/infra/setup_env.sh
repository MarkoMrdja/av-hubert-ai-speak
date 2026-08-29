#!/usr/bin/env bash
# One-shot environment setup for AV-HuBERT on any Linux + CUDA GPU machine
# (local workstation, university cluster, or any cloud GPU). Builds the pinned
# env (torch 1.13.1 CUDA build) in a fresh conda env.
#
# Requirements: an NVIDIA GPU with driver >= 11.7 (drivers are backward-
# compatible, so newer is fine), conda (installed here if absent), and ~24GB
# VRAM recommended for training. We install our own torch into the env, so the
# host's system torch/python don't matter.
#
# Usage:
#   git clone https://github.com/<you>/av-hubert-ai-speak.git
#   bash av-hubert-ai-speak/workspace/scripts/infra/setup_env.sh av-hubert-ai-speak
# (av_hubert + fairseq are VENDORED in the repo — no submodule init, no patch.)
set -euo pipefail

PROJECT="$(cd "${1:-/workspace/av-hubert-ai-speak}" && pwd)"
REQ="$PROJECT/env/requirements.txt"
echo "== project dir: $PROJECT =="
[ -f "$PROJECT/av_hubert/fairseq/setup.py" ] || {
  echo "ERROR: $PROJECT/av_hubert/fairseq not found. Is this the full repo (with vendored av_hubert)?"
  exit 1
}
[ -f "$REQ" ] || { echo "ERROR: $REQ not found."; exit 1; }

# --- 1. Miniconda: reuse an existing install (PATH / dir) or install ---------
CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [ -d "$HOME/miniconda3" ]; then
  CONDA_BASE="$HOME/miniconda3"
elif [ -d /opt/conda ]; then
  CONDA_BASE="/opt/conda"
else
  echo "== installing miniconda =="
  curl -L -o /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
  CONDA_BASE="$HOME/miniconda3"
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

# --- 2. conda env (conda-forge only; new conda blocks defaults on ToS) -----
echo "== creating conda env 'avhubert' (python 3.9) =="
conda create -y -n avhubert --override-channels -c conda-forge python=3.9
conda activate avhubert

# --- 3. cmake for dlib (compiles) ------------------------------------------
command -v cmake >/dev/null 2>&1 || (apt-get update && apt-get install -y cmake) || \
  conda install -y -c conda-forge cmake

# --- 4. install pinned deps (CUDA torch from the PyTorch index) -------------
echo "== installing deps (env/requirements.txt, CUDA torch) =="
pip install -r "$REQ" --extra-index-url https://download.pytorch.org/whl/cu117

# --- 5. fairseq editable (vendored) ----------------------------------------
echo "== installing vendored fairseq (editable) =="
pip install --no-build-isolation -e "$PROJECT/av_hubert/fairseq"
# build Cython ext (data_utils_fast etc.) — editable install may skip it
echo "== building fairseq Cython extensions =="
( cd "$PROJECT/av_hubert/fairseq" && python setup.py build_ext --inplace )
# omegaconf/hydra: kept out of requirements.txt (legacy metadata needs pip<24.1)
pip install "pip<24.1"
pip install "omegaconf==2.0.6" "hydra-core==1.0.7"

# --- 6. verify (fails loudly if the GPU can't run our torch) ---------------
echo "== verifying =="
cd "$PROJECT/av_hubert/avhubert"
python "$PROJECT/env/verify_env.py"

echo ""
echo "Activate later with:"
echo "  source \$(conda info --base)/etc/profile.d/conda.sh && conda activate avhubert"
