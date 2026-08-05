# Training on a rented RunPod GPU — first-timer guide

We train AV-HuBERT on a rented 24GB GPU (~$2–5 total) instead of the 4GB laptop.
AI-SPEAK still runs on NTP-122 per the project spec.

## The rental cycle (mental model)
Rent a remote Linux+GPU box → SSH in → build the env → upload data → train →
download results → **STOP the pod** (billing stops only when stopped/terminated).

## Docker vs. bare metal? — you get Docker, and that's fine.
RunPod **always runs your pod inside a container** (that's how the platform works —
you don't choose bare metal). You pick a **template** (a Docker image) at launch;
you never write a Dockerfile. Inside the container you have a normal Linux shell and
`sudo`/root, so from your point of view it behaves like a plain machine. We install
our conda env *inside* that container. So: **no Docker work for you** — just pick a
CUDA 11.x-compatible PyTorch template and run our setup script.

## Step-by-step

### 1. Create the pod
- runpod.io → **Deploy** → **GPU Pod**.
- GPU: **RTX 3090 or RTX 4090 (24 GB)**. Community Cloud is cheapest.
- Template: any **PyTorch / CUDA 11.8** image (e.g. a `runpod/pytorch:*cuda11.8*`).
  The host GPU driver just needs to be ≥ 11.7 — it always is on these.
- Disk: ~30–40 GB container/volume is plenty for the subset + checkpoint.
- Add your SSH public key in RunPod **Settings → SSH Keys** first, so you can `ssh` in.

### 2. Connect
The pod's **Connect** page shows a command like:
```
ssh root@213.181.xxx.xx -p 40000 -i ~/.ssh/id_ed25519
```
Run that from your Mac terminal.

### 3. Get the repo on the pod
```bash
cd /workspace
git clone --recursive https://github.com/facebookresearch/av_hubert.git \
  av-hubert-ai-speak/av_hubert
cd av-hubert-ai-speak/av_hubert && git checkout 258fb50e \
  && git submodule update --init --recursive && cd /workspace
```

### 4. Upload data + checkpoint + our scripts (from the MAC, new terminal)
```bash
bash workspace/scripts/sync_to_runpod.sh up root@213.181.xxx.xx -p 40000
```
This copies `data/`, `workspace/checkpoints/`, `workspace/configs`, `workspace/scripts`.
(Preprocess LOCALLY on the Mac first so you upload only the small subset — cheaper
than paying GPU rent during CPU-bound preprocessing.)

### 5. Build the environment (on the POD)
```bash
bash /workspace/av-hubert-ai-speak/workspace/scripts/setup_runpod.sh \
     /workspace/av-hubert-ai-speak
```
It ends by printing `CUDA available: True` + the GPU name if all is well.

### 6. Train (on the POD)
```bash
source $(conda info --base)/etc/profile.d/conda.sh && conda activate avhubert
cd /workspace/av-hubert-ai-speak
bash workspace/scripts/run_finetune_lrs2.sh          # uses the runpod config
```
Tip: run inside `tmux` so training survives an SSH disconnect:
```bash
tmux new -s train      # then run the command; detach with Ctrl-b d; reattach: tmux a -t train
```

### 7. Evaluate (on the POD)
```bash
bash workspace/scripts/evaluate_lrs2.sh              # WER in .../decode/decode.log
```

### 8. Download results (from the MAC)
```bash
bash workspace/scripts/sync_to_runpod.sh down root@213.181.xxx.xx -p 40000
```

### 9. STOP the pod
RunPod dashboard → **Stop** (keeps the disk, small storage fee) or **Terminate**
(deletes everything, no further charge). Do this every time you finish — an idle
running pod still bills by the hour.

## Cost expectations
- 24GB GPU ≈ $0.35–0.70/hr.
- One reduced LRS2 fine-tune (`max_update=18000`) ≈ 3–6 hours ≈ **$1.50–$4**.
- Keep total under ~$10 even with re-runs. Stop the pod when idle.

## Which config runs?
`run_finetune_lrs2.sh` defaults to `lrs2_base_vsr_runpod.yaml` (24GB: max_tokens
2000, update_freq 4, max_update 18000). If you OOM, drop `max_tokens` to 1000.
The old 4GB laptop config is still available: `run_finetune_lrs2.sh laptop`.
