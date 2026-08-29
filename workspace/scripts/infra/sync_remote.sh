#!/usr/bin/env bash
# Copy project files to / from a remote GPU machine over SSH (rsync).
# Works with any SSH-reachable box (cloud GPU, cluster node, etc.), e.g.
#   ssh user@HOST -p PORT -i ~/.ssh/id_ed25519
#
# Usage:
#   bash sync_remote.sh up     user@HOST -p PORT   # local -> remote
#   bash sync_remote.sh down   user@HOST -p PORT   # remote -> local (results)
#
# What goes UP: data/, workspace/checkpoints/, workspace/configs, workspace/scripts.
# We do NOT rsync the multi-GB raw dataset by default if it's huge — see note below.
# What comes DOWN: workspace/experiments/ (checkpoints, logs, decode results).
set -euo pipefail

DIR="${1:?use: up|down}"; shift
SSH_TARGET="${1:?ssh target e.g. root@HOST}"; shift
SSH_OPTS="$*"                      # e.g. "-p 40000 -i ~/.ssh/id_ed25519"
REMOTE=/workspace/av-hubert-ai-speak
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

RSYNC="rsync -avhP --stats -e ssh${SSH_OPTS:+ $SSH_OPTS}"

case "$DIR" in
  up)
    echo "== uploading data, checkpoints, workspace to $SSH_TARGET:$REMOTE =="
    ssh $SSH_OPTS "$SSH_TARGET" "mkdir -p $REMOTE"
    # code+configs+scripts (small)
    $RSYNC "$LOCAL/workspace/configs"     "$SSH_TARGET:$REMOTE/workspace/"
    $RSYNC "$LOCAL/workspace/scripts"     "$SSH_TARGET:$REMOTE/workspace/"
    # pretrained checkpoint + dlib models (~1.4GB)
    $RSYNC "$LOCAL/workspace/checkpoints" "$SSH_TARGET:$REMOTE/workspace/"
    # preprocessed data dir (the small subset you'll train on)
    $RSYNC "$LOCAL/data"                  "$SSH_TARGET:$REMOTE/"
    echo "done. Now SSH in and run infra/setup_env.sh + common/run_finetune.sh."
    ;;
  down)
    echo "== downloading experiment results from $SSH_TARGET:$REMOTE =="
    mkdir -p "$LOCAL/workspace/experiments"
    $RSYNC "$SSH_TARGET:$REMOTE/workspace/experiments/" "$LOCAL/workspace/experiments/"
    echo "done. results in $LOCAL/workspace/experiments/"
    ;;
  *) echo "unknown direction '$DIR' (use: up | down)"; exit 1 ;;
esac

# NOTE on preprocessing location:
#  * Cheapest/fastest: preprocess LOCALLY on the Mac (CPU work — landmarks/crop),
#    then upload only the small preprocessed subset + checkpoint. Rent GPU ONLY for
#    training. This minimizes paid GPU hours.
#  * Alternative: upload raw data and run the preprocess step on the pod too.
#    Simpler but you pay GPU rent during CPU-bound preprocessing. Prefer the first.
