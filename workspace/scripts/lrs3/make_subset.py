#!/usr/bin/env python3
"""
Build a small trainval subset file.list / label.list for LRS3.

Walks data/lrs3_raw/trainval/<speaker>/<clip>.mp4 + .txt, takes the first
N_TRAIN + N_VALID clips (deterministic ordering), reads each transcript's
"Text:" line, and writes:
    file.list    ids like "trainval/<speaker>/<clip>"
    label.list   transcript per id
    split.list   "train" or "valid" per id (valid = last N_VALID)
"""
import argparse
import glob
import os
from pathlib import Path


def read_text(txt):
    with open(txt, errors="ignore") as f:
        for ln in f:
            if ln.startswith("Text:"):
                return ln.split(":", 1)[1].strip()
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-valid", type=int, default=40)
    args = ap.parse_args()

    lrs3 = Path(args.lrs3)
    mp4s = sorted(glob.glob(str(lrs3 / "trainval" / "*" / "*.mp4")))
    total = args.n_train + args.n_valid
    mp4s = mp4s[:total]
    assert len(mp4s) >= total, f"only {len(mp4s)} clips available, need {total}"

    fids, labels, splits = [], [], []
    for i, mp4 in enumerate(mp4s):
        rel = os.path.relpath(mp4, lrs3)[:-4]           # trainval/<spk>/<clip>
        txt = mp4[:-4] + ".txt"
        text = read_text(txt)
        if not text:
            continue
        fids.append(rel)
        labels.append(text)
        splits.append("valid" if i >= args.n_train else "train")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "file.list").write_text("\n".join(fids) + "\n")
    (out / "label.list").write_text("\n".join(labels) + "\n")
    (out / "split.list").write_text("\n".join(splits) + "\n")
    print(f"subset: {splits.count('train')} train + {splits.count('valid')} valid "
          f"= {len(fids)} clips -> {out}")


if __name__ == "__main__":
    main()
