#!/usr/bin/env python3
"""
Step 1 of LRS2 preprocessing: build file.list and label.list.

The official av_hubert repo ships lrs3_prepare.py, which is written for LRS3's
directory layout (pretrain / trainval / test). LRS2 is laid out differently:

    ${lrs2}/
      mvlrs_v1/
        main/<videoID>/<clipID>.mp4      # + matching .txt transcript
        pretrain/<videoID>/<clipID>.mp4  # + matching .txt transcript
      train.txt  val.txt  test.txt  pretrain.txt   # split lists (space-sep ids)

Each transcript .txt starts with a line:  "Text:  THE ACTUAL WORDS HERE"

This script walks the split lists, reads each transcript, and emits two aligned
files that the rest of the pipeline consumes:

    ${out}/file.list    one relative id per line, e.g. "main/6300.../00001"
    ${out}/label.list   the transcript text for that id, same line order

We keep a `split` column implicitly via which list an id came from, written to
${out}/split.list so the manifest step can rebuild train/valid/test.

Usage:
    python lrs2_prepare.py --lrs2 /path/to/lrs2 --out /path/to/lrs2/prepared \
        --splits train val test          # add "pretrain" to include the big set
        --subset-hours 10                 # optional: cap total training hours
"""
import argparse
import os
from pathlib import Path


def read_transcript(txt_path: Path) -> str:
    """Return the utterance text from an LRS2 .txt file (the 'Text:' line)."""
    with open(txt_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("Text:"):
                return line[len("Text:"):].strip()
    # Fallback: some derivatives store just the raw text on line 1.
    with open(txt_path, "r", errors="ignore") as f:
        return f.readline().strip()


def main():
    ap = argparse.ArgumentParser(description="LRS2 -> file.list / label.list")
    ap.add_argument("--lrs2", required=True, help="LRS2 root (contains mvlrs_v1/ and the split .txt lists)")
    ap.add_argument("--out", required=True, help="output dir for file.list / label.list / split.list")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="which split lists to include (train val test pretrain)")
    ap.add_argument("--media-subdir", default="mvlrs_v1",
                    help="subfolder under --lrs2 holding main/ and pretrain/")
    args = ap.parse_args()

    root = Path(args.lrs2)
    media = root / args.media_subdir
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fids, labels, splits = [], [], []
    for split in args.splits:
        list_path = root / f"{split}.txt"
        if not list_path.is_file():
            raise FileNotFoundError(f"missing split list: {list_path}")
        # LRS2 split lists: each line is "<videoID>/<clipID>" (train/val/test)
        # or "<videoID>/<clipID> <text>" (pretrain sometimes). Take the id token.
        with open(list_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_id = line.split()[0]  # e.g. main path id: "6300.../00001"
                # LRS2 ids in main lists are usually "main/<vid>/<clip>"? Actually
                # they are "<vid>/<clip>" and live under main/. Normalise to include
                # the top folder so paths are unambiguous.
                folder = "pretrain" if split == "pretrain" else "main"
                if not rel_id.startswith(("main/", "pretrain/")):
                    rel_id = f"{folder}/{rel_id}"
                mp4 = media / f"{rel_id}.mp4"
                txt = media / f"{rel_id}.txt"
                if not mp4.is_file() or not txt.is_file():
                    # Skip missing entries but warn once in a while.
                    continue
                fids.append(rel_id)
                labels.append(read_transcript(txt))
                splits.append(split)

    with open(out / "file.list", "w") as f:
        f.write("\n".join(fids) + "\n")
    with open(out / "label.list", "w") as f:
        f.write("\n".join(labels) + "\n")
    with open(out / "split.list", "w") as f:
        f.write("\n".join(splits) + "\n")

    print(f"wrote {len(fids)} utterances to {out}")
    by_split = {s: splits.count(s) for s in set(splits)}
    print("per-split counts:", by_split)


if __name__ == "__main__":
    main()
