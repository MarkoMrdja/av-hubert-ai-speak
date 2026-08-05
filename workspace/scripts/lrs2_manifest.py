#!/usr/bin/env python3
"""
Final preprocessing step: build the fairseq data directory AV-HuBERT trains on.

Mirrors the repo's avhubert/preparation/lrs3_manifest.py, but:
  * uses LRS2 splits (from split.list produced by lrs2_prepare.py),
  * lets you cap the training set to N hours (for a small-GPU subset run),
  * points at mouth-ROI videos + extracted wavs produced by the crop/audio steps.

Produces, in --out:
    train.tsv  valid.tsv  test.tsv    # manifests
    train.wrd  valid.wrd  test.wrd    # transcripts (lowercased)
    dict.wrd.txt                      # fairseq dictionary (from SentencePiece)
    spm_unigram{V}.model / .txt       # the tokenizer you pass as tokenizer_bpe_model

.tsv format (one header line "/", then per utterance, tab-separated):
    <id>  <abs_path_video.mp4>  <abs_path_audio.wav>  <n_video_frames>  <n_audio_frames>

Run lrs2_prepare.py, the mouth-crop step, the audio-extract step, and
count_frames first so file.list / label.list / nframes.* exist.

Usage:
    python lrs2_manifest.py \
        --prepared /path/to/lrs2/prepared \
        --video-dir /path/to/lrs2/video \
        --audio-dir /path/to/lrs2/audio \
        --out /path/to/lrs2/30h_data \
        --vocab-size 1000 \
        --subset-hours 10          # optional cap on the train split
"""
import argparse
import os
import shutil
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

# Reuse the repo's SentencePiece helper so the dictionary format matches exactly.
REPO_PREP = Path(__file__).resolve().parents[2] / "av_hubert" / "avhubert" / "preparation"
sys.path.insert(0, str(REPO_PREP))
from gen_subword import gen_vocab  # noqa: E402


def load_lines(p):
    with open(p) as f:
        return [ln.rstrip("\n") for ln in f]


def main():
    ap = argparse.ArgumentParser(description="LRS2 fairseq manifest builder")
    ap.add_argument("--prepared", required=True, help="dir with file.list/label.list/split.list")
    ap.add_argument("--video-dir", required=True, help="dir with mouth-ROI <id>.mp4")
    ap.add_argument("--audio-dir", required=True, help="dir with <id>.wav (16kHz)")
    ap.add_argument("--out", required=True, help="output data dir")
    ap.add_argument("--vocab-size", type=int, default=1000)
    ap.add_argument("--subset-hours", type=float, default=None,
                    help="cap TRAIN split to about this many hours (video @25fps)")
    ap.add_argument("--fps", type=float, default=25.0, help="video frame rate")
    args = ap.parse_args()

    prep = Path(args.prepared)
    fids = load_lines(prep / "file.list")
    labels = [x.lower() for x in load_lines(prep / "label.list")]
    splits = load_lines(prep / "split.list")
    nf_video = load_lines(Path(args.prepared) / "nframes.video")
    nf_audio = load_lines(Path(args.prepared) / "nframes.audio")
    assert len(fids) == len(labels) == len(splits) == len(nf_video) == len(nf_audio), \
        "file.list / label.list / split.list / nframes.* are not the same length"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Train a SentencePiece tokenizer on the (lowercased) transcripts.
    vocab_dir = out
    spm_prefix = f"spm_unigram{args.vocab_size}"
    with NamedTemporaryFile(mode="w", delete=False) as f:
        for t in labels:
            f.write(t + "\n")
        corpus = f.name
    gen_vocab(Path(corpus), vocab_dir / spm_prefix, "unigram", args.vocab_size)
    os.unlink(corpus)
    vocab_txt = (vocab_dir / spm_prefix).as_posix() + ".txt"

    # 2) Group by LRS2 split; map val->valid to match fairseq subset names.
    split_map = {"train": "train", "val": "valid", "test": "test", "pretrain": "train"}
    buckets = {"train": [], "valid": [], "test": []}
    for fid, lab, sp, nfv, nfa in zip(fids, labels, splits, nf_video, nf_audio):
        buckets[split_map[sp]].append((fid, lab, int(nfv), int(nfa)))

    # 3) Optional subset cap on the train split (keep whole utterances).
    if args.subset_hours is not None:
        budget = int(args.subset_hours * 3600 * args.fps)  # in video frames
        kept, used = [], 0
        for row in buckets["train"]:
            if used >= budget:
                break
            kept.append(row)
            used += row[2]
        print(f"train subset: {len(kept)}/{len(buckets['train'])} utts "
              f"~= {used / args.fps / 3600:.2f}h (cap {args.subset_hours}h)")
        buckets["train"] = kept

    # 4) Write tsv + wrd for each split.
    vdir = os.path.abspath(args.video_dir)
    adir = os.path.abspath(args.audio_dir)
    for name, rows in buckets.items():
        with open(out / f"{name}.tsv", "w") as fo:
            fo.write("/\n")
            for fid, _, nfv, nfa in rows:
                fo.write("\t".join([
                    fid, f"{vdir}/{fid}.mp4", f"{adir}/{fid}.wav", str(nfv), str(nfa)
                ]) + "\n")
        with open(out / f"{name}.wrd", "w") as fo:
            for _, lab, _, _ in rows:
                fo.write(lab + "\n")
        print(f"{name}: {len(rows)} utterances")

    shutil.copyfile(vocab_txt, out / "dict.wrd.txt")
    print(f"\ndone. tokenizer_bpe_model = {(vocab_dir / spm_prefix).as_posix()}.model")
    print(f"data dir ready at {out}")


if __name__ == "__main__":
    main()
