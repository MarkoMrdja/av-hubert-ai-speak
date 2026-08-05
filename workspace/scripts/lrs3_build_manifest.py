#!/usr/bin/env python3
"""
Build the fairseq data dir (tsv/wrd/dict/spm) for the LRS3 subset.

Combines:
  * trainval subset  -> train/valid  (from prepared_subset/{file,label,split,nframes.*})
  * parquet test     -> test         (from test.{file,label,nframes.*}.list written by
                                       lrs3_test_from_parquet.py)

Outputs in --out:
    train.tsv valid.tsv test.tsv   train.wrd valid.wrd test.wrd
    dict.wrd.txt  spm_unigram{V}.model/.txt

.tsv rows: <id>  <abs video.mp4>  <abs audio.wav>  <n_video_frames>  <n_audio_frames>
Video ROIs live at  {lrs3}/video/<id>.mp4 ; audio at {lrs3}/audio/<id>.wav.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

REPO_PREP = Path(__file__).resolve().parents[2] / "av_hubert" / "avhubert" / "preparation"
sys.path.insert(0, str(REPO_PREP))
from gen_subword import gen_vocab  # noqa: E402


def load(p):
    return [ln.rstrip("\n") for ln in open(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs3", required=True)
    ap.add_argument("--prepared", required=True, help="prepared_subset dir (trainval)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=1000)
    args = ap.parse_args()

    lrs3 = Path(args.lrs3)
    prep = Path(args.prepared)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vdir = os.path.abspath(lrs3 / "video")
    adir = os.path.abspath(lrs3 / "audio")

    # --- trainval subset -> train / valid buckets ---
    fids = load(prep / "file.list")
    labels = [x.lower() for x in load(prep / "label.list")]
    splits = load(prep / "split.list")
    nfv = load(prep / "nframes.video")
    nfa = load(prep / "nframes.audio")
    buckets = {"train": [], "valid": [], "test": []}
    for fid, lab, sp, v, a in zip(fids, labels, splits, nfv, nfa):
        buckets[sp].append((fid, lab, int(v), int(a)))

    # --- parquet test -> test bucket ---
    tfids = load(lrs3 / "test.file.list")
    tlabels = [x.lower() for x in load(lrs3 / "test.label.list")]
    tnfv = load(lrs3 / "test.nframes.video")
    tnfa = load(lrs3 / "test.nframes.audio")
    for fid, lab, v, a in zip(tfids, tlabels, tnfv, tnfa):
        buckets["test"].append((fid, lab, int(v), int(a)))

    # --- tokenizer on train+valid transcripts ---
    spm_prefix = f"spm_unigram{args.vocab_size}"
    with NamedTemporaryFile(mode="w", delete=False) as f:
        for _, lab, _, _ in buckets["train"] + buckets["valid"]:
            f.write(lab + "\n")
        corpus = f.name
    gen_vocab(Path(corpus), out / spm_prefix, "unigram", args.vocab_size)
    os.unlink(corpus)
    vocab_txt = (out / spm_prefix).as_posix() + ".txt"

    for name, rows in buckets.items():
        with open(out / f"{name}.tsv", "w") as fo:
            fo.write("/\n")
            for fid, _, v, a in rows:
                fo.write("\t".join([fid, f"{vdir}/{fid}.mp4", f"{adir}/{fid}.wav",
                                    str(v), str(a)]) + "\n")
        with open(out / f"{name}.wrd", "w") as fo:
            for _, lab, _, _ in rows:
                fo.write(lab + "\n")
        print(f"{name}: {len(rows)} utterances")

    shutil.copyfile(vocab_txt, out / "dict.wrd.txt")
    print(f"\ndone. tokenizer_bpe_model = {(out / spm_prefix).as_posix()}.model")
    print(f"data dir: {out}")


if __name__ == "__main__":
    main()
