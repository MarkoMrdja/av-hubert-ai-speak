#!/usr/bin/env python3
"""
AI-SPEAK — build the fairseq data dir (tsv/wrd/dict/spm) from the prepared lists.

train/valid/test come from the speaker-disjoint split.list produced by prepare.py.
Builds a new Serbian SentencePiece vocabulary from the training transcripts.

Prereqs: run prepare.py, then count_frames on the prepared file.list.
Outputs in --out: {train,valid,test}.{tsv,wrd}, dict.wrd.txt, spm_unigram{V}.model/.txt

Usage:
  python build_manifest.py --prepared /path/prepared --out /path/ser_data --vocab-size 500
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

# build_manifest.py is at workspace/scripts/aispeak/ -> repo root is parents[3]
REPO_PREP = Path(__file__).resolve().parents[3] / "av_hubert" / "avhubert" / "preparation"
sys.path.insert(0, str(REPO_PREP))
from gen_subword import gen_vocab  # noqa: E402


def load(p):
    return [ln.rstrip("\n") for ln in open(p, encoding="utf-8")]


# strip everything that isn't a letter or space (punctuation + curly quotes are
# not visible on the lips, so we don't ask the model to predict them)
_PUNCT = re.compile(r"[^\wČĆĐŠŽčćđšž ]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text):
    """Lowercase + strip punctuation/quotes + collapse whitespace.

    LOWERCASE is REQUIRED: AV-HuBERT's s2s label processor calls `.lower()` on every
    transcript before SPM-encoding it (hubert_pretraining.py: bpe_tokenizer.encode(
    label.lower())). If the SPM vocab is built on UPPERCASE text, the lowercased eval
    refs tokenize to all-<unk> and WER is meaningless. So the vocab MUST be lowercase.
    (Case/punctuation aren't visible on the lips anyway.)"""
    text = text.replace("“", "").replace("”", "").replace("„", "").replace("\"", "")
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True, help="dir with file/label/split.list + nframes.*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=500,
                    help="SentencePiece vocab size; keep small for a small Serbian set")
    ap.add_argument("--no-normalize", action="store_true",
                    help="keep transcripts verbatim (default: uppercase + strip punctuation)")
    args = ap.parse_args()

    prep = Path(args.prepared)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fids = load(prep / "file.list")
    labels = load(prep / "label.list")
    if not args.no_normalize:
        labels = [normalize(x) for x in labels]
    splits = load(prep / "split.list")
    nfv = load(prep / "nframes.video")
    nfa = load(prep / "nframes.audio")
    assert len(fids) == len(labels) == len(splits) == len(nfv) == len(nfa), \
        "file/label/split.list and nframes.* length mismatch"

    buckets = {"train": [], "valid": [], "test": []}
    for fid, lab, sp, v, a in zip(fids, labels, splits, nfv, nfa):
        buckets[sp].append((fid, lab, int(v), int(a)))

    # Serbian SentencePiece from TRAIN transcripts (character_coverage handles ćčšžđ).
    spm_prefix = f"spm_unigram{args.vocab_size}"
    with NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
        for _, lab, _, _ in buckets["train"]:
            f.write(lab + "\n")
        corpus = f.name
    gen_vocab(Path(corpus), out / spm_prefix, "unigram", args.vocab_size)
    os.unlink(corpus)
    vocab_txt = (out / spm_prefix).as_posix() + ".txt"

    # tsv root is a PLACEHOLDER + relative paths, so the data dir is portable across
    # machines (local -> remote GPU). Substitute the root on the target: e.g.
    #   sed -i "1s|PREPARED_ROOT|/workspace/av-hubert-ai-speak/data/aispeak_prepared|" *.tsv
    for name, rows in buckets.items():
        with open(out / f"{name}.tsv", "w", encoding="utf-8") as fo:
            fo.write("PREPARED_ROOT\n")
            for fid, _, v, a in rows:
                fo.write("\t".join([fid, f"video/{fid}.mp4", f"audio/{fid}.wav",
                                    str(v), str(a)]) + "\n")
        with open(out / f"{name}.wrd", "w", encoding="utf-8") as fo:
            for _, lab, _, _ in rows:
                fo.write(lab + "\n")
        print(f"{name}: {len(rows)} utterances")

    shutil.copyfile(vocab_txt, out / "dict.wrd.txt")
    print(f"\ndone. tokenizer_bpe_model = {(out / spm_prefix).as_posix()}.model")
    print(f"data dir: {out}")


if __name__ == "__main__":
    main()
