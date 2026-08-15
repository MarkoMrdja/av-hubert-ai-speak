#!/usr/bin/env python3
"""
AI-SPEAK (Serbian) — Step 1: build file.list / label.list / split.list from the
per-speaker Excel metadata, and preprocess video+audio into AV-HuBERT format.

AI-SPEAK layout (per the dataset spec PDF):
  <root>/spkXX/
     <spkXX>.xlsx            # metadata: name, video_a/r/l, audio, transcript, language, common
     alignment/*.align       # word-level ms alignments (not used here)
     ser/  eng/              # per-language material, each with:
        video_a_anonymized/  # FRONTAL camera, lip-only anonymized mp4  (we use this)
        video_r_anonymized/  video_l_anonymized/  audio/

Key format facts (spec):
  * video frontal = 100 fps  -> we resample to 25 fps
  * audio         = 22.05 kHz mono WAV -> we resample to 16 kHz
  * video is ALREADY lip-only/anonymized (face pixelized, rest black) -> we do NOT
    run dlib face detection; we just resize the frontal clip to 96x96 grayscale.
  * transcripts live in the Excel `transcript` column (filter language == 'ser').

This script, for the chosen language (default 'ser'):
  1. reads each spkXX Excel, keeps rows for that language with a frontal video + audio,
  2. resamples/crops the frontal video to 96x96 @25fps grayscale mp4 -> <out>/video/<id>.mp4,
  3. resamples audio to 16 kHz mono wav                              -> <out>/audio/<id>.wav,
  4. writes file.list / label.list / split.list (+ nframes.* via count_frames later).

`<id>` = "spkXX/<name>" so ids are unique across speakers.
Speaker-disjoint split: last --n-valid-spk speakers -> valid, one --test-spk -> test,
rest -> train (so we measure generalization to unseen speakers, like LRS3).

Usage:
  python aispeak_prepare.py --root /path/to/ai-speak --out /path/to/ai-speak/prepared \
     --language ser --valid-speakers spk27 spk28 --test-speakers spk29 spk30
"""
import argparse
import glob
import os
import subprocess
from pathlib import Path

try:
    import openpyxl  # lightweight xlsx reader
except ImportError:
    openpyxl = None
import cv2
import numpy as np


def read_speaker_xlsx(xlsx_path, language):
    """Return list of (name, transcript) for rows matching `language`."""
    assert openpyxl is not None, "pip install openpyxl"
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c).strip().lower() if c is not None else "" for c in rows[0]]
    idx = {h: i for i, h in enumerate(header)}
    need = ["name", "transcript", "language", "video_a", "audio"]
    for n in need:
        assert n in idx, f"column '{n}' missing in {xlsx_path}; have {header}"
    out = []
    for r in rows[1:]:
        if r is None or r[idx["name"]] is None:
            continue
        lang = str(r[idx["language"]]).strip().lower()
        if lang != language:
            continue
        # require a frontal video and audio to be available
        va = str(r[idx["video_a"]]).strip().lower()
        au = str(r[idx["audio"]]).strip().lower()
        if va in ("", "false", "none", "0", "no"):
            continue
        if au in ("", "false", "none", "0", "no"):
            continue
        name = str(r[idx["name"]]).strip()
        text = str(r[idx["transcript"]]).strip()
        if not text:
            continue
        out.append((name, text))
    return out


def resize_video_25fps_gray(src_mp4, dst_mp4, ffmpeg, size=96, fps=25):
    """Resample frontal AI-SPEAK clip to <size>x<size> grayscale @<fps>fps mp4.
    AI-SPEAK video is already lip-only/anonymized, so no dlib cropping — just
    scale the (already lip-region) frame to size x size and drop to grayscale."""
    os.makedirs(os.path.dirname(dst_mp4), exist_ok=True)
    # ffmpeg: set fps, scale to size x size, force gray then back to yuv420p mp4
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", src_mp4,
        "-vf", f"fps={fps},scale={size}:{size},format=gray,format=yuv420p",
        "-an", "-c:v", "mpeg4", "-q:v", "3", dst_mp4,
    ]
    subprocess.call(cmd)


def resample_audio_16k(src, dst, ffmpeg, sr=16000):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.call([ffmpeg, "-y", "-loglevel", "error", "-i", src,
                     "-ar", str(sr), "-ac", "1", dst])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="AI-SPEAK root (contains spkXX folders)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="ser", choices=["ser", "eng"])
    ap.add_argument("--valid-speakers", nargs="*", default=["spk27", "spk28"])
    ap.add_argument("--test-speakers", nargs="*", default=["spk29", "spk30"])
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--limit-per-speaker", type=int, default=None,
                    help="optional cap on utterances per speaker (quick tests)")
    args = ap.parse_args()

    ffmpeg = args.ffmpeg or "ffmpeg"
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vid_out = out / "video"
    aud_out = out / "audio"

    spk_dirs = sorted([d for d in glob.glob(str(root / "spk*")) if os.path.isdir(d)])
    assert spk_dirs, f"no spkXX folders under {root}"

    fids, labels, splits = [], [], []
    valid_set = set(args.valid_speakers)
    test_set = set(args.test_speakers)

    for spk_dir in spk_dirs:
        spk = os.path.basename(spk_dir)
        xlsx = glob.glob(os.path.join(spk_dir, "*.xlsx"))
        if not xlsx:
            print(f"warn: no xlsx in {spk_dir}, skipping")
            continue
        rows = read_speaker_xlsx(xlsx[0], args.language)
        if args.limit_per_speaker:
            rows = rows[: args.limit_per_speaker]
        split = "valid" if spk in valid_set else ("test" if spk in test_set else "train")
        lang_dir = os.path.join(spk_dir, args.language)
        vsrc_dir = os.path.join(lang_dir, "video_a_anonymized")
        asrc_dir = os.path.join(lang_dir, "audio")

        for name, text in rows:
            # source files (try common extensions)
            vsrc = next((p for e in (".mp4", ".MP4") if os.path.isfile(p := os.path.join(vsrc_dir, name + e))), None)
            asrc = next((p for e in (".wav", ".WAV") if os.path.isfile(p := os.path.join(asrc_dir, name + e))), None)
            if not vsrc or not asrc:
                continue
            fid = f"{spk}/{name}"
            resize_video_25fps_gray(vsrc, str(vid_out / f"{fid}.mp4"), ffmpeg, args.size, args.fps)
            resample_audio_16k(asrc, str(aud_out / f"{fid}.wav"), ffmpeg)
            fids.append(fid)
            labels.append(text)
            splits.append(split)
        print(f"{spk} [{split}]: {sum(1 for s in splits if s==split and fids)} cumulative")

    (out / "file.list").write_text("\n".join(fids) + "\n")
    (out / "label.list").write_text("\n".join(labels) + "\n")
    (out / "split.list").write_text("\n".join(splits) + "\n")
    from collections import Counter
    print(f"\nwrote {len(fids)} utterances ({args.language}) -> {out}")
    print("per-split:", dict(Counter(splits)))


if __name__ == "__main__":
    main()
