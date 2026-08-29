#!/usr/bin/env python3
"""
AI-SPEAK (Serbian) — Step 1: build file.list / label.list / split.list from the
per-speaker Excel metadata, and preprocess video+audio into AV-HuBERT format.

AI-SPEAK layout (VERIFIED against the released Kaggle dataset, 2026-08):
  <root>/spkNN-MM/spkXX/            # NOTE the spkNN-MM grouping dir above spkXX
     <spkXX>.xlsx                   # 8 cols: name, video_a/r/l, audio, transcript, language, common
     alignment/<name>.align         # word-level timings, TAB-sep "start<TAB>end<TAB>word", 100ns ticks
     ser/  eng/                     # per-language material, each with:
        video_a_anonymized/         # FRONTAL camera, lip-only anonymized mp4  (we use this)
        video_r_anonymized/  video_l_anonymized/  audio/

Verified format facts:
  * frontal video = 1080x1920 portrait, 100 fps  -> we resample to 25 fps
  * audio         = 22.05 kHz mono WAV (PCM_S16LE) -> we resample to 16 kHz
  * ser/eng clips INTERLEAVE by index -> filter by the Excel `language` column, not index.

Two things that differ from a naive pipeline (see docs/AISPEAK_ROI_CROP_DECISION.md):
  1. ROI CROP (not naive resize): the frame is mostly black with the anonymized face
     as a bright "island" whose position varies per speaker. We find the non-black
     bounding box per frame, take a mouth-centered square (cy = y0 + 0.66*h,
     half = 0.34*w), and resize to 96x96 gray. dlib was rejected (fails on ~50% of
     speakers due to the pixelated upper face).
  2. TRIM via .align: clips are untrimmed (~1-4s of silence + occasional laughter).
     We cut video+audio to [first_nonsil_start - pad, last_nonsil_end + pad], dropping
     leading/trailing `sil`, so AI-SPEAK clips match LRS3's utterance-tight distribution.

This script, for the chosen language (default 'ser'):
  1. reads each spkXX Excel, keeps rows for that language with a frontal video + audio,
  2. content-bbox crops + resamples the frontal video to 96x96 @25fps grayscale mp4,
  3. trims + resamples audio to 16 kHz mono wav,
  4. writes file.list / label.list / split.list (+ nframes.* via count_frames later).

`<id>` = "spkXX/<name>" so ids are unique across speakers.
Speaker-disjoint split: --valid-speakers -> valid, --test-speakers -> test, rest -> train.

Usage:
  python prepare.py --root /path/to/ai-speak --out /path/to/ai-speak/prepared \
     --language ser --valid-speakers spk27 spk28 --test-speakers spk29 spk30 --jobs 8
"""
import argparse
import glob
import os
import subprocess
import tempfile
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path

try:
    import openpyxl  # lightweight xlsx reader
except ImportError:
    openpyxl = None
import cv2
import numpy as np

ALIGN_TICKS_PER_SEC = 1e7   # .align timestamps are in 100-nanosecond units
TRIM_PAD_SEC = 0.2          # small margin kept around the utterance


def find_speaker_dirs(root):
    """Return spkXX dirs, handling the spkNN-MM grouping level (and the flat case)."""
    # grouped: <root>/spk01-05/spk01/ ...
    grouped = sorted(glob.glob(os.path.join(root, "spk*-*", "spk*")))
    grouped = [d for d in grouped if os.path.isdir(d) and "-" not in os.path.basename(d)]
    if grouped:
        return grouped
    # flat fallback: <root>/spk01/ ...
    return sorted(d for d in glob.glob(os.path.join(root, "spk*")) if os.path.isdir(d))


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


def read_align_window(align_path):
    """Return (start_sec, end_sec) of the spoken utterance (dropping leading/trailing
    `sil`), padded by TRIM_PAD_SEC. Returns None if the file is missing/empty/all-sil."""
    if not align_path or not os.path.isfile(align_path):
        return None
    words = []
    for line in open(align_path):
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        a, b, w = parts
        if w == "sil":
            continue
        try:
            words.append((int(a), int(b)))
        except ValueError:
            continue
    if not words:
        return None
    start = words[0][0] / ALIGN_TICKS_PER_SEC - TRIM_PAD_SEC
    end = words[-1][1] / ALIGN_TICKS_PER_SEC + TRIM_PAD_SEC
    return max(0.0, start), end


def content_bbox_crop(gray, ydown=0.66, wfrac=0.34, thresh=15):
    """Mouth-centered square crop from an AI-SPEAK frame (mostly-black + face island).
    Returns a crop (variable size) or None if the frame is all black."""
    mask = gray > thresh
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0
    cx = (x0 + x1) // 2
    cy = int(y0 + ydown * bh)
    half = max(int(bw * wfrac), 24)
    H, W = gray.shape
    a, b = max(0, cx - half), max(0, cy - half)
    c, d = min(W, cx + half), min(H, cy + half)
    roi = gray[b:d, a:c]
    return roi if roi.size else None


def process_video(vsrc, dst_mp4, window, size=96, fps=25):
    """Read frontal clip, trim to `window` (start,end) sec, content-bbox crop each
    frame to size x size gray, write an mp4 at `fps`. Returns #frames written."""
    os.makedirs(os.path.dirname(dst_mp4), exist_ok=True)
    cap = cv2.VideoCapture(vsrc)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 100.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if window:
        f0 = int(window[0] * src_fps)
        f1 = min(n_total, int(window[1] * src_fps))
    else:
        f0, f1 = 0, n_total
    step = src_fps / fps            # keep every `step`-th source frame -> target fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst_mp4, fourcc, fps, (size, size), isColor=False)
    written = 0
    next_keep = f0
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    fi = f0
    while fi < f1:
        ok, frame = cap.read()
        if not ok:
            break
        if fi >= next_keep:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = content_bbox_crop(gray)
            if roi is not None:
                writer.write(cv2.resize(roi, (size, size)))
                written += 1
            next_keep += step
        fi += 1
    writer.release()
    cap.release()
    return written


def process_audio(asrc, dst_wav, window, ffmpeg="ffmpeg", sr=16000):
    os.makedirs(os.path.dirname(dst_wav), exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error"]
    if window:
        cmd += ["-ss", f"{window[0]:.3f}", "-to", f"{window[1]:.3f}"]
    cmd += ["-i", asrc, "-ar", str(sr), "-ac", "1", dst_wav]
    subprocess.call(cmd)


def _worker(job, vid_out, aud_out, size, fps, ffmpeg, trim=True):
    """One utterance: (fid, vsrc, asrc, align) -> (fid, nframes) or (fid, 0) on fail.
    trim=False keeps the FULL clip (no .align-based cutting)."""
    fid, vsrc, asrc, align = job
    window = read_align_window(align) if trim else None
    dst_v = str(Path(vid_out) / f"{fid}.mp4")
    dst_a = str(Path(aud_out) / f"{fid}.wav")
    try:
        n = process_video(vsrc, dst_v, window, size, fps)
        if n == 0:
            return (fid, 0)
        process_audio(asrc, dst_a, window, ffmpeg)
        return (fid, n)
    except Exception as e:
        print(f"  ! {fid}: {e}")
        return (fid, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="AI-SPEAK root (contains spkNN-MM/spkXX)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="ser", choices=["ser", "eng"])
    ap.add_argument("--valid-speakers", nargs="*", default=["spk27", "spk28"])
    ap.add_argument("--test-speakers", nargs="*", default=["spk29", "spk30"])
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--no-trim", action="store_true",
                    help="keep FULL clips (do NOT cut to the .align utterance window). "
                         "Use to test whether alignment-trimming clips the visual speech onset.")
    ap.add_argument("--limit-per-speaker", type=int, default=None,
                    help="optional cap on utterances per speaker (quick tests)")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vid_out, aud_out = out / "video", out / "audio"

    spk_dirs = find_speaker_dirs(str(root))
    assert spk_dirs, f"no spkXX folders under {root}"
    print(f"found {len(spk_dirs)} speakers; language={args.language}; jobs={args.jobs}")

    valid_set, test_set = set(args.valid_speakers), set(args.test_speakers)
    jobs, meta = [], {}   # meta[fid] = (split, text)

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
        align_dir = os.path.join(spk_dir, "alignment")
        for name, text in rows:
            vsrc = next((p for e in (".mp4", ".MP4") if os.path.isfile(p := os.path.join(vsrc_dir, name + e))), None)
            asrc = next((p for e in (".wav", ".WAV") if os.path.isfile(p := os.path.join(asrc_dir, name + e))), None)
            if not vsrc or not asrc:
                continue
            align = os.path.join(align_dir, name + ".align")
            fid = f"{spk}/{name}"
            jobs.append((fid, vsrc, asrc, align))
            meta[fid] = (split, text)

    print(f"processing {len(jobs)} utterances with {args.jobs} workers ...")
    if args.no_trim:
        print("  --no-trim: keeping FULL clips (no .align cutting)")
    work = partial(_worker, vid_out=str(vid_out), aud_out=str(aud_out),
                   size=args.size, fps=args.fps, ffmpeg=args.ffmpeg, trim=not args.no_trim)
    with Pool(args.jobs) as pool:
        results = pool.map(work, jobs)

    fids, labels, splits = [], [], []
    dropped = 0
    for fid, n in results:
        if n <= 0:
            dropped += 1
            continue
        split, text = meta[fid]
        fids.append(fid); labels.append(text); splits.append(split)

    (out / "file.list").write_text("\n".join(fids) + "\n")
    (out / "label.list").write_text("\n".join(labels) + "\n")
    (out / "split.list").write_text("\n".join(splits) + "\n")
    print(f"\nwrote {len(fids)} utterances ({args.language}) -> {out}  (dropped {dropped})")
    print("per-split:", dict(Counter(splits)))


if __name__ == "__main__":
    main()
