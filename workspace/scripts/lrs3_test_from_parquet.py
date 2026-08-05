#!/usr/bin/env python3
"""
Convert the LRS3 test split (parquet, from TheNHz/ellipsis-lrs3-raw) into the
mouth-ROI mp4 + 16kHz wav + transcript layout that AV-HuBERT expects.

The test parquet stores ALREADY-PREPROCESSED media:
  * video: uint8 array (T, 96, 96)  -> grayscale mouth ROIs (no dlib/crop needed!)
  * audio: int16 array (N,)         -> 16 kHz mono waveform
  * label: the transcript string
So we skip landmark detection / cropping for test and write final ROIs directly.

Outputs, under --out:
  video/test/<idx>.mp4     grayscale 96x96 @25fps
  audio/test/<idx>.wav     16 kHz mono
  test.file.list           one id per line: "test/<idx>"
  test.label.list          transcript (UPPER->lower handled by manifest step)
  test.nframes.video / .audio

Usage:
  python lrs3_test_from_parquet.py \
      --parquet-dir data/lrs3_raw/test-mattymchen/data \
      --out data/lrs3_raw --fps 25 --sr 16000
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import cv2
import pyarrow.parquet as pq
from scipy.io import wavfile
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out", required=True, help="LRS3 root (gets video/, audio/, test.* lists)")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    out = Path(args.out)
    vdir = out / "video" / "test"
    adir = out / "audio" / "test"
    vdir.mkdir(parents=True, exist_ok=True)
    adir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet")))
    assert files, f"no parquet in {args.parquet_dir}"

    fids, labels, nfv, nfa = [], [], [], []
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    global_idx = 0
    for pf_path in files:
        pf = pq.ParquetFile(pf_path)
        for rg in range(pf.num_row_groups):
            batch = pf.read_row_group(rg).to_pydict()
            n = len(batch["idx"])
            for i in tqdm(range(n), desc=os.path.basename(pf_path)):
                vid = np.array(batch["video"][i], dtype=np.uint8)   # (T,96,96)
                aud = np.array(batch["audio"][i], dtype=np.int16)   # (N,)
                text = batch["label"][i]
                fid = f"test/{global_idx:06d}"
                global_idx += 1

                # write grayscale mp4 (write as 3-channel; AV-HuBERT reads then grays)
                T, H, W = vid.shape
                vp = vdir / f"{global_idx-1:06d}.mp4"
                vw = cv2.VideoWriter(str(vp), fourcc, args.fps, (W, H), isColor=True)
                for t in range(T):
                    vw.write(cv2.cvtColor(vid[t], cv2.COLOR_GRAY2BGR))
                vw.release()

                # write wav
                wavfile.write(str(adir / f"{global_idx-1:06d}.wav"), args.sr, aud)

                fids.append(fid)
                labels.append(text.strip())
                nfv.append(T)
                nfa.append(len(aud))

    with open(out / "test.file.list", "w") as f:
        f.write("\n".join(fids) + "\n")
    with open(out / "test.label.list", "w") as f:
        f.write("\n".join(labels) + "\n")
    with open(out / "test.nframes.video", "w") as f:
        f.write("\n".join(map(str, nfv)) + "\n")
    with open(out / "test.nframes.audio", "w") as f:
        f.write("\n".join(map(str, nfa)) + "\n")

    print(f"\nwrote {len(fids)} test utterances")
    print(f"  video ROIs: {vdir}")
    print(f"  audio:      {adir}")


if __name__ == "__main__":
    main()
