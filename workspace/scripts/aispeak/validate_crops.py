#!/usr/bin/env python3
"""
Validate AI-SPEAK mouth-ROI crops produced by prepare.py — a two-tier funnel that
combines automated checks (gross failures) with a human contact sheet (subtle cases).

TIER 1 — automated checks over ALL clips. Flags a clip if:
  * black_frac  : too much of the ROI is near-black (crop missed the face)         [reliable]
  * offcenter   : centroid of non-dark pixels is far from frame center             [weak proxy]
  * instability : large erratic frame-to-frame ROI change (bbox flicker)           [reliable-ish]
  * short/empty : too few frames                                                    [reliable]

TIER 2 — a contact sheet PNG for human review: ~N random clips per speaker (does the
heuristic generalize?) PLUS every Tier-1-flagged clip. You eyeball ~60-80 images.

Usage:
  python validate_crops.py --prepared data/aispeak_prepared --out /tmp/aispeak_val \
      --per-speaker 2
"""
import argparse, glob, os, random
from collections import defaultdict
import cv2, numpy as np

DARK = 30          # pixel <= DARK counts as "near-black"


def clip_stats(mp4):
    """Return dict of stats for one preprocessed 96x96 clip (mid + sampled frames)."""
    cap = cv2.VideoCapture(mp4)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n == 0:
        cap.release()
        return dict(n=0, black=1.0, offcenter=1.0, instab=1.0, midframe=None)
    idxs = [int(n*f) for f in (0.2, 0.4, 0.5, 0.6, 0.8)]
    frames, prev, diffs = [], None, []
    mid = None
    for i in range(n):
        ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if i == n//2: mid = g
        if i in idxs: frames.append(g)
        if prev is not None:
            diffs.append(np.abs(g.astype(int)-prev.astype(int)).mean())
        prev = g
    cap.release()
    if not frames:
        return dict(n=n, black=1.0, offcenter=1.0, instab=1.0, midframe=mid)
    # black fraction (avg over sampled frames)
    black = np.mean([(f <= DARK).mean() for f in frames])
    # centroid of non-dark pixels vs center (0=centered, 1=corner)
    cf = frames[len(frames)//2]
    m = cf > DARK
    ys, xs = np.where(m)
    if len(xs):
        cx, cy = xs.mean()/cf.shape[1], ys.mean()/cf.shape[0]
        offcenter = ((cx-0.5)**2 + (cy-0.5)**2) ** 0.5 / 0.707
    else:
        offcenter = 1.0
    # instability: 95th-pct frame-diff (real speech is smooth; flicker spikes)
    instab = float(np.percentile(diffs, 95)) / 255.0 if diffs else 1.0
    return dict(n=n, black=float(black), offcenter=float(offcenter),
                instab=float(instab), midframe=mid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--out", default="/tmp/aispeak_val")
    ap.add_argument("--per-speaker", type=int, default=2)
    ap.add_argument("--black-thresh", type=float, default=0.55,
                    help="flag if >this fraction of ROI is near-black")
    ap.add_argument("--offcenter-thresh", type=float, default=0.5)
    ap.add_argument("--instab-thresh", type=float, default=0.35)
    ap.add_argument("--min-frames", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    vids = sorted(glob.glob(os.path.join(args.prepared, "video", "**", "*.mp4"), recursive=True))
    print(f"checking {len(vids)} clips ...")

    by_spk = defaultdict(list)
    flagged, stats = [], {}
    for i, v in enumerate(vids):
        fid = "/".join(v.split("/")[-2:])[:-4]
        spk = fid.split("/")[0]
        s = clip_stats(v)
        stats[v] = s
        by_spk[spk].append(v)
        reasons = []
        if s["n"] < args.min_frames: reasons.append(f"short({s['n']})")
        if s["black"] > args.black_thresh: reasons.append(f"black({s['black']:.2f})")
        if s["offcenter"] > args.offcenter_thresh: reasons.append(f"offcenter({s['offcenter']:.2f})")
        if s["instab"] > args.instab_thresh: reasons.append(f"unstable({s['instab']:.2f})")
        if reasons:
            flagged.append((v, fid, reasons))
        if (i+1) % 300 == 0: print(f"  {i+1}/{len(vids)}")

    # ---- report ----
    print(f"\n=== TIER 1: {len(flagged)}/{len(vids)} clips flagged ===")
    for v, fid, reasons in flagged[:60]:
        print(f"  {fid}: {', '.join(reasons)}")
    if len(flagged) > 60:
        print(f"  ... and {len(flagged)-60} more (see flagged.txt)")
    with open(os.path.join(args.out, "flagged.txt"), "w") as f:
        for v, fid, reasons in flagged:
            f.write(f"{fid}\t{','.join(reasons)}\n")

    # per-speaker flag rate (catches a whole speaker being bad)
    print("\n=== per-speaker flag rate (a high rate = that speaker's crops are systematically off) ===")
    spk_flag = defaultdict(int)
    for v, fid, _ in flagged: spk_flag[fid.split("/")[0]] += 1
    for spk in sorted(by_spk):
        tot = len(by_spk[spk]); bad = spk_flag[spk]
        mark = "  <-- CHECK" if bad/tot > 0.2 else ""
        print(f"  {spk}: {bad}/{tot} flagged ({100*bad/tot:.0f}%){mark}")

    # ---- TIER 2: contact sheet = N random per speaker + all flagged (capped) ----
    rng = random.Random(1337)
    picks = []
    for spk in sorted(by_spk):
        picks += rng.sample(by_spk[spk], min(args.per_speaker, len(by_spk[spk])))
    picks += [v for v, _, _ in flagged][:30]
    tiles = []
    for v in picks:
        g = stats[v]["midframe"]
        g = np.zeros((96,96), np.uint8) if g is None else g
        fid = "/".join(v.split("/")[-2:])[:-4]
        t = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        cv2.putText(t, fid.split("/")[0], (2,10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0,255,0), 1)
        tiles.append(t)
    cols = 10
    rows = [np.hstack(tiles[i:i+cols] + [np.zeros((96,96,3),np.uint8)]*(cols-len(tiles[i:i+cols])))
            for i in range(0, len(tiles), cols)]
    sheet = np.vstack(rows)
    sheet = cv2.resize(sheet, (sheet.shape[1]*2, sheet.shape[0]*2), interpolation=cv2.INTER_NEAREST)
    out_png = os.path.join(args.out, "contact_sheet.png")
    cv2.imwrite(out_png, sheet)
    print(f"\n=== TIER 2: contact sheet -> {out_png} ({len(picks)} tiles) ===")
    print("Review it: mouths should be centered & clear across ALL speakers.")


if __name__ == "__main__":
    main()
