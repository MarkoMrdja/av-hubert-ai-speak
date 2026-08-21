#!/usr/bin/env python3
"""
Extract the pretrained ENCODER from a fine-tuned AV-HuBERT seq2seq checkpoint
(e.g. base_vox_433h.pt) into a checkpoint that loads via `model.w2v_path` exactly
like a pretrained checkpoint (e.g. base_vox_iter5.pt).

WHY: our AI-SPEAK experiment "seed B" wants to start from a checkpoint whose encoder
has ALREADY been fine-tuned for English VSR (lip-reading skill, which transfers
cross-lingually), but WITHOUT its English decoder (wrong language + wrong vocab size,
1000 vs our Serbian 500). So we keep B's encoder, drop its decoder, and let the
Serbian decoder train fresh. This isolates the real question: does a VSR-tuned
encoder beat a merely-pretrained one (seed A) for Serbian?

HOW: a fine-tuned AVHubertSeq2Seq checkpoint stores keys like
   encoder.w2v_model.<...>   (the AVHubertModel encoder — what we want)
   decoder.<...>             (the English seq2seq decoder — we drop this)
A pretrained checkpoint stores the encoder keys WITHOUT the `encoder.w2v_model.`
prefix (just `<...>`), plus a `cfg` describing the AVHubertModel. We remap the keys
and reuse the fine-tuned checkpoint's cfg.model (same encoder architecture).

USAGE (run where fairseq is importable — e.g. on RunPod):
   python extract_encoder.py --in base_vox_433h.pt --out base_vox_433h_encoder.pt

Then use --out as model.w2v_path, same as seed A. VALIDATE by loading it through the
normal fine-tune launch (it should print the same "[LoRA] adapted N layers" line).
"""
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="fine-tuned seq2seq .pt")
    ap.add_argument("--out", required=True, help="encoder-only .pt (w2v_path-compatible)")
    args = ap.parse_args()

    ckpt = torch.load(args.inp, map_location="cpu", weights_only=False)
    model = ckpt["model"]

    # Collect encoder weights. In AVHubertSeq2Seq the encoder is wrapped as
    # `encoder.w2v_model.<...>`. A pretrained checkpoint expects bare `<...>` keys.
    prefix = "encoder.w2v_model."
    enc = {}
    n_enc = n_dec = n_other = 0
    for k, v in model.items():
        if k.startswith(prefix):
            enc[k[len(prefix):]] = v
            n_enc += 1
        elif k.startswith("decoder."):
            n_dec += 1
        else:
            n_other += 1
    print(f"keys: encoder={n_enc}  decoder(dropped)={n_dec}  other(dropped)={n_other}")
    assert n_enc > 0, ("no 'encoder.w2v_model.' keys found — checkpoint layout differs; "
                       "inspect list(model.keys())[:20] and adjust the prefix")

    # Build a pretrained-style checkpoint: same cfg (encoder arch), encoder weights.
    out = {
        "model": enc,
        "cfg": ckpt.get("cfg"),
        "args": ckpt.get("args"),
    }
    # The pretrained-load path reads cfg.model as the AVHubertModel config. A finetuned
    # cfg.model is av_hubert_seq2seq; the encoder-building path (hubert_asr.py) reads
    # w2v_args.model and builds AVHubertModel from it — validate on first RunPod load.
    torch.save(out, args.out)
    print(f"wrote encoder-only checkpoint -> {args.out}")
    print("VALIDATE: use as model.w2v_path in a short run; confirm it builds + trains.")


if __name__ == "__main__":
    main()
