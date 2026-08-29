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

The w2v_path loader (hubert_asr.py) reads state["cfg"].model to BUILD an AVHubertModel
and expects fairseq training-state metadata (best_loss, optimizer_history, ...). A
fine-tuned checkpoint's cfg.model is `av_hubert_seq2seq` (wrong — would rebuild the whole
seq2seq) and it lacks the exact metadata layout the pretrained-load path upgrades.

So we use the PRETRAINED checkpoint (seed A) as the TEMPLATE — its cfg + metadata are
proven to load via w2v_path — and only overwrite its encoder WEIGHTS with seed B's
(both are the same Base AVHubertModel architecture, so keys match). Result: a checkpoint
that loads exactly like seed A but carries seed B's VSR-tuned encoder.

USAGE (run where fairseq is importable — i.e. in the CUDA training env):
   python extract_encoder.py --in base_vox_433h.pt --template base_vox_iter5.pt \
       --out base_vox_433h_encoder.pt

Then use --out as model.w2v_path, same as seed A. VALIDATE with a short run
(it should print the same "[LoRA] adapted N layers" line and loss should drop).
"""
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="fine-tuned seq2seq .pt (source of the encoder)")
    ap.add_argument("--template", required=True, help="pretrained .pt (seed A) whose cfg+metadata we reuse")
    ap.add_argument("--out", required=True, help="encoder-only .pt (w2v_path-compatible)")
    args = ap.parse_args()

    ft = torch.load(args.inp, map_location="cpu", weights_only=False)
    tmpl = torch.load(args.template, map_location="cpu", weights_only=False)

    ft_model = ft["model"]
    tmpl_model = tmpl["model"]

    # Pull seed B's encoder weights, stripping the seq2seq `encoder.w2v_model.` prefix
    # so keys match the pretrained (template) layout.
    prefix = "encoder.w2v_model."
    enc = {}
    n_enc = n_dec = n_other = 0
    for k, v in ft_model.items():
        if k.startswith(prefix):
            enc[k[len(prefix):]] = v
            n_enc += 1
        elif k.startswith("decoder."):
            n_dec += 1
        else:
            n_other += 1
    print(f"seed B keys: encoder={n_enc}  decoder(dropped)={n_dec}  other(dropped)={n_other}")
    assert n_enc > 0, "no 'encoder.w2v_model.' keys in --in; inspect list(ft['model'].keys())[:20]"

    # Overwrite the template's model weights with seed B's encoder, key by key.
    # Report coverage so a mismatch is visible.
    matched = [k for k in tmpl_model if k in enc]
    tmpl_only = [k for k in tmpl_model if k not in enc]
    enc_only = [k for k in enc if k not in tmpl_model]
    print(f"template keys={len(tmpl_model)}  matched={len(matched)}  "
          f"template-only(kept from A)={len(tmpl_only)}  encoderB-only(ignored)={len(enc_only)}")
    if enc_only[:5]:
        print(f"  note: seed-B keys not in template (ignored): {enc_only[:5]}")
    new_model = dict(tmpl_model)
    for k in matched:
        new_model[k] = enc[k]

    # Reuse the ENTIRE template checkpoint (cfg, args, metadata) — swap only 'model'.
    out = dict(tmpl)
    out["model"] = new_model
    torch.save(out, args.out)
    print(f"wrote encoder-swapped checkpoint -> {args.out}")
    print(f"  (template A's cfg/metadata + {len(matched)} encoder tensors from seed B)")
    print("VALIDATE: use as model.w2v_path in a short run; confirm it builds + loss drops.")


if __name__ == "__main__":
    main()
