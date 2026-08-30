#!/usr/bin/env python3
"""
Extract the encoder from a fine-tuned AV-HuBERT seq2seq checkpoint (e.g. base_vox_433h.pt)
into a checkpoint that loads via `model.w2v_path` exactly like a pretrained one
(base_vox_iter5.pt). Used to start Serbian fine-tuning from an English-VSR-tuned encoder
while dropping its English decoder.

A fine-tuned AVHubertSeq2Seq checkpoint stores encoder keys as `encoder.w2v_model.<...>`
and decoder keys as `decoder.<...>`; a pretrained checkpoint stores the encoder keys bare
(`<...>`) plus a `cfg` the w2v_path loader uses to rebuild the AVHubertModel. Since the
fine-tuned checkpoint's cfg is `av_hubert_seq2seq` and lacks the metadata the pretrained
load path expects, we use the pretrained checkpoint as a TEMPLATE (cfg + metadata) and
overwrite only its encoder weights with the fine-tuned encoder (same Base architecture,
matching keys).

Usage (run where fairseq is importable, i.e. in the CUDA training env):
   python extract_encoder.py --in base_vox_433h.pt --template base_vox_iter5.pt \
       --out base_vox_433h_encoder.pt
Then pass --out as model.w2v_path.
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
