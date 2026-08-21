"""
Minimal, dependency-free LoRA (Low-Rank Adaptation) for the fairseq AV-HuBERT model.

REFERENCE COPY (not imported at runtime). The version that actually runs during
training is embedded in `av_hubert/avhubert/hubert_asr.py` (`LoRALinear` +
`_apply_lora`), because our change to upstream must be a single self-contained
patch to one file inside the pinned submodule — see our_code_changes/. This file
is the clean, standalone, documented version kept for study and the report.

Why hand-rolled: HuggingFace `peft` drags in torch 2.x, which breaks the pinned
torch 1.13 / 2021-fairseq environment. This is ~1 file, no new deps, and fully
transparent — good for an academic report where you must explain what you did.

LoRA idea (Hu et al., 2021, arXiv:2106.09685):
  A pretrained linear layer computes  y = W x  (W frozen, d_out x d_in).
  LoRA adds a low-rank update:         y = W x + (alpha / r) * B (A x)
  where A is (r x d_in), B is (d_out x r), r << min(d_in, d_out).
  Only A and B train; W stays frozen. B is zero-initialised so training starts
  exactly at the pretrained function (the adapter is a no-op at step 0).

Trainable params per adapted layer: r*(d_in + d_out) instead of d_in*d_out.
For a 768x768 attention proj at r=8: 12,288 vs 589,824 params (~48x fewer).

Usage:
    from lora import inject_lora, mark_only_lora_trainable, lora_state_dict
    n = inject_lora(model, target_names=("q_proj","v_proj"), r=8, alpha=16)
    mark_only_lora_trainable(model)     # freeze everything except LoRA (+ optional bias)
    # ... train ...
    torch.save(lora_state_dict(model), "lora_adapter.pt")   # tiny file
"""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear and adds a trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False            # freeze the pretrained weight (+bias)

        self.r = r
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        d_in, d_out = base.in_features, base.out_features
        # A: down-projection (r x d_in); B: up-projection (d_out x r)
        self.lora_A = nn.Parameter(torch.zeros(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # A ~ small random
        nn.init.zeros_(self.lora_B)                            # B = 0 -> adapter starts as no-op

    def forward(self, x):
        out = self.base(x)                                     # frozen path
        update = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + self.scaling * update


def inject_lora(model: nn.Module, target_names=("q_proj", "v_proj"),
                r: int = 8, alpha: int = 16, dropout: float = 0.0,
                name_filter: str = None) -> int:
    """
    Replace every nn.Linear whose LEAF name is in `target_names` with a LoRALinear.

    target_names: attention/FFN leaf names. For AV-HuBERT they are among:
        q_proj, k_proj, v_proj, out_proj, fc1, fc2   (both encoder and decoder use these)
    name_filter: if given, only adapt modules whose full path contains this substring
        (e.g. "decoder" to do decoder-only first, then "encoder" for the second pass).
    Returns the number of layers adapted.
    """
    to_replace = []
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = full_name.split(".")[-1]
        if leaf not in target_names:
            continue
        if name_filter is not None and name_filter not in full_name:
            continue
        to_replace.append(full_name)

    for full_name in to_replace:
        parent = model
        parts = full_name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        base = getattr(parent, parts[-1])
        setattr(parent, parts[-1], LoRALinear(base, r=r, alpha=alpha, dropout=dropout))

    return len(to_replace)


def mark_only_lora_trainable(model: nn.Module, train_bias: bool = False) -> None:
    """Freeze all params except LoRA A/B (and optionally biases)."""
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
        elif train_bias and name.endswith("bias"):
            p.requires_grad = True
        else:
            p.requires_grad = False


def lora_state_dict(model: nn.Module) -> dict:
    """Return only the LoRA params — a tiny checkpoint (~MBs, not GBs)."""
    return {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def count_trainable(model: nn.Module):
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return trn, tot, 100.0 * trn / tot
