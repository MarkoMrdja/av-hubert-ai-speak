# Provenance of vendored `av_hubert/`

This directory is a **vendored copy** of Meta's AV-HuBERT plus its `fairseq`
dependency, committed directly into this repository (rather than as a git
submodule) so that our one modification is tracked normally by git.

## Upstream versions
- **av_hubert**: [facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert)
  at commit `258fb50e155134eec2c4b49c2ae8de267075fd18`
- **fairseq** (`av_hubert/fairseq/`): [pytorch/fairseq](https://github.com/pytorch/fairseq)
  at commit `afc77bdf4bb51453ce76f1572ef2ee6ddcda8eeb`

Both are MIT-licensed; their `LICENSE` files are retained.

## Our modification
The only change from pristine upstream is **LoRA (Low-Rank Adaptation) support**
in `avhubert/hubert_asr.py` (class `LoRALinear` + `_apply_lora` + config fields).
To see the exact diff against upstream:

```bash
git log -p -- av_hubert/avhubert/hubert_asr.py
# or diff against Meta's original file at commit 258fb50
```

No other upstream files were modified.
