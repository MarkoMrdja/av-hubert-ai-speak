# Architecture → Code Mapping (task 4)

Connects the concepts in your presentation to their concrete implementation in
this exact code version:

- **`av_hubert`** commit `258fb50e155134eec2c4b49c2ae8de267075fd18`
- **`fairseq`** submodule commit `afc77bdf4bb51453ce76f1572ef2ee6ddcda8eeb`

All paths are under `av_hubert/avhubert/` unless noted. Line numbers are
approximate (this code version) — search the symbol if they drift.

---

## 1. The big picture: one model, two heads

| Phase | Model class | Loss (criterion) | Target | You run it? |
|---|---|---|---|---|
| Pretraining | `AVHubertModel` (`hubert.py`) | `AVHubertCriterion` (`hubert_criterion.py`) | k-means **cluster IDs** (`.km`) | No (checkpoint) |
| Fine-tuning (VSR) | `AVHubertSeq2Seq` (`hubert_asr.py:412`) | `label_smoothed_cross_entropy` (fairseq) | **transcript tokens** (`.wrd`→spm) | **Yes, on LRS2** |

Fine-tuning *wraps* the pretrained `AVHubertModel` as its encoder and bolts a
`TransformerDecoder` (`decoder.py`) on top.

---

## 2. Slide-by-slide mapping

### Slide "Pregled arhitekture" — two modalities → fusion → Transformer → cluster IDs
- Whole model: **`AVHubertModel`** in `hubert.py`.
- Forward pass entry: `AVHubertModel.forward(...)`.
- Feature extraction + fusion: **`AVHubertModel.forward_features`** in `hubert.py`.

### Slide "Ekstraktori obeležja i fuzija"
- **Video: modified ResNet-18 over mouth ROI** → `resnet.py` (the `ResEncoder`
  / ResNet blocks), instantiated inside `hubert.py` as the visual `feature_extractor`.
- **Audio: single linear layer over log-mel** → in `hubert.py`, the audio path is a
  simple linear projection (the `feature_extractor_audio` / linear proj). This is
  why audio can't dominate — it has almost no capacity before fusion.
- **Fusion by channel concatenation (`f^av`)** → in `forward_features`, the video
  and audio feature sequences are concatenated along the channel/feature dim, then
  projected into the Transformer's model dim.
- **BASE (103M, 12 layers, 768) vs LARGE (325M, 24 layers, 1024)** → set by the
  encoder config (`encoder_layers`, `encoder_embed_dim`) baked into the checkpoint
  you load; the config files under `conf/pretrain/{base,large}_*` show the numbers.

### Slide "HuBERT: osnova modela" (clustering ↔ masked prediction loop)
- **k-means clustering → discrete targets `z_{1:T}`** → the `clustering/` folder
  (offline; produces `.km` label files). Not run by you.
- **Masked prediction (BERT-style)** → masking in `hubert.py` (the `apply_mask` /
  `mask_prob` logic), prediction head compares encoder outputs to learned
  **cluster-embedding** vectors.

### Slide with the objective function (THE key slide) — masked cluster prediction
- File: **`hubert_criterion.py`**, class `AVHubertCriterion` (registered as
  `"av_hubert"`).
- The loss is cross-entropy over cluster classes, split into masked and unmasked:
  ```python
  # hubert_criterion.py (forward)
  loss_m = F.cross_entropy(logp_m, targ_m)   # over MASKED frames
  loss_u = F.cross_entropy(logp_u, targ_u)   # over UNMASKED frames
  loss = pred_masked_weight * Σ loss_m  +  pred_nomask_weight * Σ loss_u
  ```
- Config knobs matching the paper's weighting: `pred_masked_weight` (α on masked),
  `pred_nomask_weight` (usually 0 → predict only masked frames), lines ~21–28.
- `logp_m` / `targ_m` come from the model: `logit_m_list`, `target_m_list` in
  `AVHubertModel`'s output dict. `targ_m` = the cluster ID of each masked frame.

### Modality dropout (the trick that makes the visual encoder strong)
- In `hubert.py` (the `modality_dropout` / `audio_dropout` fields and where inputs
  are zeroed in `forward_features`) and in `hubert_dataset.py` (input assembly).

### Noise augmentation (Robust paper / AVSR)
- `hubert_dataset.py` — noise mixing into the audio stream.
- Prepared by `preparation/musan_prepare.py` + `noise_manifest.py`.

### Fine-tuning: seq2seq decoder + CE (what you train on LRS2)
- `AVHubertSeq2Seq` (`hubert_asr.py:412`): builds `HubertEncoderWrapper` around the
  pretrained model + `TransformerDecoder` (`hubert_asr.py:487-488`).
- `AVHubertCtc` (`hubert_asr.py:153`): the alternative CTC head (you won't use it).
- Decoder itself: `decoder.py` (`TransformerDecoder`), 6 layers by default.
- Loss: `label_smoothed_cross_entropy` (fairseq built-in), set in the finetune
  config's `criterion._name`.
- Encoder freezing: `freeze_finetune_updates` — encoder stays frozen for the first
  N updates so only the decoder trains early.

---

## 3. Data path (how one example flows)

```
.tsv manifest ──► hubert_dataset.py ──► (mouth ROI frames, audio feats, tokens)
                                          │
                                          ▼
              resnet.py (video) + linear (audio) ──► channel-concat fusion  [hubert.py forward_features]
                                          │
                                          ▼
                          Transformer encoder  [hubert.py]
                                          │
              ┌───────────────────────────┴──────────────────────────┐
     PRETRAIN │ cluster-prediction head                     FINETUNE  │ TransformerDecoder [decoder.py]
              ▼ hubert_criterion.py (masked CE vs .km)                 ▼ label_smoothed_cross_entropy (vs .wrd tokens)
```

---

## 4. The fairseq "task" that wires it together
- `hubert_pretraining.py` — registers task `av_hubert_pretraining`, interprets
  config flags (`is_s2s`, `labels`, `modalities`, `fine_tuning`,
  `stack_order_audio`), builds the dataset and dictionary. Both pretraining and
  fine-tuning use this same task; `fine_tuning: true` + `is_s2s: true` switch it
  into seq2seq recognition mode.

---

## 5. How to verify all this yourself
Run `workspace/scripts/inspect_model.py` on the downloaded checkpoint — it prints
the real module tree and parameter counts so you can confirm ResNet frontend,
Transformer depth/width, and Base's ~103M params against these notes.

### Confirmed numbers from the actual Base checkpoint (`base_vox_iter5.pt`)
Verified by loading the checkpoint — use these exact figures in your report:

| Module | Class | Params |
|---|---|---|
| `feature_extractor_video` | `ResEncoder` (Conv3d stem + 2D ResNet trunk) | 11.6M |
| `feature_extractor_audio` | `SubModel` = **one Linear** `104→768` | **80.6K** |
| `post_extract_proj` | Linear (after channel-concat fusion) | 1.2M |
| `encoder` | `TransformerEncoder`, 12 layers × 768 dim | 89.8M |
| `final_proj` | Linear (cluster-prediction head, pretraining) | 196.9K |
| **TOTAL** | | **103.3M** |

Key config values baked into the checkpoint (match your slides):
- `encoder_layers=12`, `encoder_embed_dim=768`, `encoder_attention_heads=12` → **Base**
- `modality_fuse='concat'` → confirms channel-concatenation fusion (`f^av`)
- `modality_dropout=0.5`, `audio_dropout=0.5` → modality dropout during pretraining
- `label_rate=25` → 25 Hz frame/target rate
- `audio_feat_dim=104` with `stack_order_audio=4` → 26-dim log-mel × 4 stacked frames
- `masking_type='input'`, `sim_type='cosine'` → masking on inputs; cosine-similarity
  logits against cluster embeddings (the objective-function slide)

Note: the audio path is literally a single `Linear(104→768)` — 80.6K params vs the
video ResNet's 11.6M. This is the concrete reason audio can't dominate the fused
representation, exactly as your slide claims.
