# AV-HuBERT × AI-SPEAK — Vizuelno prepoznavanje govora

Mašinsko učenje 2 - projekat (FTN Novi Sad): analiza, reprodukcija i adaptacija
Meta modela **AV-HuBERT** za vizuelno prepoznavanje govora (čitanje sa usana) —
reprodukovano na podskupu baze LRS3 i adaptirano na srpski jezik na korpusu AI-SPEAK.

Ovaj dokument je **uputstvo za korišćenje raspoloživog koda** (priprema okruženja,
pretprocesiranje, obuka i test) i povezuje arhitekturu modela (opisanu u izveštaju,
tačka 2) sa konkretnim klasama i funkcijama u kodu.

Repozitorijum sadrži **kod, konfiguracije, skripte i dokumente**. Baze podataka
(LRS3, AI-SPEAK) i checkpoint-i modela **nisu uključeni**.

---

## 1. Verzija koda

AV-HuBERT nije samostalan program, već proširenje `fairseq` radnog okvira. Obe komponente
su kopirane direktno u repozitorijum.

- **av_hubert**: [facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert),
  commit `258fb50e155134eec2c4b49c2ae8de267075fd18`
- **fairseq** (`av_hubert/fairseq/`): [pytorch/fairseq](https://github.com/pytorch/fairseq),
  commit `afc77bdf4bb51453ce76f1572ef2ee6ddcda8eeb`

Obe komponente su pod MIT licencom (v. `av_hubert/PROVENANCE.md`). 
**Naša izmena** je podrška za LoRA adaptaciju u `av_hubert/avhubert/hubert_asr.py` (klasa `LoRALinear` +
funkcija `_apply_lora` + polja u konfiguraciji).

---

## 2. Priprema okruženja

AV-HuBERT zavisi od stare verzije `fairseq`-a (jun 2021), pa su verzije biblioteka osetljive.
Koristi se `conda` okruženje sa Python 3.9. Sve zavisnosti su zaključane u
`env/requirements.txt` (Linux/CUDA) odnosno `env/requirements_mac.txt` (macOS/CPU).

```bash
git clone https://github.com/MarkoMrdja/av-hubert-ai-speak.git
cd av-hubert-ai-speak
bash workspace/scripts/infra/setup_env.sh .    # conda env + zavisnosti (bilo koja CUDA GPU mašina)
```

Skripta `setup_env.sh` automatizuje ceo postupak: pravi conda okruženje, instalira zavisnosti,
instalira vendovan `fairseq` (editable) i gradi njegove Cython ekstenzije, potom instalira
`omegaconf`/`hydra` (koje zahtevaju stariji `pip`).

### 2.1 Preuzimanje podataka i modela

Baze podataka i checkpoint-i **nisu** u repozitorijumu; preuzimaju se zasebno.

**Pretrenirani model + dlib modeli** (BASE checkpoint `base_vox_iter5.pt` sa Meta-inog servera,
+ dlib detektor lica i prediktor 68 tačaka, potrebni za LRS3 pretprocesiranje):

```bash
bash workspace/scripts/common/download_checkpoint.sh   # -> workspace/checkpoints/
```

**LRS3** (engleski; sa proverenog HuggingFace mirror-a, jer zvanična distribucija više nije
dostupna) — [huggingface.co/datasets/TheNHz/ellipsis-lrs3-raw](https://huggingface.co/datasets/TheNHz/ellipsis-lrs3-raw)
(licenca CC BY 4.0):

```bash
bash workspace/scripts/common/download_lrs3.sh         # -> data/lrs3_raw/
```

**AI-SPEAK** (srpski; Kaggle, licenca CC BY-NC-SA 4.0) —
[kaggle.com/datasets/tijananosek/ai-speak-database-video-and-audio](https://www.kaggle.com/datasets/tijananosek/ai-speak-database-video-and-audio).
Zahteva Kaggle nalog + API token (`~/.kaggle/kaggle.json`); raspakuje foldere `spk01 .. spk30`,
na koje se zatim pokaže preko `AISPEAK_ROOT` u `aispeak/preprocess.sh`:

```bash
bash workspace/scripts/common/download_aispeak.sh      # -> data/aispeak_raw/
```

---

## 3. Pretprocesiranje podataka

Model ne čita sirov video direktno. Skripte su grupisane po fazi u `workspace/scripts/`:
`lrs3/` (LRS3), `aispeak/` (srpski), `common/` (zajedničko), `infra/` (okruženje).

### 3.1 LRS3 (engleski, provera toka)

Sirovi LRS3 klipovi (mp4 sa licem + `.txt`) pretvaraju se skriptom `lrs3/preprocess.sh`, koja
objedinjuje: izbor podskupa (`make_subset.py`), detekciju 68 tačaka lica (dlib), isecanje
regiona usana (ROI) na 96×96, izdvajanje zvuka na 16 kHz, i pravljenje manifesta + rečnika
(`build_manifest.py`).

### 3.2 AI-SPEAK (srpski)

AI-SPEAK video je već anonimizovan (lice pikselizovano, ostatak crn), pa dlib detekcija
otkazuje. Skripta `aispeak/preprocess.sh` koristi drugačiji tok:

```bash
# uredi CONFIG blok u preprocess.sh (AISPEAK_ROOT, govornici, vokabular), pa:
bash workspace/scripts/aispeak/preprocess.sh
```

Tok se sastoji iz tri koraka:
1. **`prepare.py`** — čita po-govorničke Excel metapodatke, iseca ROI usana heuristikom
   graničnog okvira (bez dlib-a; lice se nalazi kao svetli region na crnoj pozadini), seče
   klipove na govorni deo preko `.align` poravnanja, i resempluje video na 96×96 @25fps sive
   tonove i audio na 16 kHz.
2. **`count_frames.py`** — broji kadrove video i audio zapisa.
3. **`build_manifest.py`** — pravi fairseq data dir (`.tsv`/`.wrd`/`dict`) i **novi srpski
   SentencePiece vokabular** (veličine 500) iz trening transkripata; podela je disjunktna po
   govornicima.

Rezultat je data dir (`{train,valid,test}.tsv/.wrd`, `dict.wrd.txt`, `spm_unigram500.model`)
spreman za obuku.

**Format `.tsv` manifesta** (po redu, tab-razdvojeno): prvi red je koreni direktorijum, a
svaki naredni: `<id>  <putanja_video.mp4>  <putanja_audio.wav>  <br_video_kadrova>
<br_audio_uzoraka>`.

---

## 4. Obuka (fine-tuning)

Obuka se pokreće alatom `fairseq-hydra-train`, koji čita YAML konfiguraciju iz `workspace/configs/`
(Hydra sistem). Radi jednostavnosti, skripta `common/run_finetune.sh` objedinjuje poziv (bira
konfiguraciju prema prvom argumentu i popunjava putanje).

```bash
# LRS3 (LoRA):
DATA=.../subset_data CKPT=.../base_vox_iter5.pt EXP=.../lrs3_lora \
  bash workspace/scripts/common/run_finetune.sh lora

# AI-SPEAK (srpski) — LoRA na dekoderu:
DATA=.../ser_data CKPT=.../base_vox_iter5.pt EXP=.../aispeak_dec \
  bash workspace/scripts/common/run_finetune.sh aispeak

# AI-SPEAK — LoRA na enkoderu+dekoderu:
#   ... run_finetune.sh aispeak-enc
```

Konfiguracije: `aispeak_lora_decoder.yaml` (samo dekoder) i `aispeak_lora_encdec.yaml`
(enkoder+dekoder). Ključne LoRA opcije: `lora_enable: true`, `lora_scope: decoder|all`,
`lora_r: 16`, `lora_alpha: 16`, `lora_targets: 'q_proj,v_proj'`.

Opcioni env override-i: `LORA_R` (rang), `MAX_UPDATE` (gornja granica koraka), `PATIENCE`
(rano zaustavljanje). Primer za pretragu ranga: `LORA_R=16 MAX_UPDATE=3000 PATIENCE=8 ...`.

Polazni enkoder se bira preko `CKPT`: `base_vox_iter5.pt` (samo pretrenirani) ili enkoder
izdvojen iz engleskog VSR modela pomoću `aispeak/extract_encoder.py` (engleski dekoder se
odbacuje, srpski se obučava iznova).

---

## 5. Test (inference i WER)

Dekodiranje test skupa i računanje WER-a radi se skriptom `common/evaluate.sh`, koja obmotava
`avhubert/infer_s2s.py` (seq2seq beam-search dekodiranje).

```bash
DATA=.../ser_data \
FT_CKPT=.../aispeak_dec/checkpoints/checkpoint_best.pt \
RESULTS=.../aispeak_dec/decode_test \
GEN_SUBSET=test MODALITIES=video \
  bash workspace/scripts/common/evaluate.sh
```

`GEN_SUBSET` bira skup (`test` | `valid`); `MODALITIES` bira modalitet (`video` za VSR;
`audio,video` za AVSR). WER se ispisuje u `$RESULTS/wer.*`, a hipoteze (REF/HYP parovi) u
`hypo-*.json`.

---

## 6. Povezivanje arhitekture i koda

Sledeća tabela povezuje koncepte iz izveštaja (tačka 2) sa klasama/funkcijama u kodu. Sve
putanje su relativne u odnosu na `av_hubert/avhubert/`.

### 6.1 Model — dve faze

| Faza | Klasa modela | Ciljna funkcija | Meta |
|---|---|---|---|
| Pretrening | `AVHubertModel` (`hubert.py`) | `AVHubertCriterion` (`hubert_criterion.py`) | ID-jevi klastera |
| Fine-tuning (VSR) | `AVHubertSeq2Seq` (`hubert_asr.py`) | `label_smoothed_cross_entropy` (fairseq) | tokeni transkripta |

Fine-tuning „obmotava" pretrenirani `AVHubertModel` kao enkoder (`HubertEncoderWrapper`) i
dodaje `TransformerDecoder` (`decoder.py`).

### 6.2 Komponente arhitekture

| Koncept (izveštaj, tačka 2) | Realizacija u kodu |
|---|---|
| Video ekstraktor (3D stem + ResNet-18) | `resnet.py` (klasa `ResEncoder`), instanciran u `hubert.py` |
| Audio ekstraktor (jedan linearni sloj) | `hubert.py` (klasa `SubModel` sa `resnet=None`) |
| Fuzija konkatenacijom po kanalima | `hubert.py`, `AVHubertModel.forward` (`modality_fuse='concat'`) |
| Transformer enkoder | `hubert.py` (`TransformerEncoder`) |
| Modality dropout | `hubert.py`, `AVHubertModel.forward` (polja `modality_dropout`, `audio_dropout`) |
| Seq2seq dekoder (fine-tuning) | `hubert_asr.py` (`AVHubertSeq2Seq`) + `decoder.py` (`TransformerDecoder`) |
| Učitavanje podataka (ROI, audio, tokeni) | `hubert_dataset.py` (klasa `AVHubertDataset`) |
| fairseq „task" (spaja sve) | `hubert_pretraining.py` (`AVHubertPretrainingTask`) |
| Beam-search dekodiranje | `sequence_generator.py`, pokretano iz `infer_s2s.py` |

### 6.3 Ciljne funkcije

**Pretrening — maskirana predikcija klastera** (`hubert_criterion.py`, klasa `AVHubertCriterion`).
Ciljna funkcija je unakrsna entropija nad klasterima, podeljena na maskirane i nemaskirane
kadrove; u praksi je težina nemaskiranih nula (predviđaju se samo maskirani). Logiti se računaju
kosinusnom sličnošću između izlaza enkodera i naučenih klaster-embedinga (podešeno u
`hubert.py`). Ciljevi (`z_t`) su ID-jevi klastera po kadru.

**Fine-tuning — label-smoothed cross-entropy** nad tokenima transkripta. Kriterijum je
fairseq-ov `label_smoothed_cross_entropy` (`label_smoothing = 0.1`), izabran u konfiguraciji
(`criterion._name`). Dekoder je autoregresivan i obučava se teacher forcing-om.

### 6.4 Naša izmena — LoRA

Podrška za LoRA je implementirana u `hubert_asr.py`: klasa **`LoRALinear`** (obmotava zamrznuti
`nn.Linear` i dodaje niskorangovanu ispravku `y = Wx + (α/r)·B(Ax)`; matrica B inicijalizovana
na nulu) i funkcija **`_apply_lora`** (zamenjuje ciljne linearne slojeve — `q_proj`/`v_proj`
projekcije pažnje — njihovim LoRA verzijama i zamrzava sve ostalo).

LoRA se aktivira preko `model:` bloka u YAML konfiguraciji obuke (npr.
`aispeak_lora_decoder.yaml`): polje `lora_enable` je glavni prekidač, uz `lora_r` (rang),
`lora_alpha` (skaliranje), `lora_dropout`, `lora_targets` (ciljni slojevi) i `lora_scope`
(`decoder` | `encoder` | `all`). Ako je `lora_enable: false`, model se gradi normalno i sve se
trenira (obično fine-tuniranje); ako je `true`, `build_model` poziva `_apply_lora` i ubacuje
adaptere. Ista klasa i ista funkcija za izgradnju modela pokrivaju oba slučaja — razlikuje ih
samo konfiguracija.

### 6.5 Provera arhitekture

`workspace/scripts/common/inspect_model.py` učitava checkpoint i ispisuje stvarnu strukturu
modula i broj parametara — čime se potvrđuju ResNet frontend, dubina/širina enkodera i ukupno
~103M parametara (BASE) navedeni u izveštaju.

---

## 7. Struktura projekta

```
av_hubert/          vendovan upstream (av_hubert + fairseq) uklj. našu LoRA izmenu
  PROVENANCE.md       poreklo (upstream commit-ovi) + naša izmena
env/                requirements (CUDA + macOS) + verify_env.py
workspace/
  configs/          konfiguracije obuke (LRS3 + AI-SPEAK LoRA)
  scripts/
    common/         run_finetune.sh, evaluate.sh, inspect_model.py, download_*
    lrs3/           priprema LRS3 podskupa
    aispeak/        prepare.py, build_manifest.py, extract_encoder.py, validate_crops.py, preprocess.sh
    infra/          setup_env.sh, sync_remote.sh
  experiments/      logovi obuke + rezultati dekodiranja (checkpoint-i izuzeti)
deliverables/       prezentacija + izveštaj (PDF)
```

---

## 8. Rezultati (sažetak)

Detaljni rezultati su u izveštaju (`deliverables/`). Ukratko:

- **LRS3 (provera toka):** WER očekivano visok (podskup od 3.000 klipova, kratka obuka) — cilj je
  bila potvrda da ceo tok radi, ne reprodukcija rezultata iz rada.
- **AI-SPEAK (srpski):** najbolji rezultat **73,1% WER** na neviđenim govornicima (LoRA
  enkoder+dekoder, r=16). Model pouzdano prepoznaje kratke strukturisane iskaze (brojevi, dani),
  a slabije duži slobodan govor — usko grlo je količina označenih srpskih podataka (~2,7 h).
