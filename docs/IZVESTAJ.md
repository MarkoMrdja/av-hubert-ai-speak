# Vizuelno prepoznavanje govora pomoću AV-HuBERT modela

**Finalni izveštaj**
Marko Mrđa · Mašinsko učenje · Fakultet tehničkih nauka, Univerzitet u Novom Sadu

> Finalni izveštaj (task 6). Sekcije `Metode i materijali` popunjene proverenim
> činjenicama o arhitekturi (potvrđeno učitavanjem stvarnog `base_vox_iter5.pt`
> checkpointa). Rezultati popunjeni iz stvarnih eksperimenata (LRS3 podskup +
> AI-SPEAK adaptacija na srpski, RunPod A40, avgust 2026).

---

## 1. Uvod

- **Problem: vizuelno prepoznavanje govora (VSR / lip reading).** Prepoznavanje
  izgovorenog teksta isključivo iz pokreta usana, bez zvučnog signala.
- **Zašto je značajno:**
  - Pomoć osobama sa oštećenim sluhom i u asistivnim tehnologijama.
  - Robusno prepoznavanje govora u bučnim uslovima (vizuelni signal dopunjava
    audio kad je audio degradiran).
  - Primene: čitanje govora sa video-snimaka bez tona, biometrija, HCI.
- **Ključni izazovi:**
  - *Homofeme* — različiti glasovi (npr. /p/, /b/, /m/) izgledaju identično na
    usnama; potreban je vremenski kontekst, jedan kadar nije dovoljan.
  - *Zavisnost od označenih podataka* — raniji pristupi (npr. Makino i dr., 2019)
    tražili su desetine hiljada sati transkribovanih snimaka; označavanje je skupo
    i nedostupno za većinu jezika.
- **Doprinos AV-HuBERT-a (motivacija rada):** samonadgledano učenje audio-vizuelnih
  reprezentacija smanjuje potrebu za označenim podacima ~1000× uz bolji rezultat.
- **Cilj ovog rada:** analizirati arhitekturu i ciljnu funkciju AV-HuBERT-a,
  reprodukovati postupak fine-tuninga na podskupu **LRS3** baze (provera ispravnosti
  celog toka), i **adaptirati model na srpski jezik** na AI-SPEAK korpusu putem
  parametarski-efikasne LoRA adaptacije.

---

## 2. Metode i materijali

### 2.1 Model — AV-HuBERT (arhitektura)

*(Napomena: ovde se NE povezuje sa kodom — to je zaseban dokument/uputstvo.)*

- **Opšta ideja:** proširenje HuBERT-a (samonadgledani okvir za govor, Hsu i dr.
  2021) na dva modaliteta — niz slika usana (video) + akustična obeležja (audio).
- **Dva ekstraktora obeležja:**
  - *Video:* modifikovani ResNet-18 nad regionom usana (ROI). Ulaz je Conv3D
    "stem" (kernel 5×7×7) → 2D ResNet trunk. **~11,6M parametara.**
  - *Audio:* **jedan linearni sloj** nad stakovanim log-mel obeležjima
    (dimenzija 104 = 26 log-mel × 4 stakovana kadra). **~80,6K parametara** — mali
    kapacitet namerno, da audio ne dominira nad vizuelnim modalitetom.
- **Fuzija:** konkatenacija po kanalima (`modality_fuse = concat`) → linearna
  projekcija u dimenziju modela (768).
- **Transformer enkoder:** deljeni enkoder nad fuzionisanom reprezentacijom.
  - BASE: 12 blokova, 768 dim, 12 glava → **~103M parametara ukupno** (korišćeno).
  - LARGE: 24 bloka, 1024 dim → ~325M (nije korišćeno zbog ograničenja GPU-a).
- **Modality dropout** (`0.5`): tokom pretreninga se nasumično izostavlja ceo audio
  ILI video stream, čime se sprečava oslanjanje na samo jedan modalitet i jača
  vizuelni enkoder.
- **Frekvencija kadrova / ciljeva:** `label_rate = 25` Hz.

### 2.2 Obuka i ciljna funkcija

- **Faza 1 — Pretrening (samonadgledano, NIJE rađeno ovde):**
  - K-means klasterizacija obeležja → svakom kadru se dodeljuje diskretni
    **klaster-ID** (pseudo-oznaka). Klasteri se iterativno preračunavaju iz sve
    boljih obeležja modela (5 iteracija).
  - **Maskirana predikcija klastera:** deo kadrova se maskira; model predviđa
    klaster-ID maskiranih kadrova. Logiti se računaju **kosinusnom sličnošću**
    (`sim_type = cosine`) između izlaza enkodera i naučenih klaster-embedinga.
  - **Ciljna funkcija (masked cluster prediction):**

    $$ L = \alpha \sum_{t \in M} \mathrm{CE}\big(p_t, z_t\big) \;+\; (1-\alpha)\sum_{t \notin M} \mathrm{CE}\big(p_t, z_t\big) $$

    gde je $M$ skup maskiranih kadrova, $z_t$ klaster-ID kadra $t$, $p_t$
    predviđena raspodela nad $K$ klastera. U praksi težina nemaskiranih kadrova je
    obično 0 (predviđaju se samo maskirani).
  - *Korišćeni checkpoint:* BASE, pretreniran na LRS3 + VoxCeleb2 (`base_vox_iter5`).

- **Faza 2 — Fine-tuning / adaptacija (nadgledano, RAĐENO na LRS3 i AI-SPEAK):**
  - Na pretrenirani enkoder se dodaje **Transformer dekoder** (seq2seq,
    `av_hubert_seq2seq`, 6 slojeva).
  - Dekoder autoregresivno predviđa **subword tokene transkripta**
    (SentencePiece unigram vokabular izgrađen iz transkripata).
  - **Ciljna funkcija:** label-smoothed cross-entropy (`label_smoothing = 0.1`).
  - Za srpsku adaptaciju korišćena je **LoRA** (Hu i dr., 2021): ceo pretrenirani
    model se zamrzava, obučavaju se samo niskorangovani adapteri (`r=8`) na
    projekcijama pažnje `q_proj`/`v_proj` — 0,18–0,37% parametara. Pogodno za mali
    korpus jer otporno na preprilagođavanje. (Puna implementacija u
    `our_code_changes/`; objašnjenje u uputstvu.)

### 2.3 Podaci

- **LRS3** (Oxford VGG / TED): rečenice iz TED/TEDx govora, engleski. (LRS2 je
  prvobitno planiran, ali BBC nije odobrio pristup; LRS3 je baza koju sam rad
  benchmark-uje. Korišćen verifikovani HuggingFace mirror.)
  - Korišćen **mali podskup (~3.000 klipova)** samo za **proveru ispravnosti toka**
    obuke i evaluacije — ne za reprodukciju rezultata iz rada.
  - Preprocesiranje: detekcija lica i 68 tačaka (dlib) → isecanje ROI usana 96×96 →
    audio na 16 kHz → manifesti (`.tsv`), transkripti (`.wrd`), SentencePiece vokabular.
- **AI-SPEAK** korpus (Kaggle, CC BY-NC-SA 4.0): 30 govornika (15 M / 15 Ž),
  bilingvalni. Korišćen **srpski** deo: **~2.375 iskaza ≈ ~2,7 h**. Video: frontalna
  kamera 100 fps, **anonimizovano** (usne oštre, ostatak pikselizovan/crn); audio
  22,05 kHz. Preprocesiranje se razlikuje od LRS3 (v. sekciju 3.2 i
  `docs/AISPEAK_REZULTATI.md`): **heuristika graničnog okvira sadržaja** umesto dlib-a
  (dlib pada na anonimizovanim snimcima), sečenje na iskaz preko `.align`, srpski
  SentencePiece vokabular (500), split disjunktan po govornicima.
  - *Cross-lingual transfer:* vizuelni enkoder je pretreniran na engleskom
    (LRS3+Vox); pokreti usana su uglavnom jezički nezavisni, pa se enkoder prenosi,
    dok se dekoder i vokabular obučavaju iznova za srpski.

### 2.4 Eksperimentalna postavka

- Hardver: **rentani RunPod GPU** — NVIDIA A40 (48 GB), EU-SE-1. (Preprocesiranje
  lokalno na CPU; samo obuka na GPU, ukupno ~2 h ≈ ~2 USD.)
- Konfiguracija (jedan GPU): fp16, `max_tokens=2000`, `update_freq=4` (akumulacija
  gradijenta), `lr=3e-3`, tri_stage scheduler (warmup 400, decay 4000),
  `max_update=5000`. Modalitet: samo video (VSR).
- LoRA (AI-SPEAK): `r=8`, `alpha=16`, targeti `q_proj,v_proj`.
- Verzija koda: av_hubert `258fb50e` (git submodul), fairseq `afc77bdf` (v. uputstvo).
- Metrika: **WER** (Word Error Rate), beam search dekodiranje (`beam=50`).

---

## 3. Rezultati i diskusija

### 3.1 LRS3 (provera ispravnosti toka)

Cilj nije bio dostizanje rezultata iz rada, već potvrda da ceo tok (preprocesiranje
→ obuka → dekodiranje → WER) ispravno radi. Zato je korišćen mali podskup.

| Konfiguracija | Obučivih par. | Test WER |
|---|---|---|
| Pun fine-tuning | 161M (100%) | 74,4% |
| LoRA (dekoder q/v) | 295K (0,18%) | 93,6% |

WER je visok u odnosu na 26–32% iz rada **namerno**: ~3.000 klipova / 3.000 koraka
naspram 433 h / 30.000 koraka. Tok je potvrđen kao ispravan i LoRA put validiran pre
prelaska na srpsku adaptaciju.

### 3.2 AI-SPEAK (adaptacija na srpski)

Detaljno u `docs/AISPEAK_REZULTATI.md`. Tri eksperimenta (test: neviđeni govornici
spk27–30, 2535 reči):

| # | Polazni enkoder | LoRA opseg | WER |
|---|---|---|---:|
| 1 | engleski VSR | samo dekoder | **72,2%** |
| 2 | engleski VSR | enkoder+dekoder | 75,9% |
| 3 | samo pretrening | enkoder+dekoder | **71,7%** |

Raspodela kvaliteta (Run 3, 311 klipova): 35% klipova ≥80% reči tačno (kratki
strukturisani iskazi — brojevi, komande, dani), 56% uglavnom pogrešno (duge slobodne
rečenice). Model uči srpsko čitanje sa usana, ali ograničeno na ~2,7 h podataka.

**Nalazi:** (1 vs 2) adaptacija enkodera nije pomogla — preprilagođavanje na malom
skupu. (2 vs 3) engleski VSR enkoder nije doneo prednost nad sirovim pretreniranim.
Dva najbolja rezultata (71,7% i 72,2%) su u okviru šuma; rezultat je dominantno
određen **količinom podataka**, ne arhitektonskim izborima.

### 3.3 Diskusija
- Ograničenja: mali podskupovi, kratka obuka vs. originalni rad; po jedan pokretaj
  svakog eksperimenta (bez intervala poverenja, mali test skup) — razlike od nekoliko
  procenata tumačiti kao u okviru šuma.
- Šta bi poboljšalo rezultat: više označenih srpskih podataka (najvažnije); niži LoRA
  rang (jača regularizacija); LARGE model; AVSR (audio+video) umesto samo video.

---

## 4. Zaključak
- Analizirana je arhitektura i ciljna funkcija AV-HuBERT-a i potvrđena učitavanjem
  stvarnog checkpointa (103M parametara, BASE).
- Reprodukovan je ceo tok obuke/evaluacije na LRS3 podskupu (provera ispravnosti),
  i implementirana LoRA adaptacija.
- Model je adaptiran na **srpski jezik** (AI-SPEAK) putem LoRA: najbolji rezultat
  **71,7% WER** na neviđenim govornicima. Model pouzdano prepoznaje kratke
  strukturisane iskaze, ali mu nedostaje podataka (~2,7 h) za slobodan govor.
- Ključni uvid iz ablacija: na malom korpusu ni adaptacija enkodera ni englesko
  VSR-inicijalizovanje ne daju jasnu prednost — usko grlo je **količina označenih
  srpskih podataka**, a ne arhitektura. Vizuelni enkoder se dobro prenosi među
  jezicima.
- Mogući nastavak: više srpskih podataka, jača regularizacija (niži LoRA rang),
  LARGE model, AVSR.

---

## Reference
1. Shi, B., Hsu, W.-N., Lakhotia, K., Mohamed, A. *Learning Audio-Visual Speech
   Representation by Masked Multimodal Cluster Prediction.* arXiv:2201.02184 (2022).
2. Shi, B., Hsu, W.-N., Mohamed, A. *Robust Self-Supervised Audio-Visual Speech
   Recognition.* arXiv:2201.01763 (2022).
3. Hsu, W.-N. i dr. *HuBERT: Self-Supervised Speech Representation Learning by
   Masked Prediction of Hidden Units.* (2021).
4. Afouras, T. i dr. *LRS3-TED: a large-scale dataset for visual speech
   recognition.* arXiv:1809.00496 (2018).
5. Hu, E. J. i dr. *LoRA: Low-Rank Adaptation of Large Language Models.*
   arXiv:2106.09685 (2021).
6. Nosek, T. i dr. *AI-SPEAK baza* (Kaggle, CC BY-NC-SA 4.0).
