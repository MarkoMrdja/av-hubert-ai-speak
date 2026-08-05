# Vizuelno prepoznavanje govora pomoću AV-HuBERT modela

**Finalni izveštaj — skelet**
Marko Mrđa · Mašinsko učenje · Fakultet tehničkih nauka, Univerzitet u Novom Sadu

> Skelet finalnog izveštaja (task 6). Sekcije `Metode i materijali` su unapred
> popunjene proverenim činjenicama o arhitekturi (potvrđeno učitavanjem stvarnog
> `base_vox_iter5.pt` checkpointa). Rezultati se popunjavaju posle obuke.
> `[POPUNITI]` označava mesta koja zavise od eksperimenata.

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
  reprodukovati postupak fine-tuninga na podskupu LRS2 baze, i evaluirati na
  AI-SPEAK korpusu.

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

- **Faza 2 — Fine-tuning (nadgledano, RAĐENO na LRS2):**
  - Na pretrenirani enkoder se dodaje **Transformer dekoder** (seq2seq,
    `av_hubert_seq2seq`, 6 slojeva).
  - Dekoder autoregresivno predviđa **subword tokene transkripta**
    (SentencePiece unigram vokabular izgrađen iz transkripata).
  - **Ciljna funkcija:** label-smoothed cross-entropy (`label_smoothing = 0.1`).
  - Enkoder je zamrznut prvih $N$ koraka (`freeze_finetune_updates`), pa se
    odmrzava — u početku se obučava samo dekoder.

### 2.3 Podaci

- **LRS2** (Oxford VGG / BBC): rečenice iz BBC emisija, engleski.
  - Splitovi: pretrain ~96.318, train ~45.839, val ~1.082, test ~1.243 iskaza.
  - Korišćen **podskup od ~[POPUNITI]h** za obuku (zbog ograničenih resursa).
  - Preprocesiranje: detekcija lica i 68 tačaka (dlib) → isecanje ROI usana 96×96 →
    audio na 16 kHz → manifesti (`.tsv`), transkripti (`.wrd`), SentencePiece vokabular.
- **AI-SPEAK** korpus: [POPUNITI — jezik, broj sati/iskaza, format, uslovi snimanja].
  - Napomena o *cross-lingual transferu*: vizuelni/audio enkoder je pretreniran na
    engleskom (LRS3+Vox); za drugi jezik enkoder se prenosi, ali se dekoder i
    vokabular obučavaju iznova.

### 2.4 Eksperimentalna postavka

- Hardver: RTX 3050 Laptop (4 GB VRAM) za LRS2; NTP-122 za AI-SPEAK.
- Konfiguracija (jedan GPU): fp16, `max_tokens=[POPUNITI]`, `update_freq=8`
  (akumulacija gradijenta), `max_update=[POPUNITI]`, `lr=1e-3`, tri_stage scheduler.
- Verzija koda: av_hubert `258fb50e`, fairseq `afc77bdf` (v. uputstvo).
- Metrika: **WER** (Word Error Rate), beam search dekodiranje (`beam=50`).

---

## 3. Rezultati i diskusija

### 3.1 LRS2 (reprodukcija)
- Tabela: konfiguracija × WER na test/valid skupu. [POPUNITI]
- Kriva učenja (loss/accuracy vs. koraci) — screenshot iz TensorBoard-a. [POPUNITI]
- Komentar: uticaj veličine podskupa, dužine obuke, zamrzavanja enkodera. [POPUNITI]

### 3.2 AI-SPEAK
- Tabela WER za konkretne slučajeve. [POPUNITI]
- Diskusija: cross-lingual transfer, kvalitet podataka, poređenje sa LRS2. [POPUNITI]

### 3.3 Diskusija
- Ograničenja: mali podskup, jedan GPU, kratka obuka vs. originalni rad.
- Šta bi poboljšalo rezultat: više podataka, duža obuka, LARGE model, AVSR (audio+video).

---

## 4. Zaključak
- Najvažniji aspekti urađenog: [POPUNITI — reprodukovan postupak, dobijen WER,
  potvrđeno da samonadgledani pretrening + mali fine-tuning daje upotrebljiv VSR].
- Glavni uvidi i mogući nastavak rada.

---

## Reference
1. Shi, B., Hsu, W.-N., Lakhotia, K., Mohamed, A. *Learning Audio-Visual Speech
   Representation by Masked Multimodal Cluster Prediction.* arXiv:2201.02184 (2022).
2. Shi, B., Hsu, W.-N., Mohamed, A. *Robust Self-Supervised Audio-Visual Speech
   Recognition.* arXiv:2201.01763 (2022).
3. Hsu, W.-N. i dr. *HuBERT: Self-Supervised Speech Representation Learning by
   Masked Prediction of Hidden Units.* (2021).
4. Afouras, T. i dr. *LRS2: Lip Reading Sentences in the Wild.* (2018).
5. [POPUNITI — AI-SPEAK referenca ako postoji]
