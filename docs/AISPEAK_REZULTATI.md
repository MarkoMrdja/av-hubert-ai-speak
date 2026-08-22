# AI-SPEAK — rezultati i diskusija (za izveštaj, sekcija 3.2)

> Materijal za finalni izveštaj (task 6). Popunjava sekciju *3.2 AI-SPEAK* i dopunjava
> *2.3 Podaci* / *2.4 Eksperimentalna postavka* stvarnim brojevima iz eksperimenata
> izvedenih 2026-08-21 (RunPod A40, EU-SE-1).

## Podaci (AI-SPEAK, srpski)

- Korpus: **AI-SPEAK** (Kaggle, `tijananosek/ai-speak-database-video-and-audio`,
  licenca CC BY-NC-SA 4.0). 30 govornika (15 M / 15 Ž), bilingvalni (srpski + engleski).
- Korišćen je **srpski** deo: **~2.375 iskaza ≈ ~2,7 h** govora.
- Video: frontalna kamera, 100 fps, rezolucija 1080×1920, **anonimizovano** (usne
  oštre, ostatak lica pikselizovan/maskiran crnom bojom). Audio: 22,05 kHz mono WAV.
- **Preprocesiranje (razlike u odnosu na LRS3):**
  - *Isecanje ROI-a bez dlib-a.* dlib detektor lica ne radi na anonimizovanim
    snimcima (pao na ~50% testiranih govornika). Umesto toga korišćena je
    **heuristika graničnog okvira sadržaja**: pošto je kadar pretežno crn, region
    lica se pronalazi kao skup ne-crnih piksela, iseca se kvadrat centriran na usta
    i skalira na 96×96 sivih tonova. Radi na svih 30 govornika. (Detalji:
    `docs/AISPEAK_ROI_CROP_DECISION.md`.)
  - *Sečenje na iskaz.* Klipovi su neisečeni (~1–4 s tišine/šuma po klipu). Pomoću
    `.align` datoteka (poravnanja na nivou reči) video i audio se seku na govorni
    deo, čime AI-SPEAK dobija distribuciju sličnu LRS3 (tesno oko iskaza).
  - SentencePiece unigram vokabular veličine **500** izgrađen iz trening transkripata
    (mali korpus: 79% reči se javlja samo jednom → subword jedinice su neophodne).
    Transkripti se normalizuju na mala slova (label procesor u AV-HuBERT-u poziva
    `.lower()` pre tokenizacije).
- **Split po govornicima (disjunktan):** train = spk01–23 (1822 iskaza),
  valid = spk24–26 (238), **test = spk27–30 (315 iskaza, 2535 reči)**. Test su
  **neviđeni govornici**, čime se meri generalizacija.

## Eksperimentalna postavka

- Hardver: **RunPod, NVIDIA A40 (48 GB), EU-SE-1**. (Prvobitni plan NTP-122 zamenjen
  rentanim GPU-om; sav trening ~2 h ukupno, ~2 USD.)
- Metod adaptacije: **LoRA** (Hu i dr., 2021) — pretrenirani model se zamrzava, obučavaju
  se samo niskorangovani adapteri (`r=8`, `alpha=16`) na `q_proj`/`v_proj` projekcijama
  pažnje. Obučivih parametara: 0,18–0,37% (295K–590K od 160M).
- Konfiguracija: fp16, `max_tokens=2000`, `update_freq=4`, `lr=3e-3`, tri_stage
  (warmup 400, decay 4000), `max_update=5000`. Modalitet: **samo video (VSR)**.
- Polazni checkpoint (dve varijante, v. eksperimente): A = pretrenirani enkoder
  (`base_vox_iter5.pt`); B = enkoder izdvojen iz **engleskog VSR fine-tune** checkpointa
  (`base_vox_433h.pt`) — engleski dekoder se odbacuje (neusklađen vokabular/jezik),
  srpski dekoder se obučava iznova.
- Metrika: **WER**, beam search dekodiranje (`beam=50`).

## Rezultati

Tri eksperimenta (jedan pokretaj svaki), test na neviđenim govornicima (2535 reči):

| # | Polazni enkoder | LoRA opseg | Obučivih par. | Greške | **WER** |
|---|---|---|---|---:|---:|
| 1 | B (engleski VSR) | samo dekoder | 295K (0,18%) | 1830 / 2535 | **72,2%** |
| 2 | B (engleski VSR) | enkoder+dekoder | 590K (0,37%) | 1924 / 2535 | **75,9%** |
| 3 | A (samo pretrening) | enkoder+dekoder | 590K (0,37%) | 1817 / 2535 | **71,7%** |

**Raspodela kvaliteta (Run 3, 311 test klipova).** Prosečan WER (71,7%) sakriva
bimodalnu prirodu rezultata:

| Kvalitet predikcije | Broj klipova | Udeo |
|---|---:|---:|
| Tačno (potpuno poklapanje) | 27 | 9% |
| ≥80% reči tačno | 110 | 35% |
| 40–80% tačno | 26 | 8% |
| <40% (uglavnom pogrešno) | 175 | 56% |

Model **odlično prepoznaje kratke, strukturisane iskaze** (brojevi, komande, dani u
nedelji — trećina klipova je ≥80% tačna), a **raspada se na dugim slobodnim
rečenicama** (većina). Dakle nije "ravnomerno loš" — naučio je srpsko čitanje sa
usana za ograničene iskaze, ali mu nedostaje podataka za slobodan govor.

Primeri predikcija (Run 3, najbolji model):

```
REF : potvrdi odustani obriši pošalji dalje početak kraj ...
HYP : potvrdi odustani obriši pošalji dalje početak kraj ...        (tačno)

REF : ponedeljak utorak sreda četvrtak petak subota nedelja ...
HYP : ponedeljak utorak sreda četvrtak petak subota nedelja ...     (skoro tačno)

REF : pošto idealizujete ne samo partnera već i druge ljude razočarenje ...
HYP : pošto idealizujete ne samo partnera već i druge razočarenje razoč...  (delimično)

REF : prijava je podneta protiv načelnika resora državne bezbednosti ...
HYP : po je jei da da i i i i i i i i i ...                          (raspad na dužoj rečenici)
```

## Diskusija

**Model uči srpsko čitanje sa usana, ali ograničeno.** Kratki, ograničeni iskazi
(brojevi, komande, dani u nedelji) prepoznaju se skoro savršeno; duže slobodne
rečenice se raspadaju (ponavljanja, delimična tačnost). To je očekivano pri **~2,7 h**
podataka — red veličine manje od najmanje postavke u radu (30 h → ~46% WER).

**Nalaz 1 (Run 1 vs 2): adaptacija enkodera NIJE pomogla — čak je blago pogoršala**
(72,2% → 75,9%). Udvostručenje obučivih parametara (dodavanjem adaptera na enkoder)
na tako malom skupu vodi **preprilagođavanju**: model bolje pristaje trening
govornicima, ali lošije generalizuje na neviđeni test. Validaciona tačnost bila je
slična (~25%), a WER se razišao — potpis preprilagođavanja.

**Nalaz 2 (Run 2 vs 3): engleski VSR enkoder NIJE doneo prednost** — polazak od
sirovog pretreniranog enkodera (A) bio je čak *bolji* (71,7% vs 75,9%). Moguće
objašnjenje: engleski fine-tuning specijalizuje enkoder za engleske obrasce, što je
delom neusklađeno sa srpskim; sirovi (samonadgledani, jezički neutralan) enkoder je
neutralnija polazna tačka.

**Glavni, oprezan zaključak.** Dva najbolja rezultata (71,7% i 72,2%) su
**praktično nerazlučiva** (~0,5% na 315 klipova). Dakle: na ~2,7 h srpskog, ni
englesko VSR-inicijalizovanje ni adaptacija enkodera ne daju jasnu prednost;
rezultati se grupišu oko **~72–76% WER**, dominantno određeni **ograničenjem
podataka**, a ne ovim arhitektonskim izborima. Vizuelni enkoder se dobro prenosi
među jezicima (pokreti usana su uglavnom jezički nezavisni), pa usko grlo nije jezik
enkodera već količina označenih srpskih podataka.

**Ograničenja.** Po jedan pokretaj svakog eksperimenta (bez intervala poverenja),
mali test skup (315 klipova) — razlike od nekoliko procenata treba tumačiti kao
**u okviru šuma**, ne kao definitivne. Odbranjive tvrdnje: (a) adaptacija enkodera
ne pomaže na ovom obimu podataka; (b) izbor polaznog enkodera je približno nebitan.

**Šta bi poboljšalo rezultat:** više označenih srpskih podataka (najvažnije);
niži LoRA rang (r=4/2) kao jača regularizacija za mali skup [eksperiment u toku];
LARGE model; AVSR (audio+video) umesto samo video.
