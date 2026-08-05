const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
  TableOfContents, PageBreak, Math: DMath, MathRun,
} = require("docx");

const FONT = "Arial";
const CW = 9360;

const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const P = (t) => new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.JUSTIFIED, children: [new TextRun(t)] });
const runs = (arr) => new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.JUSTIFIED, children: arr });
const bold = (t) => new TextRun({ text: t, bold: true });
const it = (t) => new TextRun({ text: t, italics: true });
const txt = (t) => new TextRun(t);
const BULLET = (t) => new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 60 }, children: [new TextRun(t)] });
const bulletRuns = (arr) => new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 60 }, children: arr });
const NUM = (t) => new Paragraph({ numbering: { reference: "n", level: 0 }, spacing: { after: 60 }, children: [new TextRun(t)] });
const REF = (t) => new Paragraph({ numbering: { reference: "r", level: 0 }, spacing: { after: 60 }, children: [new TextRun(t)] });
const code = (lines) => lines.map((ln, i) => new Paragraph({
  shading: { fill: "F2F2F2", type: ShadingType.CLEAR }, spacing: { after: i === lines.length - 1 ? 120 : 0 },
  children: [new TextRun({ text: ln || " ", font: "Consolas", size: 18 })],
}));

const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths) {
  const mk = (cells, head) => new TableRow({ children: cells.map((c, i) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { fill: "2E75B6", type: ShadingType.CLEAR } : { fill: "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text: c, bold: head, color: head ? "FFFFFF" : "000000", size: 20 })] })],
  })) });
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths,
    rows: [mk(headers, true), ...rows.map((r) => mk(r, false))] });
}

const children = [];

children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1400, after: 160 },
  children: [new TextRun({ text: "Vizuelno prepoznavanje govora pomoću AV-HuBERT modela", bold: true, size: 40 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "Finalni izveštaj", size: 28 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 800 },
  children: [new TextRun({ text: "Marko Mrđa · Mašinsko učenje · FTN, Univerzitet u Novom Sadu · 2026", size: 22, color: "666666" })] }));
children.push(new Paragraph({ children: [new TextRun({ text: "Sadržaj", bold: true, size: 28 })] }));
children.push(new TableOfContents("Sadržaj", { hyperlink: true, headingStyleRange: "1-2" }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// 1. UVOD
children.push(H1("1. Uvod"));
children.push(P("Vizuelno prepoznavanje govora (engl. Visual Speech Recognition, VSR), poznato i kao čitanje sa usana, jeste zadatak prepoznavanja izgovorenog teksta isključivo iz pokreta usana govornika, bez zvučnog signala. Za razliku od klasičnog prepoznavanja govora, ovde model raspolaže samo vizuelnom informacijom — nizom slika regiona usana kroz vreme."));
children.push(P("Problem je značajan iz više razloga. Vizuelni signal dopunjava sluh i posebno je koristan u bučnim uslovima, gde je audio degradiran. Ima primenu u asistivnim tehnologijama za osobe sa oštećenim sluhom, u prepoznavanju govora sa nemih snimaka, kao i u multimodalnim sistemima gde kombinovanje audio i video modaliteta povećava robusnost."));
children.push(P("Ključni izazov je fenomen homofema: različiti glasovi (na primer /p/, /b/, /m/) izgledaju gotovo identično na usnama. Zbog toga jedan kadar nije dovoljan — neophodan je vremenski kontekst da bi se razrešila višeznačnost. Drugi izazov je zavisnost od velikih količina označenih podataka: raniji pristupi zahtevali su desetine hiljada sati transkribovanih snimaka, što je skupo i nedostupno za većinu od oko 7000 svetskih jezika."));
children.push(P("AV-HuBERT rešava taj drugi problem samonadgledanim učenjem audio-vizuelnih reprezentacija: iz velike količine NEoznačenih snimaka uči opšte reprezentacije govora, a zatim se uz vrlo malo označenih podataka prilagođava konkretnom zadatku. U originalnom radu, sa svega 30 sati označenih podataka postiže se bolji rezultat (WER 32,5%) nego kod ranijih modela obučenih na 31.000 sati — dakle uz oko 1000× manje označenih podataka."));
children.push(runs([bold("Cilj ovog rada"), txt(" jeste: (1) analizirati arhitekturu i ciljnu funkciju AV-HuBERT modela, (2) reprodukovati proceduru fine-tuninga na podskupu baze podataka radi provere da kod ispravno radi i uvida u samu implementaciju, i (3) pripremiti postupak prilagođavanja modela na srpski jezik (korpus AI-SPEAK), pri čemu se za prilagođavanje razmatra parametarski efikasna LoRA adaptacija.")]));

// 2. METODE I MATERIJALI
children.push(H1("2. Metode i materijali"));
children.push(H2("2.1 Model — arhitektura AV-HuBERT"));
children.push(P("AV-HuBERT je proširenje HuBERT okvira (samonadgledano učenje reprezentacija govora) na dva modaliteta: niz slika regiona usana (video) i akustična obeležja (audio). Model se sastoji iz laganih ulaznih ekstraktora obeležja, fuzije, i deljenog Transformer enkodera."));
children.push(bulletRuns([bold("Video ekstraktor: "), txt("modifikovani ResNet-18 nad regionom usana (ROI) dimenzija 96×96. Ulaz prolazi kroz 3D konvoluciju, a zatim kroz 2D ResNet. Ima oko 11,6 miliona parametara.")]));
children.push(bulletRuns([bold("Audio ekstraktor: "), txt("jedan linearni sloj nad stakovanim log-mel obeležjima (dimenzija 104 = 26 log-mel × 4 kadra). Ima svega oko 80 hiljada parametara — namerno mali kapacitet, kako audio ne bi dominirao nad vizuelnim modalitetom.")]));
children.push(bulletRuns([bold("Fuzija: "), txt("konkatenacija po kanalima dva niza obeležja, potom linearna projekcija u dimenziju modela (768).")]));
children.push(bulletRuns([bold("Transformer enkoder: "), txt("deljeni enkoder nad fuzionisanom reprezentacijom. BASE konfiguracija: 12 blokova, 768 dimenzija, 12 glava pažnje — ukupno oko 103 miliona parametara. LARGE (24 bloka, 1024 dim, ~325M) nije korišćena zbog ograničenja resursa.")]));
children.push(bulletRuns([bold("Modality dropout: "), txt("tokom pretreninga se nasumično izostavlja ceo audio ILI video tok (verovatnoća 0,5), čime se sprečava oslanjanje samo na jedan modalitet i jača vizuelni enkoder.")]));
children.push(runs([it("Napomena: navedene brojke parametara su potvrđene direktnim učitavanjem BASE checkpoint-a (base_vox_iter5.pt).")]));

children.push(H2("2.2 Obuka i ciljna funkcija"));
children.push(runs([bold("Faza 1 — Pretrening (samonadgledano; nije rađeno u ovom radu). "), txt("Prvo se k-means klasterizacijom obeležja svakom kadru dodeljuje diskretni ID klastera (pseudo-oznaka). Klasteri se iterativno preračunavaju iz sve boljih obeležja modela (5 iteracija). Zatim se deo kadrova maskira, a model predviđa ID klastera maskiranih kadrova (BERT-oliki zadatak). Ciljna funkcija maskirane predikcije klastera je:")]));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 80 },
  children: [new TextRun({ text: "L = α · Σ_{t∈M} CE(p_t, z_t)  +  (1−α) · Σ_{t∉M} CE(p_t, z_t)", italics: true, size: 22 })] }));
children.push(P("gde je M skup maskiranih kadrova, z_t ID klastera kadra t, a p_t predviđena raspodela nad K klastera. Logiti se računaju kosinusnom sličnošću između izlaza enkodera i naučenih klaster-embedinga. U praksi je težina nemaskiranih kadrova obično nula (predviđaju se samo maskirani). Korišćen je BASE checkpoint pretreniran na LRS3 + VoxCeleb2."));
children.push(runs([bold("Faza 2 — Fine-tuning (nadgledano; rađeno u ovom radu). "), txt("Na pretrenirani enkoder dodaje se Transformer dekoder (seq2seq, tip modela av_hubert_seq2seq, 6 slojeva). Dekoder autoregresivno predviđa subword tokene transkripta (SentencePiece vokabular napravljen iz transkripata). Ciljna funkcija je label-smoothed cross-entropy (label_smoothing = 0,1). Enkoder je zamrznut prvih N koraka (parametar freeze_finetune_updates), pa se odmrzava.")]));

children.push(H2("2.3 LoRA adaptacija"));
children.push(P("Za prilagođavanje na malu količinu podataka razmatra se LoRA (Low-Rank Adaptation): pretrenirani slojevi se zamrznu, a u linearne slojeve se ubacuju male niskorangovane matrice A i B, tako da je izlaz W·x + (α/r)·B(A·x). Obučavaju se samo A i B (rang r je mali, npr. 8), dok W ostaje zamrznuto. Time se broj obučavanih parametara drastično smanjuje, čime se smanjuje i rizik od preprilagođavanja (overfitting) na malim skupovima. U našoj postavci, LoRA na dekoderu (q_proj, v_proj, r = 8) obučava svega 0,18% parametara."));

children.push(H2("2.4 Podaci"));
children.push(P("Korišćena je baza LRS3-TED (rečenice iz TED/TEDx govora, engleski jezik) — ista baza na kojoj se AV-HuBERT ocenjuje u originalnom radu. Pošto zvanična distribucija LRS3 više nije dostupna, korišćen je proveren javni mirror (HuggingFace: TheNHz/ellipsis-lrs3-raw, licenca CC BY 4.0). Zbog ograničenih resursa i cilja rada (provera ispravnosti koda, a ne potpuna reprodukcija), korišćen je podskup:"));
children.push(BULLET("Obuka (train): 3.000 klipova; validacija: 200 klipova (iz trainval podskupa)."));
children.push(BULLET("Test: kompletan zvanični test skup — 1.321 iskaz."));
children.push(P("Pretprocesiranje: detekcija lica i 68 tačaka (dlib) → isecanje ROI usana na 96×96 → audio na 16 kHz → manifesti (.tsv), transkripti (.wrd) i SentencePiece vokabular veličine 1000."));
children.push(runs([bold("AI-SPEAK korpus (srpski): "), it("[POPUNITI — jezik, broj sati/iskaza, format, uslovi snimanja kada podaci budu dostupni].")]));
children.push(P("Napomena o transferu: vizuelni/audio enkoder je pretreniran na engleskom (LRS3+VoxCeleb2); pri prelasku na srpski, enkoder se prenosi, ali se dekoder i vokabular obučavaju iznova — što je i motivacija za LoRA adaptaciju."));

children.push(H2("2.5 Eksperimentalna postavka"));
children.push(BULLET("Hardver: iznajmljen GPU NVIDIA RTX A6000 (48 GB) na platformi RunPod, za eksperimente na LRS3."));
children.push(BULLET("Konfiguracija (jedan GPU): fp16, max_tokens = 1500, update_freq = 2, max_update = 3000, lr = 1e-3, tri_stage raspored."));
children.push(bulletRuns([txt("Verzija koda: av_hubert "), new TextRun({ text: "258fb50", font: "Consolas", size: 18 }), txt(", fairseq "), new TextRun({ text: "afc77bd", font: "Consolas", size: 18 }), txt(" (detalji u uputstvu).")]));
children.push(BULLET("Metrika: WER (Word Error Rate), beam-search dekodiranje (beam = 50), modalitet: samo video (VSR)."));

// 3. REZULTATI
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("3. Rezultati i diskusija"));
children.push(H2("3.1 Rezultati na LRS3 podskupu"));
children.push(P("Sprovedena su dva eksperimenta pod istim uslovima (isti podskup, 3.000 koraka, samo video): puni fine-tuning i LoRA adaptacija dekodera. Rezultati na test skupu (1.321 iskaz):"));
children.push(table(
  ["Eksperiment", "Obučavani parametri", "Tačnost (train)", "WER (test)"],
  [
    ["Puni fine-tuning", "161M (100%)", "86,8%", "74,4%"],
    ["LoRA (dekoder q/v)", "294.912 (0,18%)", "16,5%", "93,6%"],
  ],
  [2600, 2760, 2000, 2000]
));
children.push(runs([bold("Primer predikcije "), txt("(puni fine-tuning; REF = tačan transkript, HYP = predikcija modela iz nemog videa):")]));
children.push(...code([
  "REF: we were wrong",
  "HYP: what we were wrong",
  "",
  "REF: it turned out that we were doing a lot of low level drug cases ...",
  "HYP: it turns out out that we were a lot of our lot of ...",
]));
children.push(P("Iako WER ostaje visok, model uspešno prepoznaje reči isključivo iz pokreta usana (u prvom primeru 3 od 4 reči tačno), što potvrđuje da ceo lanac — pretprocesiranje, pretrenirani enkoder, seq2seq fine-tuning, beam-search dekodiranje i računanje WER-a — radi ispravno na stvarnim podacima."));

children.push(H2("3.2 Diskusija"));
children.push(P("Dobijeni WER (74,4%) je znatno viši od rezultata iz originalnog rada (26–32%). Razlike su očekivane i posledica su namerno smanjene postavke, u skladu sa ciljem rada:"));
children.push(BULLET("Količina podataka: 3.000 klipova (oko 2–3 sata) naspram 433 sata u originalu — oko 150× manje."));
children.push(BULLET("Dužina obuke: 3.000 koraka naspram 30.000 — 10× kraće."));
children.push(BULLET("Cilj: potvrda ispravnosti koda i uvid u model, a ne optimizacija WER-a."));
children.push(P("Poređenje punog fine-tuninga i LoRA je poučno. Na ovom malom skupu i na istom (engleskom) jeziku, puni fine-tuning postiže bolji WER jer njegov veći kapacitet i delimično odmrzavanje enkodera omogućavaju bolje uklapanje u podatke (visoka train tačnost 86,8% delom je i preprilagođavanje). LoRA, sa svega 0,18% obučavanih parametara, ne može da «zapamti» mali skup na isti način (train tačnost 16,5%), pa je na ovom scenariju slabija. Međutim, prava prednost LoRA se očekuje u scenariju sa malo podataka i promenom jezika (AI-SPEAK, srpski), gde bi puni fine-tuning lako preprilagodio model. Time je i glavna svrha LoRA eksperimenta na LRS3 bila da se validira ispravnost LoRA implementacije pre prelaska na srpski — što je uspešno urađeno (tokom validacije je otkrivena i ispravljena greška u ophođenju sa atributima linearnog sloja)."));
children.push(P("Šta bi poboljšalo rezultat: više podataka i duža obuka, LARGE konfiguracija modela, kao i AVSR postavka (audio + video zajedno) za robusnost."));

// 4. ZAKLJUCAK
children.push(H1("4. Zaključak"));
children.push(P("U radu je analiziran AV-HuBERT model za vizuelno prepoznavanje govora i reprodukovana je procedura fine-tuninga na podskupu baze LRS3. Uspešno je pokazan ceo tok rada: od pretprocesiranja sirovih snimaka, preko korišćenja pretreniranog samonadgledanog enkodera i obuke seq2seq dekodera, do dekodiranja i merenja WER-a. Dobijen je konkretan, ponovljiv rezultat (WER 74,4% pri punom fine-tuningu, 93,6% pri LoRA adaptaciji dekodera na malom podskupu), uz jasno objašnjenje razlika u odnosu na originalni rad."));
children.push(P("Najvažniji aspekti urađenog: (1) potvrđeno je da samonadgledani pretrening uz relativno mali fine-tuning daje upotrebljiv sistem za čitanje sa usana; (2) implementirana je i validirana LoRA adaptacija unutar fairseq okvira, spremna za prilagođavanje na srpski jezik; (3) pripremljene su sve skripte i konfiguracije za obuku i testiranje. Naredni korak je prilagođavanje na korpus AI-SPEAK, primenom LoRA adaptacije prvo na dekoderu, a zatim i na enkoderu."));

// Reference
children.push(H1("Reference"));
children.push(REF("Shi, B., Hsu, W.-N., Lakhotia, K., Mohamed, A. Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction. arXiv:2201.02184 (2022)."));
children.push(REF("Shi, B., Hsu, W.-N., Mohamed, A. Robust Self-Supervised Audio-Visual Speech Recognition. arXiv:2201.01763 (2022)."));
children.push(REF("Hsu, W.-N. i dr. HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. (2021)."));
children.push(REF("Afouras, T., Chung, J. S., Zisserman, A. LRS3-TED: a large-scale dataset for visual speech recognition. arXiv:1809.00496 (2018)."));
children.push(REF("Hu, E. i dr. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685 (2021)."));

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: "1F4E79" }, paragraph: { spacing: { before: 300, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: FONT, color: "2E75B6" }, paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 620, hanging: 300 } } } }] },
    { reference: "n", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 620, hanging: 300 } } } }] },
    { reference: "r", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "[%1]", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 620, hanging: 400 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync("deliverables/Izvestaj_AV-HuBERT.docx", buf); console.log("wrote deliverables/Izvestaj_AV-HuBERT.docx"); });
