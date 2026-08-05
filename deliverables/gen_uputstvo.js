const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
  TableOfContents, PageBreak,
} = require("docx");

const FONT = "Arial";
const CW = 9360; // content width, US Letter 1" margins

// ---- helpers ----
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const P = (t) => new Paragraph({ spacing: { after: 120 }, children: [new TextRun(t)] });
const runs = (arr) => new Paragraph({ spacing: { after: 120 }, children: arr });
const bold = (t) => new TextRun({ text: t, bold: true });
const txt = (t) => new TextRun(t);
const mono = (t) => new TextRun({ text: t, font: "Consolas", size: 20 });
const BULLET = (t) => new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 60 }, children: [new TextRun(t)] });
const bulletRuns = (arr) => new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 60 }, children: arr });
const NUM = (ref, t) => new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 60 }, children: [new TextRun(t)] });

// code block: shaded monospace paragraph(s)
const code = (lines) => lines.map((ln, i) =>
  new Paragraph({
    shading: { fill: "F2F2F2", type: ShadingType.CLEAR },
    spacing: { after: i === lines.length - 1 ? 120 : 0 },
    children: [new TextRun({ text: ln || " ", font: "Consolas", size: 18 })],
  })
);

const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths) {
  const mk = (cells, head) => new TableRow({
    children: cells.map((c, i) => new TableCell({
      borders, width: { size: widths[i], type: WidthType.DXA },
      shading: head ? { fill: "2E75B6", type: ShadingType.CLEAR } : { fill: "FFFFFF", type: ShadingType.CLEAR },
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({ children: [new TextRun({ text: c, bold: head, color: head ? "FFFFFF" : "000000", size: 20 })] })],
    })),
  });
  return new Table({
    width: { size: CW, type: WidthType.DXA }, columnWidths: widths,
    rows: [mk(headers, true), ...rows.map((r) => mk(r, false))],
  });
}

const children = [];

// Title
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1200, after: 120 },
  children: [new TextRun({ text: "Uputstvo za upotrebu koda", bold: true, size: 44 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "Vizuelno prepoznavanje govora pomoću AV-HuBERT modela", size: 28 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
  children: [new TextRun({ text: "Marko Mrđa · Mašinsko učenje · FTN, Univerzitet u Novom Sadu", size: 22, color: "666666" })] }));

children.push(new Paragraph({ children: [new TextRun({ text: "Sadržaj", bold: true, size: 28 })] }));
children.push(new TableOfContents("Sadržaj", { hyperlink: true, headingStyleRange: "1-3" }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// 1. Uvod
children.push(H1("1. Uvod i verzija koda"));
children.push(P("Ovo uputstvo opisuje kako se koristi kod za obuku (fine-tuning) i testiranje AV-HuBERT modela za vizuelno prepoznavanje govora (VSR), i povezuje arhitekturu modela (predstavljenu u prezentaciji) sa konkretnim klasama i funkcijama u kodu. Opisuje i modifikaciju koda koju smo uveli — LoRA adaptaciju."));
children.push(runs([bold("Korišćena verzija koda (obavezno navesti):")]));
children.push(BULLET("Zvanični repozitorijum: github.com/facebookresearch/av_hubert"));
children.push(bulletRuns([txt("av_hubert commit: "), mono("258fb50e155134eec2c4b49c2ae8de267075fd18")]));
children.push(bulletRuns([txt("fairseq (podmodul) commit: "), mono("afc77bdf4bb51453ce76f1572ef2ee6ddcda8eeb")]));
children.push(P("fairseq je uključen kao git podmodul (submodule) unutar av_hubert repozitorijuma, zaključan na tačno određenu verziju iz juna 2021. AV-HuBERT nije samostalan program nego proširenje fairseq radnog okvira: autori su dodali model, ciljnu funkciju, klasu skupa podataka i «task», a fairseq obezbeđuje petlju obuke, rad sa više GPU-a, učitavanje podataka, optimizatore i čuvanje checkpoint-a."));

// 2. Okruzenje
children.push(H1("2. Priprema okruženja"));
children.push(P("AV-HuBERT zavisi od stare verzije fairseq-a, pa su verzije biblioteka osetljive. Koristi se conda okruženje sa Python 3.9."));
children.push(H2("2.1 Kreiranje okruženja"));
children.push(...code([
  "conda create -n avhubert --override-channels -c conda-forge python=3.9 -y",
  "conda activate avhubert",
  "# PyTorch 1.13.1 (na Linux/CUDA mašini: +cu117 build):",
  "pip install numpy==1.23.5 torch==1.13.1 torchaudio==0.13.1 torchvision==0.14.1",
  "# ostale zavisnosti:",
  "pip install opencv-python==4.9.0.80 sentencepiece editdistance \\",
  "    scikit-image==0.19.3 python_speech_features pydub tqdm 'Cython<3' 'setuptools<70'",
  "pip install dlib scikit-video pyarrow   # za pretprocesiranje",
]));
children.push(H2("2.2 Instalacija fairseq-a"));
children.push(...code([
  "cd av_hubert/fairseq",
  "pip install --no-build-isolation -e .",
  "# fairseq povlači prestare omegaconf/hydra (ImportError: II) -> zakucati verzije:",
  "pip install 'pip<24.1'",
  "pip install omegaconf==2.0.6 hydra-core==1.0.7",
]));
children.push(runs([bold("Napomena o GPU-u: "), txt("torch 1.13.1 (CUDA 11.7) radi na modernim GPU-ovima (npr. RTX A6000, sm_86) jer su NVIDIA drajveri unazad kompatibilni. Ovo je provereno pokretanjem stvarne operacije na GPU-u tokom postavljanja.")]));

// 3. Pretprocesiranje
children.push(H1("3. Pretprocesiranje podataka"));
children.push(P("Model ne čita sirov video direktno. Sirovi LRS3 klipovi (mp4 sa licem + .txt transkript) se pretvaraju u format koji AV-HuBERT očekuje. Koraci (skripta preprocess_lrs3_subset.sh ih objedinjuje):"));
children.push(NUM("s", "Izbor podskupa i pravljenje file.list / label.list (lrs3_make_subset.py)."));
children.push(NUM("s", "Detekcija 68 tačaka lica pomoću dlib biblioteke (preparation/detect_landmark.py)."));
children.push(NUM("s", "Isecanje regiona usana (ROI) na 96×96 (preparation/align_mouth.py)."));
children.push(NUM("s", "Izdvajanje zvuka na 16 kHz (ffmpeg) i brojanje kadrova (preparation/count_frames.py)."));
children.push(NUM("s", "Test skup LRS3 je u parquet formatu (već isečeni ROI 96×96) — konvertuje se skriptom lrs3_test_from_parquet.py."));
children.push(NUM("s", "Pravljenje manifesta i rečnika (lrs3_build_manifest.py): {train,valid,test}.tsv, {train,valid,test}.wrd, dict.wrd.txt i SentencePiece model."));
children.push(runs([bold("Format .tsv manifesta"), txt(" (po redu, tab-razdvojeno): ")]));
children.push(...code([
  "<id>  <apsolutna_putanja_video.mp4>  <apsolutna_putanja_audio.wav>  <br_video_kadrova>  <br_audio_uzoraka>",
]));
children.push(runs([bold("Format .wrd: "), txt("transkript (mala slova) po redu, poravnat sa .tsv. dict.wrd.txt je fairseq rečnik napravljen iz SentencePiece modela.")]));

// 4. Obuka
children.push(H1("4. Obuka (fine-tuning)"));
children.push(P("Obuka se pokreće alatom fairseq-hydra-train, koji čita YAML konfiguraciju iz foldera conf/ (Hydra sistem). Bilo koja vrednost se može pregaziti u komandnoj liniji."));
children.push(H2("4.1 Puni fine-tuning (dekoder + enkoder)"));
children.push(...code([
  "cd av_hubert/avhubert",
  "fairseq-hydra-train --config-dir workspace/configs --config-name lrs2_base_vsr_runpod \\",
  "  task.data=$DATA task.label_dir=$DATA \\",
  "  task.tokenizer_bpe_model=$DATA/spm_unigram1000.model \\",
  "  model.w2v_path=$CKPT \\",
  "  common.user_dir=$(pwd) \\",
  "  optimization.max_update=3000 model.freeze_finetune_updates=2000 \\",
  "  hydra.run.dir=$EXP",
]));
children.push(runs([bold("model.w2v_path"), txt(" pokazuje na pretrenirani checkpoint (base_vox_iter5.pt, LRS3+VoxCeleb2). "), bold("freeze_finetune_updates=2000"), txt(" znači da je enkoder zamrznut prvih 2000 koraka (obučava se samo dekoder), a nakon toga se odmrzava i obučavaju se i enkoder i dekoder.")]));
children.push(H2("4.2 LoRA fine-tuning (parametarski efikasno)"));
children.push(P("LoRA (Low-Rank Adaptation) zamrzava ceo pretrenirani model i obučava samo male niskorangovane matrice ubačene u linearne slojeve. Naša implementacija je konfiguraciona (bez izmene komandi):"));
children.push(...code([
  "fairseq-hydra-train --config-dir workspace/configs --config-name lrs3_base_vsr_lora \\",
  "  task.data=$DATA task.label_dir=$DATA \\",
  "  task.tokenizer_bpe_model=$DATA/spm_unigram1000.model \\",
  "  model.w2v_path=$CKPT common.user_dir=$(pwd) hydra.run.dir=$EXP",
  "# Ključne LoRA opcije u konfiguraciji:",
  "#   model.lora_enable: true",
  "#   model.lora_scope: decoder   (ili 'encoder' ili 'all')",
  "#   model.lora_r: 8   model.lora_alpha: 16   model.lora_targets: 'q_proj,v_proj'",
]));

// 5. Testiranje
children.push(H1("5. Testiranje (inference i WER)"));
children.push(P("Dekodiranje test skupa i računanje WER-a se radi skriptom infer_s2s.py (seq2seq beam-search dekodiranje):"));
children.push(...code([
  "cd av_hubert/avhubert",
  "python infer_s2s.py --config-dir ./conf --config-name s2s_decode \\",
  "  common.user_dir=$(pwd) dataset.gen_subset=test \\",
  "  override.modalities='[video]' \\",
  "  override.data=$DATA override.label_dir=$DATA \\",
  "  common_eval.path=$FT_CKPT common_eval.results_path=$RES",
  "# WER se ispisuje u $RES/decode.log ; hipoteze u hypo-*.json",
]));

// 6. Povezivanje arhitekture i koda
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("6. Povezivanje arhitekture i koda"));
children.push(P("Sve putanje su relativne u odnosu na av_hubert/avhubert/. Sledeća tabela povezuje koncepte iz prezentacije sa konkretnim klasama/funkcijama u ovoj verziji koda."));
children.push(H2("6.1 Model, dve «glave» (pretrening vs fine-tuning)"));
children.push(table(
  ["Faza", "Klasa modela", "Ciljna funkcija", " Meta / cilj"],
  [
    ["Pretrening", "AVHubertModel (hubert.py)", "AVHubertCriterion (hubert_criterion.py)", "ID-jevi klastera (.km)"],
    ["Fine-tuning (VSR)", "AVHubertSeq2Seq (hubert_asr.py)", "label_smoothed_cross_entropy", "tokeni transkripta (.wrd)"],
  ],
  [1500, 2900, 2960, 2000]
));
children.push(P("Pretrening (klasterizacija + maskirana predikcija) je urađen unapred — mi koristimo gotov checkpoint. Fine-tuning «obmotava» pretrenirani AVHubertModel kao enkoder i dodaje Transformer dekoder (decoder.py)."));

children.push(H2("6.2 Mapiranje po slajdovima prezentacije"));
children.push(table(
  ["Koncept sa slajda", "Fajl / funkcija"],
  [
    ["Video frontend (modifikovani ResNet-18)", "resnet.py (ResEncoder), instanciran u hubert.py"],
    ["Audio frontend (jedan linearni sloj)", "hubert.py — Linear(104→768), ~80K param."],
    ["Fuzija konkatenacijom po kanalima (f^av)", "hubert.py, AVHubertModel.forward_features (modality_fuse='concat')"],
    ["Transformer enkoder", "hubert.py (TransformerEncoder), 12 blokova × 768 (BASE)"],
    ["Modality dropout", "hubert.py (modality_dropout=0.5, audio_dropout=0.5)"],
    ["Ciljna funkcija (maskirana predikcija klastera)", "hubert_criterion.py, AVHubertCriterion"],
    ["Seq2seq dekoder (fine-tuning)", "hubert_asr.py AVHubertSeq2Seq + decoder.py TransformerDecoder"],
    ["CTC varijanta (nije korišćena)", "hubert_asr.py AVHubertCtc"],
    ["Zamrzavanje enkodera na početku", "hubert_asr.py forward() + freeze_finetune_updates"],
    ["fairseq «task» (spaja sve)", "hubert_pretraining.py (av_hubert_pretraining)"],
  ],
  [3400, 5960]
));

children.push(H2("6.3 Ciljna funkcija u kodu"));
children.push(P("Ciljna funkcija pretreninga je unakrsna entropija (cross-entropy) nad klasterima, podeljena na maskirane i nemaskirane kadrove — tačno kao izraz sa slajda. U hubert_criterion.py:"));
children.push(...code([
  "loss_m = F.cross_entropy(logp_m, targ_m)   # nad MASKIRANIM kadrovima",
  "loss_u = F.cross_entropy(logp_u, targ_u)   # nad NEMASKIRANIM kadrovima",
  "loss = pred_masked_weight * sum(loss_m) + pred_nomask_weight * sum(loss_u)",
]));
children.push(P("targ_m je ID klastera svakog maskiranog kadra; logp_m je predviđena raspodela modela (logiti se računaju kosinusnom sličnošću u odnosu na naučene klaster-embedinge). Fine-tuning umesto toga koristi label-smoothed cross-entropy nad tokenima transkripta."));

children.push(H2("6.4 Potvrđeni parametri (učitavanjem BASE checkpoint-a)"));
children.push(table(
  ["Modul", "Klasa", "Parametri"],
  [
    ["feature_extractor_video", "ResEncoder (Conv3D + 2D ResNet)", "11.6M"],
    ["feature_extractor_audio", "Linear 104→768", "80.6K"],
    ["encoder", "TransformerEncoder (12×768)", "89.8M"],
    ["UKUPNO (samo enkoder)", "AVHubertModel", "103.3M"],
    ["seq2seq (enkoder + dekoder)", "AVHubertSeq2Seq", "~161M"],
  ],
  [3600, 4160, 1600]
));

// 7. Modifikacija koda - LoRA
children.push(H1("7. Naša modifikacija koda: LoRA"));
children.push(P("Pošto fairseq iz 2021. nema LoRA, a biblioteka HuggingFace peft povlači torch 2.x (što bi pokvarilo okruženje), implementirali smo LoRA ručno unutar hubert_asr.py. Izmene:"));
children.push(BULLET("Klasa LoRALinear — obmotava zamrznuti nn.Linear i dodaje niskorangovanu ispravku y = Wx + (α/r)·B(Ax). Matrica B je inicijalizovana na nulu, pa je adapter na početku identitet (stabilan start). Dodate su i property-je weight/bias/in_features/out_features koje delegiraju na osnovni sloj (fairseq introspektuje ove atribute)."));
children.push(BULLET("Funkcija _apply_lora(model, cfg) — zamenjuje ciljne linearne slojeve LoRALinear-om i zamrzava sve ostalo. Podržava opseg 'decoder', 'encoder' ili 'all'."));
children.push(BULLET("Konfiguraciona polja u AVHubertSeq2SeqConfig: lora_enable, lora_r, lora_alpha, lora_dropout, lora_targets, lora_scope. Poziv _apply_lora je na kraju AVHubertSeq2Seq.build_model, uslovljen sa lora_enable."));
children.push(runs([bold("Efekat: "), txt("pri lora_scope=decoder, targets=q_proj,v_proj, r=8 — obučava se svega 294.912 od 160.9M parametara (0.18%). Ostatak modela je zamrznut. Adapter se čuva kao mali dodatak, a ne kao ceo model.")]));

// 8. Struktura projekta
children.push(H1("8. Struktura projekta (prateći kod)"));
children.push(...code([
  "av_hubert/                  zvanični repo (model + fairseq podmodul); modifikovan hubert_asr.py",
  "workspace/",
  "  configs/                  konfiguracije obuke:",
  "    lrs2_base_vsr_runpod.yaml   puni fine-tuning (24GB GPU)",
  "    lrs3_base_vsr_lora.yaml     LoRA fine-tuning",
  "    lrs2_base_vsr_1gpu.yaml     rezerva za 4GB GPU",
  "  scripts/                  skripte:",
  "    lrs3_make_subset.py, lrs3_test_from_parquet.py, lrs3_build_manifest.py",
  "    preprocess_lrs3_subset.sh, download_lrs3.sh, download_checkpoint.sh",
  "    setup_runpod.sh, sync_to_runpod.sh, inspect_model.py, lora.py",
  "  checkpoints/              pretrenirani BASE + dlib modeli",
  "  experiments/              rezultati obuke (checkpoint, log, dekodiranje)",
  "data/lrs3_raw/              LRS3 podaci (trainval, test, video ROI, audio, manifesti)",
]));

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: "1F4E79" },
        paragraph: { spacing: { before: 300, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: FONT, color: "2E75B6" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: FONT },
        paragraph: { spacing: { before: 140, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 620, hanging: 300 } } } }] },
      { reference: "s", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 620, hanging: 300 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("deliverables/Uputstvo_AV-HuBERT.docx", buf);
  console.log("wrote deliverables/Uputstvo_AV-HuBERT.docx");
});
