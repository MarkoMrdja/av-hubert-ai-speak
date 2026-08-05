# Is this the RIGHT LRS2? — a checklist

You've already hit two wrong variants (a separation "rebuild", a visual-TTS feature
dump). Before spending effort, verify any LRS2 copy passes ALL of these.

## The one test that matters: **transcripts exist as text**

The official LRS2 has, next to every video, a `.txt` transcript whose first line is:

```
Text:  THE ACTUAL SPOKEN WORDS
```

If you can't find `Text:` lines, it's the wrong dataset — stop.

## Expected structure

```
<lrs2_root>/
  mvlrs_v1/
    main/<videoID>/<clipID>.mp4        # video
    main/<videoID>/<clipID>.txt        # transcript (has "Text:" line)
    pretrain/<videoID>/<clipID>.mp4    # longer clips (optional for us)
    pretrain/<videoID>/<clipID>.txt
  train.txt     # ~45,839 utterance ids
  val.txt       # ~1,082
  test.txt      # ~1,243
  pretrain.txt  # ~96,318  (we usually skip this — too big for a subset run)
```

Split-list lines look like `6300000000000000000/00001` (a `<videoID>/<clipID>` id).

## Quick verification commands

```bash
LRS2=/path/to/lrs2

# 1. media folder present?
ls "$LRS2/mvlrs_v1"                       # expect: main  (and maybe pretrain)

# 2. videos are mp4?
find "$LRS2/mvlrs_v1/main" -name '*.mp4' | head

# 3. THE test — transcripts with a Text: line?
find "$LRS2/mvlrs_v1/main" -name '*.txt' | head -1 | xargs head -3
# expect first line: "Text:  SOME WORDS"

# 4. split lists present with sane counts?
wc -l "$LRS2"/{train,val,test}.txt
```

## Red flags (wrong dataset)

- Folders named `wav8k/`, `wav16k/`, `min/`, or files like `mix_2_spk_*` → speech
  **separation** variant. Wrong.
- `.ark` / `.scp` files, `video_ssl_avhubert`, `data.json` with `"task":
  "visual_tts"` → ESPnet feature dump. Wrong (no raw frames).
- Only `.npz` mouth crops and no `.txt` transcripts → the "rebuild" you already
  deleted. Wrong.

## If you only get transcripts + our already-cropped mouths
Possible fallback: the ids in the deleted rebuild matched official LRS2 ids
(`<videoID>_<clipID>`). If your professor provides only the official transcripts,
we could realign them to pre-cropped mouths by id. Ask first — cleanest is the full
official download.
```
