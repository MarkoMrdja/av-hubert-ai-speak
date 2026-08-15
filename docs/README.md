# docs/ — index

Project overview, layout, results, and setup live in the root
[`README.md`](../README.md) (single source of truth). This folder holds the
project documentation:

| File | What it's for |
|---|---|
| [`ARCHITECTURE_TO_CODE.md`](ARCHITECTURE_TO_CODE.md) | Maps every architecture concept from the presentation to the exact file/function in the pinned code (the core of task 4). |
| [`IZVESTAJ_skelet.md`](IZVESTAJ_skelet.md) | Final report skeleton (task 6); Metode section pre-filled, `[POPUNITI]` marks experiment-dependent gaps. |
| [`AV-HuBERT_prezentacija.pdf`](AV-HuBERT_prezentacija.pdf) | The source presentation the architecture map refers to. |

## Environment
Conda env `avhubert` (Python 3.9). Exact pins in
[`../env/requirements-frozen-macos-arm64.txt`](../env/requirements-frozen-macos-arm64.txt).
On a GPU (Linux) box, rebuild with a CUDA torch build but the same version pins.
