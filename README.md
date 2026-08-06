# GAVE2 — LaTIM submission materials

Code for our submission to the **GAVE2 Challenge** (MICCAI 2026, OMIA Workshop):
retinal artery/vein segmentation (Tasks 1–2) and biomarker quantification
(Task 3).

Built on the official GAVE2 baseline (RRWNet / CMRRWNet), with two bug fixes in
the provided code and a post-processing chain consisting of threshold
calibration, decision-level fusion, and endpoint reconnection.

The **model weights, configuration, final predictions and technical report** are
available in the [Releases](../../releases) section.

## Requirements

See `requirements.txt`. Python 3.10, PyTorch 2.2 (CUDA 12.1).

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Files

| File | Contents |
|---|---|
| `model.py` | RRWNet and CMRRWNet, including our topological-anchor fix |
| `preprocessing.py` | image loading, padding, channel conventions |
| `transformations.py` | D4 test-time augmentation |
| `postprocessing.py` | threshold calibration, fusion, density matching, endpoint reconnection |
| `biomarkers.py` | Task 3 measurements and per-biomarker threshold assembly |
| `infer.py` | checkpoint ensemble + TTA inference |
| `final.py` | end-to-end submission assembly, verification and packaging |
| `replay_task3.py` | reconstructs the exact Task 3 files of the scored submission |
| `pipeline.yaml` | every threshold and fusion weight used |
| `Dockerfile` | inference image (Tasks 1–2 end-to-end; Task 3 given disc masks) |
| `run_pipeline.sh` | one-command end-to-end run inside the container |

## Docker

```bash
docker build -t latim/gave2:1.0.0 .
docker run --gpus all \
    -v $PWD/data:/data -v $PWD/weights:/weights -v $PWD/work:/work \
    latim/gave2:1.0.0 bash run_pipeline.sh /data /weights /work
```

Optic-disc segmentation is **not** in this image: it requires TensorFlow 2.13,
which does not coexist cleanly with PyTorch 2.2. Precomputed disc masks for the
validation set are included in the Release, so Task 3 runs from the container as
long as they are mounted; the TensorFlow script is provided separately for use
on other data.

## Inference

```bash
# Task 1 probabilities (two model variants)
python infer.py --task task1 --checkpoints weights/task1_gave2/fold{0,1,2,3}.pth \
    --images-path data/images --masks-path data/masks --save-path work/p_t1_gave2
python infer.py --task task1 --checkpoints weights/task1_hrf/fold{0,1,2,3}.pth \
    --images-path data/images --masks-path data/masks --save-path work/p_t1_hrf

# Task 2 probabilities (CFP + FFA)
python infer.py --task task2 --checkpoints weights/task2/fold{0,1,2,3}.pth \
    --images-path data/images --masks-path data/masks \
    --a-path data/FFA_A --av-path data/FFA_AV --save-path work/p_t2

# Assemble the submission
python final.py --prob-task1-gave2 work/p_t1_gave2 --prob-task1-hrf work/p_t1_hrf \
    --prob-task2 work/p_t2 --disc-dir work/optic_disc \
    --out work/submission --zip work/submission.zip
```

Optic-disc masks (needed by Task 3) are produced by a separate TensorFlow-based
step; precomputed masks for the validation set are included in the Release.

## Reproducing the scored submission exactly

`final.py` implements the method as described in the technical report: every
Task 3 biomarker computed from one probability field, each at its own threshold.

The Task 3 files we actually submitted were instead assembled biomarker by
biomarker across several earlier submissions, keeping for each the source that
scored best on the validation leaderboard. To reproduce those exact files from
the archived per-submission outputs (included in the Release):

```bash
python replay_task3.py --archive archive/task3 --out work/submission/Task3
```

The provenance table and the reasoning are documented in `replay_task3.py`, and
this selection procedure is described in the technical report. Tasks 1 and 2 are
produced identically by both routes.

## Note on channel conventions

The original RRWNet uses `R = artery, G = vein, B = vessel`, whereas GAVE2 uses
`R = artery, G = vessel, B = vein`. The GAVE2 baseline migrated its loss
function to the new order but not its model code, which continued to anchor
channel index 2 — "vessel" upstream, but "vein" under GAVE2. The evidence is a
self-contradiction inside the baseline's own `losses.py`, whose docstring states
the upstream order while its body implements the GAVE2 one.

Correcting this is the first of our two bug fixes; see the technical report.

## Citation

```bibtex
@article{morano2024rrwnet,
  title   = {RRWNet: Recursive Refinement Network for Effective Retinal
             Artery/Vein Segmentation and Classification},
  author  = {Morano, Jos{\'e} and Aresta, Guilherme and Bogunovi{\'c}, Hrvoje},
  journal = {Expert Systems with Applications},
  volume  = {256},
  pages   = {124970},
  year    = {2024},
  doi     = {10.1016/j.eswa.2024.124970}
}

@article{hemelings2019artery,
  title   = {Artery--vein segmentation in fundus images using a fully
             convolutional network},
  author  = {Hemelings, Ruben and Elen, Bart and Stalmans, Ingeborg and
             Van Keer, Karel and De Boever, Patrick and Blaschko, Matthew B.},
  journal = {Computerized Medical Imaging and Graphics},
  volume  = {76},
  pages   = {101636},
  year    = {2019}
}
```

Official GAVE2 baseline: <https://github.com/Peng2004/CMRRWNet>
