# GAVE2 — submission materials

**Authors:** Hasna MRABENT, Mostafa EL HABIB DAHO — LaTIM UMR1101 INSERM, University of Western Brittany, Brest, France.

Code for our submission to the **GAVE2 Challenge** (MICCAI 2026, OMIA Workshop):
retinal artery/vein segmentation (Tasks 1–2) and biomarker quantification
(Task 3).

Built on the official GAVE2 baseline (RRWNet / CMRRWNet), with two bug fixes in
the provided code and a post-processing chain consisting of threshold
calibration, decision-level fusion, and endpoint reconnection.

The **model weights, configuration, final predictions and technical report** are
available in the [Releases](../../releases) section.

**DockerHub:** <https://hub.docker.com/r/hasnamrabet/gave2>

## Results

Final leaderboard: **2nd / 44 teams**, total **8.04062**.

| | Total | Task 1 | Task 2 | Task 3 |
|---|---|---|---|---|
| Ours | **8.04062** | 8.39456 | 8.44125 | 7.46302 |
| Rank | **2nd** / 44 | 3rd | 2nd | **1st** |

Preliminary leaderboard: **6th / 44 teams**, total **7.98544**.

| | Total | Task 1 | Task 2 | Task 3 |
|---|---|---|---|---|
| Ours | **7.98544** | 8.26139 | 8.29600 | 7.53690 |

Score:

```
Total = 0.2 * Task1 + 0.4 * Task2 + 0.4 * Task3

Task_i = 10 * mean_over_classes(
    0.4 * DSC
  + 0.3 * (0.3*Sen + 0.3*Spe + 0.4*Acc)
  + 0.3 * (0.5*(1 - INF) + 0.5*COR)
)
```

> The official metric samples 100 random paths per image without a fixed seed.
> We measured its variance four times by resubmitting byte-identical files: the
> reported score moves by **±0.02–0.05**. Differences below that band are not
> interpretable.

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
| `replay_task3.py` | reconstructs the exact Task 3 files of the scored preliminary submission |
| `pipeline.yaml` | every threshold and fusion weight used |
| `Dockerfile` | inference image (Tasks 1–2 end-to-end; Task 3 given disc masks) |
| `run_pipeline.sh` | one-command end-to-end run inside the container |
| `scripts/02_segment_optic_disc.py` | optic-disc segmentation (TensorFlow 2.13, separate environment — see below) |

## Docker

```bash
# Pull the pre-built image
docker pull hasnamrabet/gave2:1.0.0

docker run --gpus all \
    -v $PWD/data:/data -v $PWD/weights:/weights -v $PWD/work:/work \
    hasnamrabet/gave2:1.0.0 bash run_pipeline.sh /data /weights /work

# Or build it locally
docker build -t hasnamrabet/gave2:1.0.0 .
```
### Setting up `weights/` from the Release

Release assets are flat. Download the
12 `.pth` files and the two `.tar.gz` archives from
[Releases](../../releases/tag/v1.0), then arrange them as:

- `weights/task1_gave2/fold0.pth` ← `task1_gave2_fold0.pth`
- `weights/task1_gave2/fold1.pth` ← `task1_gave2_fold1.pth`
- `weights/task1_gave2/fold2.pth` ← `task1_gave2_fold2.pth`
- `weights/task1_gave2/fold3.pth` ← `task1_gave2_fold3.pth`
- `weights/task1_hrf/fold0.pth` ← `task1_hrf_fold0.pth`
- `weights/task1_hrf/fold1.pth` ← `task1_hrf_fold1.pth`
- `weights/task1_hrf/fold2.pth` ← `task1_hrf_fold2.pth`
- `weights/task1_hrf/fold3.pth` ← `task1_hrf_fold3.pth`
- `weights/task2/fold0.pth` ← `task2_fold0.pth`
- `weights/task2/fold1.pth` ← `task2_fold1.pth`
- `weights/task2/fold2.pth` ← `task2_fold2.pth`
- `weights/task2/fold3.pth` ← `task2_fold3.pth`

Extract `optic_disc_masks.tar.gz` (→ `optic_disc_masks/`) and `task3_archive.tar.gz` (→ `archive/task3/`) directly — they already unpack to the layout `run_pipeline.sh` and `replay_task3.py` expect.

[Release v1.1](../../releases/tag/v1.1) additionally provides
`optic_disc_masks_final.tar.gz`, the disc masks for the 100-image final set
(also → `optic_disc_masks/`, superseding the 50-image v1.0 archive for that
folder), and `submission_final.zip`, the submission sent for the final round.

Optic-disc segmentation is **not** in this image: it requires TensorFlow 2.13,
which does not coexist cleanly with PyTorch 2.2. Precomputed disc masks for the
validation and final sets are included in the Release, so Task 3 runs from the
container as long as they are mounted; `scripts/02_segment_optic_disc.py` is
included in this repo for running on other data — see below.

### Optic-disc segmentation (separate environment)

`scripts/02_segment_optic_disc.py` produces the disc masks Task 3 needs. It
ports the disc detector from
[MNet_DeepCDR](https://github.com/HzFu/MNet_DeepCDR) (Fu et al., see
Citation) into `tf.keras`, preserving the original layer names so the
authors' released `Model_DiscSeg_ORIGA.h5` weights load directly. It needs
TensorFlow, which does not coexist cleanly with the PyTorch environment used
by everything else in this repo, so it runs in its own environment:

```bash
conda create -n gave2-od python=3.10
conda activate gave2-od
pip install tensorflow==2.13.1 scikit-image numpy

python scripts/02_segment_optic_disc.py \
    --images data/validation/images \
    --weights external/MNet_DeepCDR/deep_model/Model_DiscSeg_ORIGA.h5 \
    --out work/optic_disc
```

The weights file is not redistributed here; download it from the
MNet_DeepCDR repository linked above. Output is one binary PNG (0/255) per
input image at the original resolution, ready to pass as `--disc-dir`.

> The masks used for our submitted results are provided in the Release
> (`optic_disc_masks.tar.gz` for the 50 validation images,
> `optic_disc_masks_final.tar.gz` for the 100 final images). When we re-ran
> this script in a fresh environment as a cross-check, the masks differed
> substantially from them (mean IoU 0.12 on the validation images; cause not
> investigated, possibly environment-dependent). For exact reproduction of
> our results, please use the released masks.

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

Optic-disc masks (needed by Task 3) are produced by `scripts/02_segment_optic_disc.py`
(separate TensorFlow environment, see above); precomputed masks for the
validation and final sets are included in the Release.

## Reproducing the submissions

**Final round (100 images).** The final submission was produced end-to-end by
`final.py` (see Inference above, or `run_pipeline.sh` in the container). It is
provided in Release v1.1 as `submission_final.zip`.

**Preliminary round (50 images).** `final.py` implements the method as described
in the technical report: every Task 3 biomarker computed from one probability
field, each at its own threshold. The Task 3 files of our scored preliminary
submission were instead assembled biomarker by biomarker across several earlier
submissions, keeping for each the source that scored best on the validation
leaderboard. To reproduce those exact files from the archived per-submission
outputs (included in Release v1.0):

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
channel index 2 — "vessel" upstream, but "vein" under GAVE2.

The two halves of the defect sit in different files of the official baseline,
and both are checkable:

- **`train/losses.py`** — `BCE3Loss` contradicts itself. Its docstring states the
  upstream order (`artery: 0, vein: 1, vessel_tree: 2`) while its body reads
  `pred_v = pred[:, 2]` and `pred_vt = pred[:, 1]`, i.e. the GAVE2 order. The
  body was migrated; the docstring was not.
- **`model.py`** and **`train/models.py`** — the anchor was never migrated at
  all. Both still take `pred_1[:, 2:3]` as the fixed topological anchor, which
  is the vein channel under the GAVE2 order.

So the recursive refinement, the mechanism that carries topological quality, was
holding the wrong channel fixed. Correcting it is the first of our two bug
fixes; see the technical report.

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

@article{budai2013robust,
  title   = {Robust Vessel Segmentation in Fundus Images},
  author  = {Budai, Attila and Bock, R{\"u}diger and Maier, Andreas and
             Hornegger, Joachim and Michelson, Georg},
  journal = {International Journal of Biomedical Imaging},
  volume  = {2013},
  pages   = {154860},
  year    = {2013}
}

@article{fu2018joint,
  title   = {Joint Optic Disc and Cup Segmentation Based on Multi-Label Deep
             Network and Polar Transformation},
  author  = {Fu, Huazhu and Cheng, Jun and Xu, Yanwu and Wong, Damon Wing Kee
             and Liu, Jiang and Cao, Xiaochun},
  journal = {IEEE Transactions on Medical Imaging},
  volume  = {37},
  number  = {7},
  pages   = {1597--1605},
  year    = {2018}
}
```

**What each is used for.** Morano et al. is the RRWNet architecture the GAVE2
baseline derives from. Hemelings et al. released the artery/vein annotations for
HRF that we added to our training split (obtained from
<https://github.com/rubenhx/av-segmentation>, which requests this citation);
Budai et al. is the underlying HRF image database, downloadable at
<https://www5.cs.fau.de/research/data/fundus-images/>. Fu et al. is
[MNet_DeepCDR](https://github.com/HzFu/MNet_DeepCDR), whose released disc
detector we ported (`scripts/02_segment_optic_disc.py`) to produce the
optic-disc masks required by the Task 3 measurements.

Official GAVE2 baseline: <https://github.com/Peng2004/CMRRWNet>
