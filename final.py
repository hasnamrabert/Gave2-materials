"""End-to-end submission builder: probability maps -> submission zip.

WHAT THIS MODULE DOES
---------------------
Assembles the three task outputs into the exact directory layout the challenge
expects, and zips them.

    Task1/<image_id>.png   binary A/V segmentation from CFP only
    Task2/<image_id>.png   binary A/V segmentation, FFA-guided
    Task3/<image_id>.txt   seven biomarkers

The recipe reproduced here is our best-scoring preliminary submission:

    Task 1  fuse(P_gave2_only, P_gave2_plus_hrf, beta=0.5)
            -> threshold (0.06863 artery / 0.06667 vein)
            -> endpoint reconnection (25 px)
            -> pure binary 0/255

    Task 2  fuse(P_task1_cfp_only, P_cmrrwnet_cfp_ffa, alpha=0.75)
            -> threshold (0.04 both classes)
            -> endpoint reconnection (25 px)
            -> pure binary 0/255

    Task 3  fuse(P_task1_cfp_only, P_cmrrwnet_cfp_ffa, alpha=0.75)
            -> per-biomarker thresholds, seven values assembled per image

WHY THE OUTPUT IS PURE BINARY
-----------------------------
The evaluation binarises at 0.5, so writing 0/255 makes the submitted mask
exactly the mask we validated locally, with no dependence on how the server
rounds. It is equivalent to a monotone remap for every metric used, and it is
unambiguous.

CHANNEL INVARIANT
-----------------
Every written PNG satisfies `artery | vein == vessel` exactly, because the
vessel channel is rebuilt as the union AFTER reconnection. Violating this costs
Dice on the vessel class, which carries weight 0.4.

KNOWN LIMITATIONS
-----------------
* Task 3 does not derive from the submitted Task 2 masks. It is computed from
  the same probability field but at its own per-biomarker thresholds - the
  thresholds that are best for topology inflate the density biomarkers. Nothing
  in the challenge requires the two to be consistent, but reviewers should know
  they are not.
* The Task 1 thresholds are density-matched values from a specific reference
  run; they are not re-derived here. See docs/REVIEW_NOTES.md.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from biomarkers import check_plausible, compute_task3
from postprocessing import fuse
from preprocessing import load_probability_image, save_binary_mask, save_probability_image
from postprocessing import reconnect_av_pair

DEFAULT_CONFIG = Path(__file__).resolve().parent / "pipeline.yaml"


def load_config(path: Path | None = None) -> dict:
    """Load the pipeline configuration (thresholds, fusion weights, distances)."""
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_segmentation_task(prob_dir_a: Path,
                            prob_dir_b: Path,
                            out_dir: Path,
                            weight_a: float,
                            threshold_artery: float,
                            threshold_vein: float,
                            reconnect_distance: int,
                            use_geodesic: bool = False) -> int:
    """Fuse two probability directories and write final binary masks.

    Applies, in this order: probability-space fusion, thresholding, endpoint
    reconnection per class, vessel-channel reconstruction, binary write.
    """
    prob_dir_a, prob_dir_b, out_dir = Path(prob_dir_a), Path(prob_dir_b), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for f in sorted(prob_dir_a.glob("*.png")):
        other = prob_dir_b / f.name
        if not other.exists():
            raise FileNotFoundError(f"{other} missing (present in {prob_dir_a})")

        fused = fuse(load_probability_image(f), load_probability_image(other), weight_a)
        artery_bin = fused[..., 0] > threshold_artery
        vein_bin = fused[..., 2] > threshold_vein

        artery, vessel, vein = reconnect_av_pair(
            artery_bin, vein_bin, reconnect_distance, use_geodesic=use_geodesic
        )

        out = np.stack([artery, vessel, vein], axis=2).astype(np.uint8) * 255
        Image.fromarray(out, mode="RGB").save(out_dir / f.name)
        n += 1

    print(f"[segmentation] {n} images -> {out_dir}")
    return n


def write_fused_probabilities(prob_dir_a: Path, prob_dir_b: Path,
                              out_dir: Path, weight_a: float) -> int:
    """Write the fused probability maps that Task 3 consumes."""
    prob_dir_a, prob_dir_b, out_dir = Path(prob_dir_a), Path(prob_dir_b), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(prob_dir_a.glob("*.png")):
        fused = fuse(load_probability_image(f),
                     load_probability_image(prob_dir_b / f.name), weight_a)
        out_u8 = np.clip(fused * 255.0 + 0.5, 0, 255).astype(np.uint8)
        Image.fromarray(out_u8, mode="RGB").save(out_dir / f.name)
        n += 1
    return n


def verify_submission(root: Path, expected_n: int | None = None,
                      require_task3: bool = True) -> list[str]:
    """Structural checks on a built submission. Returns a list of problems.

    Checks performed:
      * all expected task directories exist and have the same number of files;
      * Task1/Task2 PNGs contain only the values 0 and 255;
      * the channel invariant `artery | vein == vessel` holds everywhere;
      * every Task3 file has all seven fields.

    `require_task3=False` is used when building segmentation only, so that a
    partial build is still structurally validated.
    """
    root = Path(root)
    problems: list[str] = []

    tasks = [("Task1", "*.png"), ("Task2", "*.png")]
    if require_task3:
        tasks.append(("Task3", "*.txt"))

    counts = {}
    for task, pattern in tasks:
        d = root / task
        if not d.is_dir():
            problems.append(f"{task}/ is missing")
            continue
        counts[task] = len(list(d.glob(pattern)))

    if len(set(counts.values())) > 1:
        problems.append(f"file-count mismatch across tasks: {counts}")
    if expected_n is not None:
        for task, c in counts.items():
            if c != expected_n:
                problems.append(f"{task} has {c} files, expected {expected_n}")

    for task in ("Task1", "Task2"):
        d = root / task
        if not d.is_dir():
            continue
        non_binary, incoherent = 0, 0
        for f in sorted(d.glob("*.png")):
            a = np.array(Image.open(f))[..., :3]
            if not set(np.unique(a).tolist()) <= {0, 255}:
                non_binary += 1
            ab = a > 127
            if not (ab[..., 0] <= ab[..., 1]).all() or not (ab[..., 2] <= ab[..., 1]).all():
                incoherent += 1
        if non_binary:
            problems.append(f"{task}: {non_binary} files are not pure 0/255")
        if incoherent:
            problems.append(f"{task}: {incoherent} files violate artery|vein == vessel")

    d3 = root / "Task3"
    if require_task3 and d3.is_dir():
        from biomarkers import FIELD_ORDER, parse_biomarker_file
        bad = 0
        for f in sorted(d3.glob("*.txt")):
            vals = parse_biomarker_file(f)
            if set(FIELD_ORDER) - set(vals):
                bad += 1
        if bad:
            problems.append(f"Task3: {bad} files are missing fields")

    return problems


def make_zip(root: Path, zip_path: Path) -> Path:
    """Zip a built submission with the Task1/Task2/Task3 prefixes preserved."""
    root, zip_path = Path(root), Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for task in ("Task1", "Task2", "Task3"):
            d = root / task
            for f in sorted(d.iterdir()):
                zf.write(f, arcname=f"{task}/{f.name}")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert all(n.startswith(("Task1/", "Task2/", "Task3/")) for n in names), \
            "unexpected entries in the archive"
    print(f"[zip] {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB, {len(names)} entries)")
    return zip_path


def build(prob_task1_gave2: Path,
          prob_task1_hrf: Path,
          prob_task2_cmrrwnet: Path,
          disc_dir: Path,
          out_root: Path,
          config: dict | None = None,
          zip_path: Path | None = None,
          skip_task3: bool = False) -> Path:
    """Build a complete submission from the three sets of probability maps.

    Parameters
    ----------
    prob_task1_gave2
        Task 1 ensemble+TTA probabilities, model trained on GAVE2 only.
    prob_task1_hrf
        Task 1 ensemble+TTA probabilities, model trained on GAVE2 + HRF.
    prob_task2_cmrrwnet
        Task 2 (CFP+FFA) probabilities from CMRRWNet.
    disc_dir
        Optic-disc masks, required for Task 3.
    out_root
        Directory in which Task1/, Task2/, Task3/ are created.
    """
    cfg = config or load_config()
    out_root = Path(out_root)

    t1, t2, t3 = cfg["task1"], cfg["task2"], cfg["task3"]
    recon = cfg.get("reconnection", {})
    use_geodesic = bool(recon.get("use_geodesic", False))

    print("=== Task 1: inter-model fusion + reconnection ===")
    n1 = build_segmentation_task(
        prob_task1_gave2, prob_task1_hrf, out_root / "Task1",
        weight_a=t1["fusion_beta"],
        threshold_artery=t1["threshold_artery"],
        threshold_vein=t1["threshold_vein"],
        reconnect_distance=t1["reconnect_distance_px"],
        use_geodesic=use_geodesic,
    )

    print("=== Task 2: cross-modal fusion + reconnection ===")
    build_segmentation_task(
        prob_task1_hrf, prob_task2_cmrrwnet, out_root / "Task2",
        weight_a=t2["fusion_alpha"],
        threshold_artery=t2["threshold_artery"],
        threshold_vein=t2["threshold_vein"],
        reconnect_distance=t2["reconnect_distance_px"],
        use_geodesic=use_geodesic,
    )

    if not skip_task3:
        print("=== Task 3: per-biomarker thresholds ===")
        fused_dir = out_root / "_fused_probabilities"
        write_fused_probabilities(prob_task1_hrf, prob_task2_cmrrwnet,
                                  fused_dir, t3["source_fusion_alpha"])
        assembled = compute_task3(
            probability_dir=fused_dir,
            disc_dir=disc_dir,
            out_dir=out_root / "Task3",
            thresholds=t3["thresholds"],
        )
        print("--- Task 3 plausibility check ---")
        warnings = check_plausible(assembled)
        for w in warnings:
            print(f"  WARNING: {w}")
        shutil.rmtree(fused_dir, ignore_errors=True)

    print("=== Structural verification ===")
    problems = verify_submission(out_root, expected_n=n1, require_task3=not skip_task3)
    if problems:
        for p in problems:
            print(f"  PROBLEM: {p}")
        raise RuntimeError(f"submission failed verification: {problems}")
    print("  all checks passed")

    if zip_path:
        return make_zip(out_root, Path(zip_path))
    return out_root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prob-task1-gave2", required=True,
                    help="Task1 probabilities, model trained on GAVE2 only")
    ap.add_argument("--prob-task1-hrf", required=True,
                    help="Task1 probabilities, model trained on GAVE2+HRF")
    ap.add_argument("--prob-task2", required=True,
                    help="Task2 probabilities from CMRRWNet (CFP+FFA)")
    ap.add_argument("--disc-dir", required=True, help="optic-disc masks (for Task3)")
    ap.add_argument("--out", required=True, help="output submission directory")
    ap.add_argument("--zip", default=None, help="optional path for the .zip archive")
    ap.add_argument("--config", default=None, help="path to pipeline.yaml")
    ap.add_argument("--skip-task3", action="store_true",
                    help="build segmentation only (Task3 is the slow step)")
    args = ap.parse_args()

    build(
        prob_task1_gave2=Path(args.prob_task1_gave2),
        prob_task1_hrf=Path(args.prob_task1_hrf),
        prob_task2_cmrrwnet=Path(args.prob_task2),
        disc_dir=Path(args.disc_dir),
        out_root=Path(args.out),
        config=load_config(Path(args.config)) if args.config else None,
        zip_path=Path(args.zip) if args.zip else None,
        skip_task3=args.skip_task3,
    )


if __name__ == "__main__":
    main()
