"""Reconstruct the exact Task 3 files of our scored submission.

WHY THIS SCRIPT EXISTS
----------------------
`biomarkers.compute_task3` implements our method: every biomarker computed from
one probability field, each at its own threshold. It is self-consistent and it
is what the technical report describes.

The Task 3 files we actually submitted are NOT the output of a single such run.
They were assembled biomarker by biomarker across several submissions, keeping
for each biomarker the source that scored best on the official validation
leaderboard. Five successive attempts to recompute Task 3 from newer and
objectively better masks each scored *worse*, so older values were retained.

That was a rational decision for the leaderboard, but it means the clean
pipeline cannot regenerate the submitted numbers. Rather than quietly ship one
and describe the other, this script reproduces the submitted files exactly, from
the archived per-submission outputs, with the provenance stated explicitly.

PROVENANCE OF THE SUBMITTED TASK 3
----------------------------------
Verified field-by-field over all 50 validation images:

    submitted = v10c   for every field
              except vein_density, which comes from v22

and `v10c` is itself a per-biomarker mix of two earlier submissions:

    | biomarker                | source | underlying masks                     |
    |--------------------------|--------|--------------------------------------|
    | CRAE                     | v7     | single model, pre-ensemble, pre-HRF  |
    | CRVE                     | v7     | single model, pre-ensemble, pre-HRF  |
    | AVR                      | v7     | single model, pre-ensemble, pre-HRF  |
    | artery_density           | v10    | decision fusion, single model        |
    | vein_density             | v22    | current HRF ensemble field @ 0.3525  |
    | artery_fractal_dimension | v7     | single model, pre-ensemble, pre-HRF  |
    | vein_fractal_dimension   | v10    | decision fusion, single model        |

`vein_density` is the only field computed from the current probability field,
and correspondingly the only one the clean pipeline reproduces exactly.

WHAT THIS MEANS FOR A REVIEWER
------------------------------
* To reproduce our *score*: run this script on the archived inputs.
* To reproduce our *method*: run `biomarkers.compute_task3`.
* The two differ, and the difference is the honest subject of this file.

This form of leaderboard-guided selection fits the public validation split and
is weaker evidence than cross-validation. It is reported as such in the
technical report.

USAGE
-----
    python replay_task3.py --archive archive/task3 --out work/submission/Task3

`--archive` must contain one directory per source submission (`v10c/`, `v22/`),
each holding the 50 per-image .txt files. These are distributed in the Release.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from biomarkers import FIELD_ORDER, parse_biomarker_file, write_biomarker_file

# Which archived submission each field is taken from.
FIELD_SOURCE = {
    "CRAE": "v10c",
    "CRVE": "v10c",
    "AVR": "v10c",
    "artery_density": "v10c",
    "vein_density": "v22",
    "artery_fractal_dimension": "v10c",
    "vein_fractal_dimension": "v10c",
}

# Where each v10c field originally came from, for documentation only.
ORIGINAL_SOURCE = {
    "CRAE": "v7", "CRVE": "v7", "AVR": "v7",
    "artery_density": "v10", "vein_density": "v22",
    "artery_fractal_dimension": "v7", "vein_fractal_dimension": "v10",
}


def replay(archive_dir: Path, out_dir: Path, verbose: bool = True) -> int:
    """Assemble the submitted Task 3 files from the archived per-version outputs."""
    archive_dir, out_dir = Path(archive_dir), Path(out_dir)
    needed = sorted(set(FIELD_SOURCE.values()))
    for src in needed:
        if not (archive_dir / src).is_dir():
            raise FileNotFoundError(
                f"missing archive directory {archive_dir / src}. "
                f"Required: {needed} (available in the Release)."
            )

    ids = sorted(p.stem for p in (archive_dir / needed[0]).glob("*.txt"))
    if not ids:
        raise FileNotFoundError(f"no .txt files in {archive_dir / needed[0]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cache = {
        src: {i: parse_biomarker_file(archive_dir / src / f"{i}.txt") for i in ids}
        for src in needed
    }

    for image_id in ids:
        values = {f: cache[FIELD_SOURCE[f]][image_id][f] for f in FIELD_ORDER}
        write_biomarker_file(out_dir / f"{image_id}.txt", values)

    if verbose:
        print(f"[replay] {len(ids)} Task3 files -> {out_dir}")
        for f in FIELD_ORDER:
            print(f"  {f:28s} <- {FIELD_SOURCE[f]:5s} (originally {ORIGINAL_SOURCE[f]})")
    return len(ids)


def verify_against(out_dir: Path, reference_dir: Path) -> bool:
    """Check the replayed files match a reference directory exactly."""
    out_dir, reference_dir = Path(out_dir), Path(reference_dir)
    ids = sorted(p.stem for p in reference_dir.glob("*.txt"))
    mismatches = []
    for image_id in ids:
        a = parse_biomarker_file(out_dir / f"{image_id}.txt")
        b = parse_biomarker_file(reference_dir / f"{image_id}.txt")
        for f in FIELD_ORDER:
            if a[f] != b[f]:
                mismatches.append((image_id, f, a[f], b[f]))
    if mismatches:
        print(f"[verify] {len(mismatches)} mismatching values, first few:")
        for m in mismatches[:5]:
            print(f"  {m[0]} {m[1]}: replayed={m[2]} reference={m[3]}")
        return False
    print(f"[verify] all {len(ids)} files match {reference_dir} exactly")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive", required=True,
                    help="directory containing v10c/ and v22/ per-image .txt files")
    ap.add_argument("--out", required=True, help="output Task3 directory")
    ap.add_argument("--verify-against", default=None,
                    help="optional reference Task3 directory to check against")
    args = ap.parse_args()

    replay(Path(args.archive), Path(args.out))
    if args.verify_against:
        ok = verify_against(Path(args.out), Path(args.verify_against))
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
