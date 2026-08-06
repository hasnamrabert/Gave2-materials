#!/usr/bin/env bash
# End-to-end inference: raw images -> submission zip.
#
#   run_pipeline.sh <data_dir> <weights_dir> <work_dir> [disc_dir]
#
# <data_dir> must contain images/ masks/ and, for Task 2, FFA_A/ FFA_AV/
# <weights_dir> must contain task1_gave2/ task1_hrf/ task2/ each with 4 .pth
# [disc_dir] optic-disc masks; defaults to <data_dir>/optic_disc.
#            Task 3 is skipped if absent.
set -euo pipefail

DATA="${1:?usage: run_pipeline.sh <data_dir> <weights_dir> <work_dir> [disc_dir]}"
WEIGHTS="${2:?missing weights_dir}"
WORK="${3:?missing work_dir}"
DISC="${4:-$DATA/optic_disc}"

mkdir -p "$WORK"

echo "[1/4] Task 1 probabilities (GAVE2-only model)"
python infer.py --task task1 \
  --checkpoints "$WEIGHTS"/task1_gave2/*.pth \
  --images-path "$DATA/images" --masks-path "$DATA/masks" \
  --save-path "$WORK/prob_task1_gave2"

echo "[2/4] Task 1 probabilities (GAVE2+HRF model)"
python infer.py --task task1 \
  --checkpoints "$WEIGHTS"/task1_hrf/*.pth \
  --images-path "$DATA/images" --masks-path "$DATA/masks" \
  --save-path "$WORK/prob_task1_hrf"

echo "[3/4] Task 2 probabilities (CFP+FFA model)"
python infer.py --task task2 \
  --checkpoints "$WEIGHTS"/task2/*.pth \
  --images-path "$DATA/images" --masks-path "$DATA/masks" \
  --a-path "$DATA/FFA_A" --av-path "$DATA/FFA_AV" \
  --save-path "$WORK/prob_task2"

echo "[4/4] Assemble submission"
if [ -d "$DISC" ]; then
  python final.py \
    --prob-task1-gave2 "$WORK/prob_task1_gave2" \
    --prob-task1-hrf   "$WORK/prob_task1_hrf" \
    --prob-task2       "$WORK/prob_task2" \
    --disc-dir         "$DISC" \
    --out "$WORK/submission" --zip "$WORK/submission.zip"
else
  echo "  no optic-disc masks at $DISC -- building Tasks 1-2 only"
  python final.py \
    --prob-task1-gave2 "$WORK/prob_task1_gave2" \
    --prob-task1-hrf   "$WORK/prob_task1_hrf" \
    --prob-task2       "$WORK/prob_task2" \
    --disc-dir         "$DISC" \
    --out "$WORK/submission" --skip-task3
fi

echo "Done -> $WORK/submission"
