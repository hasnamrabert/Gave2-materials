"""Checkpoint-ensemble + test-time-augmentation prediction.

WHAT THIS MODULE DOES
---------------------
Produces the probability maps that everything downstream consumes. Two
variance-reduction techniques are combined, both averaging in PROBABILITY space
before any thresholding:

* Checkpoint ensembling - the 4 cross-validation fold checkpoints are averaged.
* Test-time augmentation - each image is predicted under the 8 transforms of the
  dihedral group D4 (4 rotations x optional horizontal flip), each prediction is
  mapped back to the original orientation, and the 8 results are averaged.

RELATION TO THE BASELINE
------------------------
The baseline predicts once, with a single checkpoint and no augmentation. Both
additions here are ours.

Honest note on their value: on our data, ensembling + TTA produced a very large
apparent gain in cross-validation (+0.30 Task 1, +0.42 Task 2) that did NOT
transfer to the official score (both changes landed inside the measurement
noise). We kept them because they are harmless and they stabilise the operating
point across folds, but they should not be presented as a source of our gains.
See docs/REVIEW_NOTES.md.

IMPORTANT ORDERING CONSTRAINT
-----------------------------
Averaging happens on raw sigmoid outputs. Any thresholding, remapping or
binarisation must happen strictly afterwards - see
`gave2/postprocessing/fusion.py` for why.

KNOWN LIMITATIONS
-----------------
* Predictions outside the field-of-view mask are forced to zero. This is
  intentional (the metric only scores inside the FOV) but means the output maps
  cannot be reused for anything needing off-FOV values.
* Padding to a multiple of 32 is applied per transform and cropped back; with a
  non-square image the rotated variants have different padding, which is handled
  but makes the 8 passes slightly non-identical numerically.
* Runs one image at a time (batch size 1). This is the memory-safe choice at
  1024x1536 but it is not the fastest possible.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from skimage import io

from model import build_model
from preprocessing import get_unet_padding
from transformations import D4_TRANSFORMS, IDENTITY_ONLY, d4_forward, d4_inverse









def predict_one(model, image: np.ndarray, device, use_tta: bool = True) -> np.ndarray:
    """Predict one image, averaging over the D4 transforms.

    Parameters
    ----------
    image
        (H, W, C) float array in [0, 1]; C is 3 for Task 1, 5 for Task 2.
    use_tta
        When False, only the identity transform is used.

    Returns
    -------
    (H, W, 3) float array of sigmoid probabilities, channel order
    [artery, vessel, vein].
    """
    transforms = D4_TRANSFORMS if use_tta else IDENTITY_ONLY
    acc = None
    for k, flip in transforms:
        timg = d4_forward(image, k, flip)
        padding = get_unet_padding(timg)
        padded = np.pad(timg, padding)
        tensor = torch.from_numpy(padded.transpose(2, 0, 1).astype("float32"))
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(tensor)
            last = torch.sigmoid(preds[-1])

        # Crop the padding back off.
        last = last[:, :, padding[0][0]:last.shape[2] - padding[0][1],
                          padding[1][0]:last.shape[3] - padding[1][1]]
        pred_np = last[0].permute(1, 2, 0).cpu().numpy()
        acc_i = d4_inverse(pred_np, k, flip)
        acc = acc_i if acc is None else acc + acc_i
    return acc / len(transforms)


def load_models(task: str, checkpoints: Sequence[Path], device) -> list:
    """Load one model per checkpoint, in eval mode, on the given device."""
    models = []
    for ckpt in checkpoints:
        m = build_model(task)
        state = torch.load(str(ckpt), map_location=device)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        m.load_state_dict(state, strict=True)
        m.eval().to(device)
        models.append(m)
    return models


def predict_directory(task: str,
                      checkpoints: Sequence[Path],
                      images_path: Path,
                      masks_path: Path,
                      save_path: Path,
                      ffa_a_path: Path | None = None,
                      ffa_av_path: Path | None = None,
                      use_tta: bool = True,
                      device: str | None = None) -> int:
    """Run ensemble + TTA prediction over a directory of images.

    For Task 2, `ffa_a_path` and `ffa_av_path` must point at the early- and
    late-phase FFA directories; they are appended as channels 4 and 5.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    in_channels = 3 if task == "task1" else 5
    if in_channels == 5 and (ffa_a_path is None or ffa_av_path is None):
        raise ValueError("task2 requires --a-path and --av-path (FFA channels)")

    models = load_models(task, checkpoints, dev)
    print(f"[predict] task={task} checkpoints={len(models)} "
          f"tta={'D4 (8)' if use_tta else 'off'} device={dev}")

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(Path(images_path).glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"no PNG images found in {images_path}")

    for image_file in image_files:
        stem = image_file.stem
        img = (io.imread(image_file) / 255.0)[..., :3]
        mask = io.imread(Path(masks_path) / f"{stem}.png") * 1.0
        if mask.ndim == 3:
            mask = mask[..., 0]

        if in_channels == 5:
            r_a = io.imread(Path(ffa_a_path) / f"{stem}.png") / 255.0
            r_av = io.imread(Path(ffa_av_path) / f"{stem}.png") / 255.0
            if r_a.ndim == 2:
                r_a = r_a[..., None]
            if r_av.ndim == 2:
                r_av = r_av[..., None]
            model_input = np.concatenate([img, r_a[..., :1], r_av[..., :1]], axis=2)
        else:
            model_input = img

        # Average across checkpoints, then mask out everything outside the FOV.
        acc = None
        for m in models:
            p = predict_one(m, model_input, dev, use_tta=use_tta)
            acc = p if acc is None else acc + p
        acc = acc / len(models)
        acc[mask < 0.5] = 0

        out_u8 = np.clip(acc * 255.0 + 0.5, 0, 255).astype(np.uint8)
        io.imsave(save_path / image_file.name, out_u8, check_contrast=False)

    print(f"[predict] {len(image_files)} images -> {save_path}")
    return len(image_files)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ensemble + TTA prediction for GAVE2.")
    ap.add_argument("--task", choices=["task1", "task2"], required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="one or more checkpoints; all are averaged (we use 4)")
    ap.add_argument("--images-path", required=True)
    ap.add_argument("--masks-path", required=True, help="field-of-view masks")
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--a-path", default=None, help="early-phase FFA (task2 only)")
    ap.add_argument("--av-path", default=None, help="late-phase FFA (task2 only)")
    ap.add_argument("--no-tta", action="store_true", help="disable test-time augmentation")
    ap.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu")
    args = ap.parse_args()

    predict_directory(
        task=args.task,
        checkpoints=[Path(c) for c in args.checkpoints],
        images_path=Path(args.images_path),
        masks_path=Path(args.masks_path),
        save_path=Path(args.save_path),
        ffa_a_path=Path(args.a_path) if args.a_path else None,
        ffa_av_path=Path(args.av_path) if args.av_path else None,
        use_tta=not args.no_tta,
        device=args.device,
    )


if __name__ == "__main__":
    main()
