"""Image loading, padding, and the channel conventions used throughout.

CHANNEL CONVENTIONS - READ THIS FIRST
-------------------------------------
Three conventions are in play. Confusing them silently produces wrong output
rather than an error, so they are stated here once and referenced everywhere.

    Upstream RRWNet (Morano et al.)   index 0 = artery, 1 = vein,   2 = vessel
    GAVE2 (this challenge)            index 0 = artery, 1 = vessel, 2 = vein
    GAVE2 ground-truth PNG files      R = artery,  G = crossing/unknown, B = vein

The GAVE2 *prediction* format is R = artery, G = vessel, B = vein, with artery
and vein subsets of vessel. The GAVE2 *ground-truth* format uses the green
channel for crossings instead, and its R/G/B channels are mutually disjoint.

The divergence between the first two rows is not cosmetic: it is the origin of
the topological-anchor bug we fixed in `model.py`. The challenge baseline
migrated its loss function to the GAVE2 order but left the model's anchor index
pointing at the upstream one. Evidence, from the baseline's own `losses.py`,
where the docstring and the code contradict each other:

    docstring:  artery: 0, vein: 1, vessel_tree: 2      (upstream order)
    body:       pred_a  = pred[:, 0]                    (artery = 0)
                pred_v  = pred[:, 2]                    (vein   = 2)
                pred_vt = pred[:, 1]                    (vessel = 1)

The body implements the GAVE2 order; the docstring still describes the upstream
one. `models.py` was likewise never updated, so it anchored index 2 - "vessel"
upstream, but "vein" under GAVE2.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Channel index of each class, GAVE2 convention.
ARTERY, VESSEL, VEIN = 0, 1, 2

# Channel index of each class in the provided ground-truth files.
GT_ARTERY, GT_UNKNOWN, GT_VEIN = 0, 1, 2


def get_unet_padding(np_image: np.ndarray, n_down: int = 5) -> tuple:
    """Padding needed so both spatial dimensions are divisible by 2**n_down.

    The U-Net trunk downsamples five times, so each dimension must be a multiple
    of 32. Padding is split as evenly as possible between the two sides.
    """
    n = 2 ** n_down
    shape = np_image.shape
    h_pad = n - shape[0] % n
    w_pad = n - shape[1] % n
    h_half, w_half = h_pad // 2, w_pad // 2
    if len(shape) == 3:
        return (h_half, h_pad - h_half), (w_half, w_pad - w_half), (0, 0)
    return (h_half, h_pad - h_half), (w_half, w_pad - w_half)


def load_probability_image(path: Path) -> np.ndarray:
    """Load an (H, W, 3) probability map in [0, 1] from an 8-bit PNG."""
    return np.array(Image.open(path))[..., :3].astype(np.float64) / 255.0


def save_probability_image(array: np.ndarray, path: Path) -> None:
    """Write an (H, W, 3) array in [0, 1] as an 8-bit PNG."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(out, mode="RGB").save(path)


def save_binary_mask(artery: np.ndarray, vessel: np.ndarray, vein: np.ndarray,
                     path: Path) -> None:
    """Write a pure 0/255 RGB mask in the GAVE2 prediction convention.

    The caller is responsible for ensuring `artery | vein == vessel`; this is
    checked by `final.verify_submission`.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = np.stack([artery, vessel, vein], axis=2).astype(np.uint8) * 255
    Image.fromarray(out, mode="RGB").save(path)


def load_fov(path: Path) -> np.ndarray:
    """Load a field-of-view mask as a boolean array."""
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    return m > 127


def load_groundtruth_av(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load artery and vein masks from a GROUND-TRUTH file.

    Ground truth uses R = artery, G = crossing/unknown, B = vein, with the three
    channels disjoint. Do NOT feed ground-truth files to the prediction-side
    `biomarkers.extract_av_masks`, which assumes G = vessel and will silently
    return nearly empty masks.
    """
    img = np.array(Image.open(path))[..., :3]
    return img[..., GT_ARTERY] > 127, img[..., GT_VEIN] > 127
