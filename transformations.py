"""Test-time augmentation transforms (dihedral group D4).

Each image is predicted under the 8 symmetries of the square - four rotations,
each with and without a horizontal flip - and every prediction is mapped back to
the original orientation before averaging. Averaging happens in probability
space, before any thresholding.

Honest note on value: on our data this produced a large apparent gain in
cross-validation that did not survive contact with the official metric (the
change landed inside the measurement noise). It is retained because it costs
only inference time and stabilises the operating point across folds, not because
it is a demonstrated source of improvement.
"""
from __future__ import annotations

import numpy as np

# The 8 elements of D4, as (number of 90-degree rotations, horizontal flip).
D4_TRANSFORMS = [(k, flip) for k in range(4) for flip in (False, True)]

# Identity only, for disabling TTA without changing the calling code.
IDENTITY_ONLY = [(0, False)]


def d4_forward(img: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """Apply a D4 element: rotate by k*90 degrees, then optionally mirror."""
    x = np.rot90(img, k, axes=(0, 1))
    if flip:
        x = np.flip(x, axis=1)
    return np.ascontiguousarray(x)


def d4_inverse(img: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """Invert `d4_forward`: unmirror first, then rotate back.

    Order matters - the inverse of (rotate, then flip) is (unflip, then
    unrotate), not (unrotate, then unflip).
    """
    x = img
    if flip:
        x = np.flip(x, axis=1)
    x = np.rot90(x, -k, axes=(0, 1))
    return np.ascontiguousarray(x)
