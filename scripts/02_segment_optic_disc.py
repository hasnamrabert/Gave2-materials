"""Optic-disc segmentation, required by the Task 3 biomarker pipeline.

WHAT THIS DOES
--------------
Produces one binary optic-disc mask per fundus image. The Task 3 measurements
are defined on annular zones whose radii are multiples of the optic-disc
diameter, so every biomarker depends on this mask.

PROVENANCE
----------
Ported from MNet_DeepCDR (`Model_DiscSeg.py`,
https://github.com/HzFu/MNet_DeepCDR). We only need the disc detector: a plain
U-Net taking 640x640 input and producing a single sigmoid channel, whose
pre-trained weights (`Model_DiscSeg_ORIGA.h5`) the authors released.

The original repository targets TensorFlow 1.14 / legacy Keras / MATLAB. Rather
than reconstruct that toolchain we re-declared the identical architecture under
`tf.keras`, preserving every layer name so the released `.h5` weights load
directly.

IMPORTANT - SEPARATE ENVIRONMENT REQUIRED
-----------------------------------------
This script needs TensorFlow, whereas the rest of the pipeline needs PyTorch.
The two do not coexist comfortably (CUDA runtime conflicts), so this step runs
in its own environment:

    conda create -n gave2-od python=3.10
    conda activate gave2-od
    pip install tensorflow==2.13.1 scikit-image numpy

This is the one step of the pipeline that is NOT covered by the main Docker
image; see docs/REVIEW_NOTES.md, item 3.1.

OUTPUT
------
One binary PNG (0/255) per input image, at the original resolution, ready to be
passed as `--disc-dir` to the biomarker pipeline (which thresholds at 200 and
takes the enclosing circle).

Usage:
    python scripts/02_segment_optic_disc.py \
        --images data/validation/images \
        --weights external/MNet_DeepCDR/deep_model/Model_DiscSeg_ORIGA.h5 \
        --out work/optic_disc
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.measure import label, regionprops
from tensorflow import keras
from tensorflow.keras.layers import (Input, concatenate, Conv2D, MaxPooling2D,
                                     Conv2DTranspose, UpSampling2D, average)

DISCSEG_SIZE = 640


def build_discseg(size_set: int = DISCSEG_SIZE) -> keras.Model:
    """Exact replica of MNet_DeepCDR Model_DiscSeg.DeepModel (identical layer names)."""
    img_input = Input(shape=(size_set, size_set, 3))
    c1 = Conv2D(32, 3, activation='relu', padding='same', name='block1_conv1')(img_input)
    c1 = Conv2D(32, 3, activation='relu', padding='same', name='block1_conv2')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(64, 3, activation='relu', padding='same', name='block2_conv1')(p1)
    c2 = Conv2D(64, 3, activation='relu', padding='same', name='block2_conv2')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(128, 3, activation='relu', padding='same', name='block3_conv1')(p2)
    c3 = Conv2D(128, 3, activation='relu', padding='same', name='block3_conv2')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(256, 3, activation='relu', padding='same', name='block4_conv1')(p3)
    c4 = Conv2D(256, 3, activation='relu', padding='same', name='block4_conv2')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    c5 = Conv2D(512, 3, activation='relu', padding='same', name='block5_conv1')(p4)
    c5 = Conv2D(512, 3, activation='relu', padding='same', name='block5_conv2')(c5)

    u6 = concatenate([Conv2DTranspose(256, 2, strides=2, padding='same', name='block6_dconv')(c5), c4], axis=3)
    c6 = Conv2D(256, 3, activation='relu', padding='same', name='block6_conv1')(u6)
    c6 = Conv2D(256, 3, activation='relu', padding='same', name='block6_conv2')(c6)
    u7 = concatenate([Conv2DTranspose(128, 2, strides=2, padding='same', name='block7_dconv')(c6), c3], axis=3)
    c7 = Conv2D(128, 3, activation='relu', padding='same', name='block7_conv1')(u7)
    c7 = Conv2D(128, 3, activation='relu', padding='same', name='block7_conv2')(c7)
    u8 = concatenate([Conv2DTranspose(64, 2, strides=2, padding='same', name='block8_dconv')(c7), c2], axis=3)
    c8 = Conv2D(64, 3, activation='relu', padding='same', name='block8_conv1')(u8)
    c8 = Conv2D(64, 3, activation='relu', padding='same', name='block8_conv2')(c8)
    u9 = concatenate([Conv2DTranspose(32, 2, strides=2, padding='same', name='block9_dconv')(c8), c1], axis=3)
    c9 = Conv2D(32, 3, activation='relu', padding='same', name='block9_conv1')(u9)
    c9 = Conv2D(32, 3, activation='relu', padding='same', name='block9_conv2')(c9)

    s6 = Conv2D(1, 1, activation='sigmoid', name='side_6')(UpSampling2D((8, 8))(c6))
    s7 = Conv2D(1, 1, activation='sigmoid', name='side_7')(UpSampling2D((4, 4))(c7))
    s8 = Conv2D(1, 1, activation='sigmoid', name='side_8')(UpSampling2D((2, 2))(c8))
    s9 = Conv2D(1, 1, activation='sigmoid', name='side_9')(c9)
    out = average([s6, s7, s8, s9])
    return keras.Model(inputs=[img_input], outputs=[out])


def largest_component(bw: np.ndarray) -> np.ndarray:
    """Keep the largest connected component (the disc)."""
    lbl = label(bw)
    if lbl.max() == 0:
        return bw
    regions = regionprops(lbl)
    biggest = max(regions, key=lambda r: r.area)
    return lbl == biggest.label


def predict_od_mask(model, img_rgb: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Return a boolean optic-disc mask at the input image resolution."""
    H, W = img_rgb.shape[:2]
    # Same preprocessing as the original Step_1_Disc_Crop: resize to 640, rescale to [0,255]
    x = resize(img_rgb[..., :3], (DISCSEG_SIZE, DISCSEG_SIZE, 3), preserve_range=False) * 255.0
    x = x[None].astype(np.float32)
    disc = model.predict(x, verbose=0).reshape(DISCSEG_SIZE, DISCSEG_SIZE)
    disc_bw = disc > thresh
    if disc_bw.sum() > 0:
        disc_bw = largest_component(disc_bw)
    # Resize back to the original resolution (nearest neighbour keeps it binary)
    disc_full = resize(disc_bw.astype(np.float32), (H, W), order=0, preserve_range=True) > 0.5
    return disc_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--thresh', type=float, default=0.5)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = build_discseg()
    model.load_weights(args.weights)
    print(f"[discseg] weights loaded from {args.weights}")

    files = sorted(Path(args.images).glob('*.png'))
    for f in files:
        img = io.imread(f)
        od = predict_od_mask(model, img, args.thresh)
        cover = 100.0 * od.mean()
        io.imsave(out / f.name, (od.astype(np.uint8) * 255), check_contrast=False)
        print(f"  {f.name}: OD {cover:.2f}% of the image")
    print(f"[discseg] {len(files)} optic-disc masks written to {out}")


if __name__ == '__main__':
    main()
