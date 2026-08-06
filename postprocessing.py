"""Post-processing: threshold calibration, fusion, density matching, reconnection.

This is where most of our score comes from. Four components, applied in this
order by `final.py`:

1. FUSION            combine models in probability space (never after
                     thresholding - see the ordering note below)
2. THRESHOLD         encode the chosen operating point into the pixels
3. RECONNECTION      repair broken vessel branches under anatomical constraints
4. (DENSITY MATCH)   used offline, to compare two models at an equivalent
                     operating point rather than at an equal threshold

WHY THRESHOLD CALIBRATION MATTERS SO MUCH
-----------------------------------------
Every GAVE2 metric binarises at 0.5 and none is threshold-free. A decision
threshold optimised locally therefore does nothing unless it is baked into the
submitted pixels. Our models operate far below 0.5, so this single step is the
largest contributor to our result.

ORDERING CONSTRAINT
-------------------
Fusion must be applied to raw probabilities. Remapping is a per-model monotone
reparameterisation; averaging two differently remapped maps mixes incompatible
probability scales and invalidates the calibrated thresholds. Always:
    fuse raw -> threshold -> reconnect
never:
    threshold -> fuse
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import convolve, distance_transform_edt
from scipy.spatial import cKDTree
from skimage.draw import line
from skimage.graph import route_through_array
from skimage.measure import label
from skimage.morphology import skeletonize

from preprocessing import load_probability_image, load_fov  # noqa: F401


# =============================================================================
# 1. Decision-level fusion
# =============================================================================
# Two fusions are used:
#   alpha (Task 2, cross-modal)  P = a*P_cfp_only + (1-a)*P_cfp_plus_ffa
#   beta  (Task 1, inter-model)  P = b*P_gave2    + (1-b)*P_gave2_plus_hrf
#
# The baseline fuses CFP and FFA at the network INPUT (5 concatenated channels).
# We measured that this degrades topological coherence relative to the CFP-only
# model on the same architecture (artery COR 0.738 -> 0.586, INF 0.253 -> 0.394):
# the FFA channels contaminate otherwise reliable CFP features. Fusing at the
# decision level avoids that by construction.


def fuse(prob_a: np.ndarray, prob_b: np.ndarray, weight_a: float) -> np.ndarray:
    """Weighted average of two probability maps.

    Returns `weight_a * prob_a + (1 - weight_a) * prob_b`.
    """
    if not 0.0 <= weight_a <= 1.0:
        raise ValueError(f"weight_a must be in [0, 1], got {weight_a}")
    if prob_a.shape != prob_b.shape:
        raise ValueError(f"shape mismatch: {prob_a.shape} vs {prob_b.shape}")
    return weight_a * prob_a + (1.0 - weight_a) * prob_b


def fuse_dirs(dir_a: Path, dir_b: Path, out_dir: Path, weight_a: float,
              verbose: bool = True) -> int:
    """Fuse every matching PNG pair from two directories, writing 8-bit PNGs.

    Files are matched by name. Both directories must contain the same names.
    """
    dir_a, dir_b, out_dir = Path(dir_a), Path(dir_b), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(dir_a.glob("*.png")):
        other = dir_b / f.name
        if not other.exists():
            raise FileNotFoundError(f"{other} missing (present in {dir_a})")
        fused = fuse(load_probability_image(f), load_probability_image(other), weight_a)
        out_u8 = np.clip(fused * 255.0 + 0.5, 0, 255).astype(np.uint8)
        Image.fromarray(out_u8, mode="RGB").save(out_dir / f.name)
        n += 1
    if verbose:
        print(f"[fuse] {n} images: weight_a={weight_a} -> {out_dir}")
    return n



# =============================================================================
# 2. Threshold calibration
# =============================================================================

def remap_monotonic(p: np.ndarray, t: float) -> np.ndarray:
    """Monotonically remap probabilities so that `t` becomes 0.5.

    `p` and `t` are in [0, 1]. The mapping is bijective and order-preserving, so
    it changes no ranking - only where the 0.5 cut falls.
    """
    t = np.clip(t, 1e-6, 1 - 1e-6)
    out = np.empty_like(p, dtype=np.float64)
    below = p < t
    out[below] = 0.5 * p[below] / t
    out[~below] = 0.5 + 0.5 * (p[~below] - t) / (1 - t)
    return out


def binarize_channels(prob_rgb: np.ndarray, t_artery: float, t_vein: float) -> np.ndarray:
    """Binarise an (H, W, 3) probability image into a 0/1 float image.

    The vessel channel is rebuilt as the union of the binarised artery and vein
    channels, so that `artery | vein == vessel` holds exactly, as GAVE2 requires.
    Thresholding the vessel channel independently would break that invariant.
    """
    r_out = (prob_rgb[..., 0] > t_artery).astype(np.float64)
    b_out = (prob_rgb[..., 2] > t_vein).astype(np.float64)
    g_out = np.maximum(r_out, b_out)
    return np.stack([r_out, g_out, b_out], axis=2)


def remap_channels(prob_rgb: np.ndarray, t_artery: float, t_vein: float) -> np.ndarray:
    """Remap an (H, W, 3) probability image so the chosen thresholds land on 0.5."""
    r_out = remap_monotonic(prob_rgb[..., 0], t_artery)
    b_out = remap_monotonic(prob_rgb[..., 2], t_vein)
    g_out = remap_monotonic(prob_rgb[..., 1], (t_artery + t_vein) / 2.0)
    return np.stack([r_out, g_out, b_out], axis=2)


def process_image(src_path: Path, dst_path: Path, t_artery: float, t_vein: float,
                  mode: str = "binarize") -> None:
    """Apply threshold encoding to one PNG and write the result."""
    im = np.array(Image.open(src_path))[..., :3].astype(np.float64) / 255.0

    if mode == "binarize":
        out = binarize_channels(im, t_artery, t_vein)
    elif mode == "remap":
        out = remap_channels(im, t_artery, t_vein)
    else:
        raise ValueError(f"unknown mode {mode!r}, expected 'binarize' or 'remap'")

    out_u8 = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out_u8, mode="RGB").save(dst_path)


def process_dir(src_dir: Path, dst_dir: Path, t_artery: float, t_vein: float,
                mode: str = "binarize", verbose: bool = True) -> int:
    """Apply threshold encoding to every PNG in a directory. Returns the count."""
    n = 0
    for f in sorted(Path(src_dir).glob("*.png")):
        process_image(f, Path(dst_dir) / f.name, t_artery, t_vein, mode)
        n += 1
    if verbose:
        print(f"[{mode}] {n} images: {src_dir} -> {dst_dir} "
              f"(t_artery={t_artery}, t_vein={t_vein})")
    return n


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="directory of probability PNGs")
    ap.add_argument("--dst", required=True, help="output directory")
    ap.add_argument("--t-artery", type=float, required=True)
    ap.add_argument("--t-vein", type=float, required=True)
    ap.add_argument("--mode", choices=["binarize", "remap"], default="binarize")
    args = ap.parse_args()
    process_dir(Path(args.src), Path(args.dst), args.t_artery, args.t_vein, args.mode)



# =============================================================================
# 3. Density matching (offline model comparison)
# =============================================================================
# Comparing two models 'at the same threshold' is only valid if they are
# calibrated the same way. They usually are not. An externally-augmented model
# of ours appeared to gain +0.039 Dice; at the same threshold value it produced
# 7.7%% thinner masks, and a thinner mask mechanically raises Dice. At matched
# density the gain vanished. Fix the mean mask density, not the threshold.

def density_at(prob_maps: Sequence[np.ndarray], fovs: Sequence[np.ndarray],
               threshold: float) -> float:
    """Mean fraction of in-FOV pixels above `threshold`, averaged over images."""
    return float(np.mean([
        ((p > threshold) & fov).sum() / fov.sum()
        for p, fov in zip(prob_maps, fovs)
    ]))


def find_threshold_for_density(prob_maps: Sequence[np.ndarray],
                               fovs: Sequence[np.ndarray],
                               target_density: float,
                               lo: float = 0.005,
                               hi: float = 0.6,
                               max_iter: int = 40,
                               tol: float = 1e-5) -> float:
    """Bisection search for the threshold reproducing `target_density`.

    Density is monotonically decreasing in threshold, so the bracket is updated
    by moving `lo` up when the measured density is still too high.
    """
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        d = density_at(prob_maps, fovs, mid)
        if d > target_density:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


def find_threshold_for_scalar(evaluate: Callable[[float], float],
                              target_value: float,
                              lo: float,
                              hi: float,
                              max_iter: int = 8,
                              tol: float = 5e-4) -> float:
    """Generic bisection for an expensive, monotonically decreasing statistic.

    Used for biomarker-level matching, where each evaluation runs the full
    biomarker pipeline (~2-3 minutes) and only a handful of iterations are
    affordable. `evaluate(threshold) -> statistic`.

    Raises ValueError if the target is not bracketed by [lo, hi], because
    silently returning an endpoint would look like a successful match.
    """
    v_lo, v_hi = evaluate(lo), evaluate(hi)
    if not (v_lo > target_value > v_hi):
        raise ValueError(
            f"target {target_value} not bracketed: f({lo})={v_lo}, f({hi})={v_hi}"
        )
    mid = (lo + hi) / 2.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        v_mid = evaluate(mid)
        if abs(v_mid - target_value) < 1e-4 or (hi - lo) < tol:
            return mid
        if v_mid > target_value:
            lo = mid
        else:
            hi = mid
    return mid



# =============================================================================
# 4. Endpoint reconnection
# =============================================================================

MAX_ANGLE_DEG = 45.0
# Number of skeleton pixels walked back from the endpoint to estimate the tangent.
TANGENT_STEPS = 8
# Geodesic mode: maximum fraction of the routed path allowed to fall outside the
# vessel mask before we fall back to a straight line.
GEODESIC_OFF_VESSEL_MAX = 0.3
# Cost assigned to off-vessel pixels when routing geodesically.
GEODESIC_OFF_COST = 25.0
# Padding around the bounding box of the two endpoints when routing geodesically.
GEODESIC_MARGIN = 10


def find_endpoints(skel: np.ndarray) -> np.ndarray:
    """Return a boolean mask of skeleton pixels having exactly one neighbour.

    Uses a weighted convolution: the centre pixel contributes 10 and each of the
    8 neighbours contributes 1, so a value of exactly 11 means "on the skeleton
    and having a single neighbour", i.e. a degree-1 endpoint.
    """
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    conv = convolve(skel.astype(np.int32), kernel, mode="constant")
    return (conv == 11) & skel


def _walk_skeleton(skel: np.ndarray, start: tuple[int, int], max_steps: int) -> list[tuple[int, int]]:
    """Walk along the skeleton from `start`, never revisiting a pixel.

    Returns the ordered list of visited pixels (including `start`), moving away
    from `start`. Used to estimate the local direction of a fragment.
    """
    visited = [start]
    seen = {start}
    cur = start
    for _ in range(max_steps):
        y, x = cur
        nxt = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if (0 <= ny < skel.shape[0] and 0 <= nx_ < skel.shape[1]
                        and skel[ny, nx_] and (ny, nx_) not in seen):
                    nxt = (ny, nx_)
                    break
            if nxt is not None:
                break
        if nxt is None:
            break
        visited.append(nxt)
        seen.add(nxt)
        cur = nxt
    return visited


def _tangent_direction(skel: np.ndarray, endpoint: tuple[int, int],
                       max_steps: int = TANGENT_STEPS) -> np.ndarray | None:
    """Unit vector (dy, dx) pointing in the direction the fragment would continue.

    Computed as (endpoint - farthest_walked_pixel), i.e. from the interior of the
    fragment outwards. Returns None if the fragment is too short to define a
    direction.
    """
    path = _walk_skeleton(skel, endpoint, max_steps)
    if len(path) < 2:
        return None
    far = np.array(path[-1], dtype=np.float64)
    near = np.array(path[0], dtype=np.float64)  # == endpoint
    v = near - far
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return None
    return v / norm


def _geodesic_path(vessel_mask: np.ndarray, p0: tuple[int, int], p1: tuple[int, int],
                   margin: int = GEODESIC_MARGIN):
    """Shortest path from p0 to p1 constrained to the vessel mask.

    Cost is 1 inside the vessel mask and GEODESIC_OFF_COST outside, computed on a
    local window around the two points to keep the search cheap.

    Returns (list of pixels, fraction of the path lying outside the vessel mask).
    The caller uses that fraction to decide whether to keep the geodesic route or
    fall back to a straight line.
    """
    y0, x0 = p0
    y1, x1 = p1
    ymin = max(0, min(y0, y1) - margin)
    ymax = min(vessel_mask.shape[0], max(y0, y1) + margin + 1)
    xmin = max(0, min(x0, x1) - margin)
    xmax = min(vessel_mask.shape[1], max(x0, x1) + margin + 1)
    sub_vessel = vessel_mask[ymin:ymax, xmin:xmax]
    costs = np.where(sub_vessel, 1.0, GEODESIC_OFF_COST).astype(np.float64)
    start = (y0 - ymin, x0 - xmin)
    end = (y1 - ymin, x1 - xmin)
    path, _ = route_through_array(costs, start, end, fully_connected=True)
    off_vessel = sum(1 for r, c in path if not sub_vessel[r, c])
    off_fraction = off_vessel / max(1, len(path))
    path_global = [(r + ymin, c + xmin) for r, c in path]
    return path_global, off_fraction


def reconnect_endpoints(binary_mask: np.ndarray,
                        distance_threshold: float,
                        max_angle_deg: float = MAX_ANGLE_DEG,
                        vessel_mask: np.ndarray | None = None) -> np.ndarray:
    """Reconnect orphan fragments of a single-class binary mask.

    Parameters
    ----------
    binary_mask
        2D boolean array for ONE class only (artery or vein, never both). Passing
        a merged mask would break the class-coherence constraint.
    distance_threshold
        Maximum Euclidean distance, in pixels, between an orphan endpoint and a
        candidate point of the main structure.
    max_angle_deg
        Half-angle of the acceptance cone around the estimated tangent.
    vessel_mask
        Optional boolean vessel mask (artery OR vein, BEFORE reconnection). If
        given, reconnections follow the shortest path constrained to this mask
        instead of a straight line, with automatic fallback to a straight line
        when the route strays too far off-vessel. Disabled by default.

    Returns
    -------
    Boolean array of the same shape, with plausible reconnections added.
    """
    if binary_mask.sum() == 0:
        return binary_mask
    # skeletonize() requires a C-contiguous array; slicing a 3-channel image
    # yields a strided view, so we normalise here rather than at every call site.
    binary_mask = np.ascontiguousarray(binary_mask)

    skel = skeletonize(binary_mask)
    labeled = label(skel, connectivity=2)
    if labeled.max() <= 1:
        # Single connected component: nothing is orphaned.
        return binary_mask

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    main_label = int(np.argmax(sizes))
    main_mask = labeled == main_label
    main_ys, main_xs = np.where(main_mask)
    if len(main_ys) == 0:
        return binary_mask
    main_pts = np.column_stack([main_ys, main_xs]).astype(np.float64)
    tree = cKDTree(main_pts)

    # Endpoints of every component EXCEPT the main structure.
    endpoints = find_endpoints(skel) & ~main_mask
    ep_ys, ep_xs = np.where(endpoints)
    if len(ep_ys) == 0:
        return binary_mask

    result = binary_mask.copy()
    dist_map = distance_transform_edt(binary_mask)
    cos_thresh = np.cos(np.radians(max_angle_deg))

    for ey, ex in zip(ep_ys, ep_xs):
        endpoint = (int(ey), int(ex))
        tangent = _tangent_direction(skel, endpoint)
        if tangent is None:
            continue

        idxs = tree.query_ball_point([ey, ex], distance_threshold)
        if not idxs:
            continue

        # Among candidates inside the directional cone, take the closest one.
        best_idx, best_dist = None, np.inf
        for i in idxs:
            ty, tx = main_pts[i]
            v = np.array([ty - ey, tx - ex])
            d = np.linalg.norm(v)
            if d < 1e-6:
                continue
            v_unit = v / d
            cos_angle = np.dot(tangent, v_unit)
            if cos_angle >= cos_thresh and d < best_dist:
                best_idx, best_dist = i, d

        if best_idx is None:
            continue  # no candidate inside the directional cone

        ty, tx = main_pts[best_idx]
        ty_i, tx_i = int(round(ty)), int(round(tx))

        path_pts = None
        if vessel_mask is not None:
            geo_path, off_fraction = _geodesic_path(vessel_mask, (int(ey), int(ex)), (ty_i, tx_i))
            if off_fraction <= GEODESIC_OFF_VESSEL_MAX:
                path_pts = geo_path
        if path_pts is None:
            rr, cc = line(int(ey), int(ex), ty_i, tx_i)
            path_pts = list(zip(rr, cc))

        # Dilate the drawn path to the local vessel radius so the added stroke
        # matches the calibre of the branch being reconnected.
        local_r = max(1, int(round(dist_map[ey, ex])))
        for r, c in path_pts:
            r0, r1 = max(0, r - local_r), min(result.shape[0], r + local_r + 1)
            c0, c1 = max(0, c - local_r), min(result.shape[1], c + local_r + 1)
            result[r0:r1, c0:c1] = True

    return result


def reconnect_av_pair(artery_bin: np.ndarray,
                      vein_bin: np.ndarray,
                      distance_threshold: float,
                      use_geodesic: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconnect artery and vein independently, then rebuild the vessel channel.

    This is the entry point the submission builder uses. It enforces constraints
    1 and 3 from the module docstring: the two classes never interact during
    reconnection, and the vessel channel is recomputed as their union afterwards
    so that `artery | vein == vessel` holds exactly.

    Returns
    -------
    (artery, vessel, vein) boolean arrays.
    """
    vessel_before = (artery_bin | vein_bin) if use_geodesic else None
    artery_out = reconnect_endpoints(artery_bin, distance_threshold, vessel_mask=vessel_before)
    vein_out = reconnect_endpoints(vein_bin, distance_threshold, vessel_mask=vessel_before)
    vessel_out = np.maximum(artery_out, vein_out)
    return artery_out, vessel_out, vein_out
