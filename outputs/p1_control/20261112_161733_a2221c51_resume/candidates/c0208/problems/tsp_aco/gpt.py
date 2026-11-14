import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, entropy-balanced edge-promise matrix for TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    eps = np.finfo(float).eps

    positive = d[valid & (d > 0.0)]
    if positive.size:
        temperature = max(float(np.median(positive)), eps)
    else:
        temperature = 1.0

    scaled_distance = d / temperature
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        affinity = np.exp(-np.minimum(scaled_distance, 60.0))

    affinity = np.where(valid, affinity, 0.0)
    np.fill_diagonal(affinity, 0.0)

    active = np.sum(affinity, axis=1) > 0.0
    if not np.any(active):
        return np.zeros((n, n), dtype=float)

    scaling = np.where(active, 1.0, 0.0)

    for _ in range(48):
        denominator = affinity @ scaling
        updated = np.zeros_like(scaling)
        usable = active & (denominator > eps)
        updated[usable] = 1.0 / denominator[usable]

        if np.allclose(updated, scaling, rtol=1e-7, atol=1e-12):
            scaling = updated
            break

        scaling = updated

    balanced = (
        scaling[:, None]
        * affinity
        * scaling[None, :]
    )
    balanced = np.where(valid, balanced, 0.0)

    row_mass = np.sum(affinity, axis=1)
    transition = np.zeros_like(affinity)
    usable_rows = row_mass > eps
    transition[usable_rows] = (
        affinity[usable_rows]
        / row_mass[usable_rows, None]
    )

    two_step = transition @ transition
    two_step = np.maximum(two_step, 0.0)

    direct_probability = np.zeros_like(affinity)
    direct_probability[usable_rows] = (
        transition[usable_rows]
    )

    replaceability = two_step / (
        two_step + direct_probability + eps
    )
    distinctiveness = 1.0 - np.clip(replaceability, 0.0, 1.0)

    score = balanced * (
        0.70 + 0.30 * distinctiveness
    )

    score = np.where(valid, score, 0.0)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
