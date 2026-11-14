import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(
        d,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d) & (d > 0.0)
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = d[valid]
    eps = np.finfo(float).eps
    global_scale = max(float(np.median(positive)), eps)

    largest = max(float(np.max(positive)), global_scale)
    replacement = min(
        largest * (n + 1.0),
        np.finfo(float).max / 4.0,
    )
    replacement = max(replacement, global_scale)

    work = np.where(valid, d, replacement)
    np.fill_diagonal(work, replacement)

    # Each row defines a soft distribution over its preferred edges.
    logits = -work / global_scale
    logits[~valid] = -np.inf

    row_max = np.max(logits, axis=1, keepdims=True)
    shifted = logits - row_max
    weights = np.zeros_like(work, dtype=float)

    finite_logits = np.isfinite(shifted)
    weights[finite_logits] = np.exp(
        np.clip(shifted[finite_logits], -745.0, 0.0)
    )

    row_sum = np.sum(weights, axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, eps)
    probabilities = weights / row_sum

    mutual_preference = np.sqrt(
        probabilities * probabilities.T
    )

    # Estimate the shortest two-edge replacement path for every edge.
    replacement_path = np.full((n, n), np.inf, dtype=float)

    for intermediate in range(n):
        candidate = (
            work[:, intermediate, None]
            + work[None, intermediate, :]
        )
        candidate[intermediate, :] = np.inf
        candidate[:, intermediate] = np.inf
        candidate[np.arange(n), np.arange(n)] = np.inf
        replacement_path = np.minimum(
            replacement_path,
            candidate,
        )

    direct = np.where(valid, d, replacement)
    detour_margin = replacement_path - direct
    detour_margin = np.maximum(detour_margin, 0.0)

    # Saturated margin gives a bounded bonus to non-redundant edges.
    nonredundancy = 1.0 - np.exp(
        -detour_margin / (global_scale + eps)
    )
    nonredundancy = np.where(
        np.isfinite(nonredundancy),
        nonredundancy,
        1.0,
    )

    global_preference = np.exp(
        -0.35 * direct / (global_scale + eps)
    )

    score = (
        mutual_preference
        * (0.5 + 0.5 * nonredundancy)
        * global_preference
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    score[~np.isfinite(score)] = 0.0
    return score.astype(float, copy=False)
