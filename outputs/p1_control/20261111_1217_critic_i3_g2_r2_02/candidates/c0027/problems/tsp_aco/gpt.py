import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros_like(d, dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(d)

    if not np.all(finite | diagonal):
        raise ValueError(
            "distance_matrix contains invalid non-finite off-diagonal values"
        )

    work = np.where(finite, np.maximum(d, 0.0), np.inf)
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(1.0, fallback)

    finite_values = work[np.isfinite(work)]
    scale = float(np.median(finite_values)) if finite_values.size else fallback
    scale = max(scale, eps)

    affinity = np.zeros((n, n), dtype=float)
    valid_edges = np.isfinite(work)
    affinity[valid_edges] = np.exp(-work[valid_edges] / scale)
    np.fill_diagonal(affinity, 0.0)

    degree_mass = np.sum(affinity, axis=1)
    degree_mass = np.maximum(degree_mass, eps)

    soft_neighbors = affinity / degree_mass[:, None]

    scarcity = 1.0 / np.sqrt(
        degree_mass[:, None] * degree_mass[None, :]
    )

    neighborhood_overlap = soft_neighbors @ soft_neighbors.T
    overlap_scale = np.max(neighborhood_overlap)
    if overlap_scale > 0.0:
        normalized_overlap = neighborhood_overlap / overlap_scale
    else:
        normalized_overlap = np.zeros_like(neighborhood_overlap)

    bridge_factor = 1.0 + 0.35 * (1.0 - normalized_overlap)

    detour = np.full((n, n), np.inf, dtype=float)
    for k in range(n):
        through_k = work[:, k, None] + work[k, None, :]
        detour = np.minimum(detour, through_k)

    detour_advantage = np.zeros_like(work)
    detour_valid = np.isfinite(detour) & np.isfinite(work)
    if np.any(detour_valid):
        direct = work[detour_valid]
        alternative = detour[detour_valid]
        gain = alternative - direct
        denominator = alternative + direct + eps
        detour_advantage[detour_valid] = np.clip(
            gain / denominator,
            0.0,
            1.0,
        )

    necessity_factor = 1.0 + 0.80 * detour_advantage

    score = affinity
    score *= scarcity
    score *= bridge_factor
    score *= necessity_factor

    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
