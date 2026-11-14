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

    k = max(1, min(n - 1, int(np.sqrt(n))))

    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.median(nearest, axis=1)

    valid_scale = np.isfinite(local_scale) & (local_scale > 0.0)
    scale_fallback = (
        float(np.median(local_scale[valid_scale]))
        if np.any(valid_scale)
        else fallback
    )
    local_scale = np.where(
        np.isfinite(local_scale),
        np.maximum(local_scale, eps),
        max(scale_fallback, eps),
    )

    candidate_order = np.argpartition(work, kth=k - 1, axis=1)[:, :k]
    candidate_mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    candidate_mask[rows, candidate_order] = True
    candidate_mask &= finite
    np.fill_diagonal(candidate_mask, False)

    endpoint_scale = local_scale[:, None] + local_scale[None, :] + eps
    locality = np.exp(-work / endpoint_scale)
    locality[~finite] = 0.0

    local_affinity = np.exp(-work / (local_scale[:, None] + eps))
    local_affinity[~candidate_mask] = 0.0

    row_mass = np.sum(local_affinity, axis=1, keepdims=True)
    row_mass = np.maximum(row_mass, eps)
    choice_probability = local_affinity / row_mass

    bottleneck = choice_probability + choice_probability.T
    bottleneck_max = np.max(bottleneck)
    if bottleneck_max > 0.0:
        bottleneck /= bottleneck_max

    reciprocal = candidate_mask & candidate_mask.T
    one_sided = candidate_mask & ~reciprocal

    finite_degree = np.sum(candidate_mask, axis=1)
    isolated_level = 1.0 / np.maximum(finite_degree.astype(float), 1.0)
    isolated_level /= max(float(np.max(isolated_level)), 1.0)

    endpoint_isolation = np.maximum(
        isolated_level[:, None],
        isolated_level[None, :],
    )

    bridge_strength = np.exp(
        -work / (np.maximum(local_scale[:, None], local_scale[None, :]) + eps)
    )
    bridge_strength[~finite] = 0.0

    score = locality * (0.62 + 0.38 * reciprocal.astype(float))
    score *= 0.78 + 0.22 * bottleneck

    restricted_bridge = (
        1.0
        + 0.24
        * one_sided.astype(float)
        * endpoint_isolation
        * bridge_strength
    )
    score *= restricted_bridge

    score[~candidate_mask] = 0.0
    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
