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

    # Sparse local candidate set. argpartition avoids full row-wise sorting.
    k = max(1, min(n - 1, int(np.sqrt(n)) + 1))
    candidate_idx = np.argpartition(work, kth=k - 1, axis=1)[:, :k]

    candidate_mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    candidate_mask[rows, candidate_idx] = True
    candidate_mask |= candidate_mask.T
    np.fill_diagonal(candidate_mask, False)

    nearest = np.take_along_axis(work, candidate_idx, axis=1)
    local_scale = np.median(nearest, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        fallback,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = work / (pair_scale + eps)

    endpoint_density = np.minimum(
        local_scale[:, None] / (local_scale[None, :] + eps),
        local_scale[None, :] / (local_scale[:, None] + eps),
    )

    savings = (
        local_scale[:, None]
        + local_scale[None, :]
        - work
    ) / (local_scale[:, None] + local_scale[None, :] + eps)
    savings = 0.75 + 0.25 * np.clip(savings, 0.0, 1.0)

    # Low-cost fallback for non-candidate edges.
    score = 0.035 / (1.0 + normalized_distance)
    score *= 0.90 + 0.10 * endpoint_density

    # More expressive scoring is applied only to likely useful edges.
    candidate_distance = np.where(candidate_mask, normalized_distance, 0.0)
    locality = np.exp(-candidate_distance)

    row_competition = np.exp(
        -0.8 * work / (local_scale[:, None] + eps)
    )
    col_competition = np.exp(
        -0.8 * work / (local_scale[None, :] + eps)
    )
    mutual_competition = np.sqrt(row_competition * col_competition)

    isolation_bridge = 1.0 + 0.30 * (
        1.0 - endpoint_density
    ) * np.exp(-0.5 * candidate_distance)

    candidate_score = locality
    candidate_score *= 0.55 + 0.45 * mutual_competition
    candidate_score *= isolation_bridge
    candidate_score *= savings

    score = np.where(candidate_mask, candidate_score, score)
    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
