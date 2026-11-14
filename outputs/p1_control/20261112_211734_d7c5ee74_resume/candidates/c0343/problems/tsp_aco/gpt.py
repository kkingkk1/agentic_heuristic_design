import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = d[np.isfinite(d) & (d > 0)]
    fallback = (
        float(np.median(finite_positive))
        if finite_positive.size
        else 1.0
    )
    fallback = max(fallback, np.finfo(float).eps)

    d = np.where(np.isfinite(d), d, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    neighbors = n - 1
    k = max(1, min(neighbors, int(np.ceil(np.sqrt(n)))))

    ordered = np.argsort(work, axis=1, kind="mergesort")
    nearest_idx = ordered[:, :k]
    nearest = np.take_along_axis(work, nearest_idx, axis=1)

    local_scale = np.maximum(
        np.mean(nearest, axis=1),
        eps,
    )
    cutoff = np.maximum(
        nearest[:, -1],
        eps,
    )

    endpoint_scale = np.sqrt(
        np.maximum(
            local_scale[:, None] * local_scale[None, :],
            eps,
        )
    )
    normalized_distance = d / endpoint_scale

    attraction = np.exp(-np.minimum(normalized_distance, 12.0))

    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, ordered] = np.arange(neighbors, dtype=float)[None, :]

    mutual_rank = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )
    mutual_rank /= np.max(mutual_rank)

    candidate_strength = 0.5 * (
        np.exp(-np.minimum(ranks / max(float(k), 1.0), 12.0))
        + np.exp(-np.minimum(ranks.T / max(float(k), 1.0), 12.0))
    )

    relative_i = d / local_scale[:, None]
    relative_j = d / local_scale[None, :]

    endpoint_competition = 1.0 / np.sqrt(
        (1.0 + relative_i) * (1.0 + relative_j)
    )
    endpoint_competition = 0.82 + 0.18 * endpoint_competition

    regret_i = np.clip(
        (cutoff[:, None] - d) / (cutoff[:, None] + d + eps),
        0.0,
        1.0,
    )
    regret_j = np.clip(
        (cutoff[None, :] - d) / (cutoff[None, :] + d + eps),
        0.0,
        1.0,
    )
    regret = 0.5 * (regret_i + regret_j)
    regret_factor = 0.94 + 0.06 * regret

    affinity = np.exp(
        -np.minimum(work / local_scale[:, None], 12.0)
    )
    np.fill_diagonal(affinity, 0.0)

    row_norm = np.sqrt(
        np.maximum(np.sum(affinity * affinity, axis=1), eps)
    )
    compatibility = (affinity @ affinity.T) / (
        row_norm[:, None] * row_norm[None, :] + eps
    )
    compatibility = np.clip(compatibility, 0.0, 1.0)
    compatibility_factor = 0.96 + 0.04 * compatibility

    score = (
        attraction
        * (0.55 + 0.45 * mutual_rank)
        * (0.70 + 0.30 * candidate_strength)
        * endpoint_competition
        * regret_factor
        * compatibility_factor
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
