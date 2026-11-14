import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    positive = d[finite & (d > 0)]
    fallback = float(np.max(positive)) if positive.size else 1.0

    d = np.where(finite, d, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps

    neighborhood = max(
        1,
        min(n - 1, int(np.ceil(np.sqrt(n)))),
    )

    nearest = np.partition(
        work,
        neighborhood - 1,
        axis=1,
    )[:, :neighborhood]

    local_scale = np.maximum(
        np.mean(nearest, axis=1),
        eps,
    )

    normalized = 0.5 * (
        d / local_scale[:, None]
        + d / local_scale[None, :]
    )

    score = 1.0 / (1.0 + normalized)

    candidate_count = max(
        1,
        min(n - 1, int(np.ceil(0.20 * (n - 1)))),
    )

    candidate_order = np.argpartition(
        work,
        candidate_count - 1,
        axis=1,
    )[:, :candidate_count]

    candidate_mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    candidate_mask[rows, candidate_order] = True
    np.fill_diagonal(candidate_mask, False)

    mutual_candidate = candidate_mask & candidate_mask.T

    local_rank = np.full((n, n), np.inf, dtype=float)

    for i in range(n):
        candidates = candidate_order[i]
        candidates = candidates[candidates != i]

        if candidates.size == 0:
            continue

        ordered = candidates[
            np.argsort(work[i, candidates], kind="mergesort")
        ]

        local_rank[i, ordered] = np.arange(
            ordered.size,
            dtype=float,
        )

    if candidate_count == 1:
        rank_fraction = np.zeros((n, n), dtype=float)
    else:
        rank_fraction = np.clip(
            local_rank / float(candidate_count - 1),
            0.0,
            1.0,
        )

    reciprocal_strength = np.sqrt(
        np.clip(1.0 - rank_fraction, 0.0, 1.0)
        * np.clip(1.0 - rank_fraction.T, 0.0, 1.0)
    )

    rank_agreement = 1.0 / (
        1.0 + np.abs(rank_fraction - rank_fraction.T)
    )

    mutual_bonus = (
        0.08
        * reciprocal_strength
        * rank_agreement
        * mutual_candidate
    )

    score *= 1.0 + mutual_bonus

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    values = score[off_diagonal]

    if values.size == 0:
        return np.zeros((n, n), dtype=float)

    low = float(np.min(values))
    high = float(np.max(values))

    if high > low:
        score = (score - low) / (high - low)
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
