import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    valid = finite & (d >= 0.0)
    valid &= finite.T & (d.T >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    eps = np.finfo(float).eps

    positive = d[valid & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    work = np.where(valid, np.maximum(d, 0.0), np.inf)
    positive_work = np.where(valid, np.maximum(d, eps), np.inf)

    # Estimate a robust local scale without fully sorting every row.
    neighbor_count = min(max(2, int(np.sqrt(n)) + 1), max(n - 1, 1))
    local_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row = positive_work[i]
        finite_row = row[np.isfinite(row)]
        if finite_row.size:
            k = min(neighbor_count - 1, finite_row.size - 1)
            local_scale[i] = np.partition(finite_row, k)[k]

    local_scale = np.maximum(local_scale, eps)

    # Small nearest-neighbor candidate sets provide a cheap structural signal.
    candidate_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        row = positive_work[i]
        available = np.flatnonzero(np.isfinite(row))
        if available.size:
            k = min(neighbor_count, available.size)
            chosen = available[np.argpartition(row[available], k - 1)[:k]]
            candidate_mask[i, chosen] = True

    mutual_support = candidate_mask & candidate_mask.T
    one_sided_support = candidate_mask | candidate_mask.T

    normalized = work / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized = np.minimum(normalized, 1e6)

    distance_ratio = work / global_scale
    distance_ratio = np.minimum(distance_ratio, 1e6)

    # Direct scores avoid constructing a full rank matrix or repeated logs.
    score = (
        np.power(
            global_scale / np.maximum(work, eps),
            0.90,
        )
        * np.exp(-0.85 * normalized)
        * np.exp(-0.20 * distance_ratio)
    )

    # Preserve edges supported by both endpoints while retaining a softer
    # fallback for one-sided nearest-neighbor candidates.
    support_multiplier = (
        1.0
        + 0.55 * mutual_support.astype(float)
        + 0.15 * one_sided_support.astype(float)
    )
    score *= support_multiplier

    # Penalize edges joining regions with strongly different local scales.
    scale_gap = np.abs(
        np.log(
            np.maximum(local_scale[:, None], eps)
            / np.maximum(local_scale[None, :], eps)
        )
    )
    score *= np.exp(-0.12 * np.minimum(scale_gap, 20.0))

    score = np.where(valid, score, 0.0)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)

    return (score / maximum).astype(float, copy=False)
