import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge priorities based on two-hop geometric indispensability."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    positive = distances[valid & (distances > 0.0)]
    eps = np.finfo(float).eps

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    work = np.where(valid, distances, np.inf)
    np.fill_diagonal(work, np.inf)

    if n == 2:
        base = np.divide(
            1.0,
            np.maximum(work, eps),
            out=np.zeros_like(work),
            where=valid,
        )
        maximum = np.max(base)
        if np.isfinite(maximum) and maximum > 0.0:
            base /= maximum
        return np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)

    best_two_hop = np.full((n, n), np.inf, dtype=float)

    for middle in range(n):
        left = work[:, middle][:, None]
        right = work[middle, :][None, :]
        candidate = left + right

        candidate[:, middle] = np.inf
        candidate[middle, :] = np.inf

        best_two_hop = np.minimum(best_two_hop, candidate)

    direct = work

    finite_detour = np.isfinite(best_two_hop) & valid
    margin = np.zeros_like(direct, dtype=float)

    margin[finite_detour] = (
        best_two_hop[finite_detour] - direct[finite_detour]
    )

    scaled_margin = margin / max(scale, eps)

    indispensable = 0.5 + 0.5 * np.tanh(
        np.clip(scaled_margin, -12.0, 12.0)
    )

    direct_scale = np.divide(
        direct,
        max(scale, eps),
        out=np.full_like(direct, np.inf),
        where=valid,
    )

    absolute_support = np.exp(
        -np.clip(direct_scale, 0.0, 60.0)
    )

    scores = indispensable * absolute_support
    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
