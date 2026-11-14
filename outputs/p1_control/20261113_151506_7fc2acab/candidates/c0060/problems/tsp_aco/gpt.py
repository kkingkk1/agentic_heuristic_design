import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return stochastic-sampling priors for directed TSP edges."""

    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    eps = np.finfo(float).eps

    finite_positive = np.isfinite(distances) & (distances > 0)
    if np.any(finite_positive):
        fallback = max(float(np.max(distances[finite_positive])), 1.0)
    else:
        fallback = 1.0

    valid = np.ones((n, n), dtype=bool)
    np.fill_diagonal(valid, False)

    d = np.where(
        np.isfinite(distances) & (distances >= 0),
        distances,
        fallback,
    )
    np.fill_diagonal(d, fallback)

    # Treat zero-cost edges as highly attractive but numerically stable.
    positive_scale = np.median(d[valid]) if np.any(valid) else 1.0
    positive_scale = max(float(positive_scale), eps)
    safe_d = np.maximum(d, eps * positive_scale)

    def normalize_rows(signal: np.ndarray) -> np.ndarray:
        signal = np.where(valid & np.isfinite(signal), signal, 0.0)
        row_max = np.max(signal, axis=1, keepdims=True)
        row_max = np.maximum(row_max, eps)
        return signal / row_max

    # Signal 1: direct short-edge preference.
    direct = normalize_rows(1.0 / safe_d)

    # Signal 2: local rank preference, robust to distance scale.
    ranks = np.zeros_like(d)
    for i in range(n):
        order = np.argsort(safe_d[i], kind="mergesort")
        order = order[order != i]
        ranks[i, order] = np.arange(order.size, dtype=float)
    rank_signal = normalize_rows(1.0 / (1.0 + ranks))

    # Signal 3: destination accessibility.
    # Destinations with several short outgoing alternatives are preferred.
    masked = np.where(valid, safe_d, np.inf)
    k = min(3, n - 1)
    nearest = np.partition(masked, kth=k - 1, axis=1)[:, :k]
    destination_scale = np.mean(nearest, axis=1)
    destination_scale = np.maximum(destination_scale, eps)

    accessibility = 1.0 / destination_scale[None, :]
    accessibility = normalize_rows(np.broadcast_to(accessibility, (n, n)))

    # Signal 4: two-edge lookahead.
    # Favor edges whose destination supports a short continuation.
    lookahead = 1.0 / (safe_d + destination_scale[None, :])
    lookahead = normalize_rows(lookahead)

    # Geometric fusion preserves complementary evidence.
    score = (
        np.power(direct + eps, 0.45)
        * np.power(rank_signal + eps, 0.20)
        * np.power(accessibility + eps, 0.15)
        * np.power(lookahead + eps, 0.20)
    )
    score = np.where(valid & np.isfinite(score), score, 0.0)

    row_sum = score.sum(axis=1, keepdims=True)
    score = np.divide(
        score,
        row_sum,
        out=np.zeros_like(score),
        where=row_sum > 0,
    )

    # Deterministic safety fallback for degenerate rows.
    for i in range(n):
        if not np.any(score[i] > 0):
            candidates = np.flatnonzero(valid[i])
            if candidates.size:
                best = candidates[np.argmin(safe_d[i, candidates])]
                score[i, best] = 1.0

    return score
