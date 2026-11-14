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

    valid = ~np.eye(n, dtype=bool)
    usable = np.isfinite(distances) & (distances >= 0)

    positive = usable & (distances > 0)
    fallback = float(np.max(distances[positive])) if np.any(positive) else 1.0
    fallback = max(fallback, 1.0)

    d = np.where(usable, distances, fallback)
    np.fill_diagonal(d, fallback)

    eps = np.finfo(float).eps

    def normalize_rows(signal: np.ndarray) -> np.ndarray:
        signal = np.where(valid & np.isfinite(signal), signal, 0.0)
        scale = np.max(signal, axis=1, keepdims=True)
        scale = np.maximum(scale, eps)
        return signal / scale

    # Signal 1: direct edge proximity.
    direct = 1.0 / (d + eps)

    # Signal 2: stable local rank preference.
    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    row_ids = np.arange(n)[:, None]
    ranks[row_ids, order] = np.arange(n, dtype=float)[None, :]
    rank_signal = 1.0 / (1.0 + ranks)

    # Signal 3: destination-aware two-edge lookahead.
    # For each destination j, retain its two best outgoing alternatives.
    masked_d = np.where(valid, d, np.inf)

    if n >= 3:
        nearest_order = np.argsort(masked_d, axis=1, kind="mergesort")
        nearest_idx = nearest_order[:, 0]
        nearest_value = masked_d[np.arange(n), nearest_idx]
        second_value = masked_d[np.arange(n), nearest_order[:, 1]]

        continuation = np.broadcast_to(nearest_value[None, :], (n, n)).copy()
        source_ids = np.arange(n)[:, None]
        destination_ids = np.arange(n)[None, :]
        source_is_nearest = nearest_idx[None, :] == source_ids
        continuation = np.where(
            source_is_nearest,
            np.broadcast_to(second_value[None, :], (n, n)),
            continuation,
        )
    else:
        continuation = np.broadcast_to(d.T, (n, n)).copy()

    lookahead = 1.0 / (d + continuation + eps)

    # Mild accessibility correction: avoid concentrating probability on
    # destinations whose outgoing neighborhood is unusually expensive.
    outgoing_scale = np.min(masked_d, axis=1)
    accessibility = 1.0 / (outgoing_scale + eps)
    accessibility /= max(float(np.max(accessibility)), eps)
    accessibility_signal = np.broadcast_to(accessibility[None, :], (n, n))

    a = normalize_rows(direct)
    b = normalize_rows(rank_signal)
    c = normalize_rows(lookahead)
    e = np.where(valid, accessibility_signal, 0.0)

    score = (
        np.power(a + eps, 0.50)
        * np.power(b + eps, 0.20)
        * np.power(c + eps, 0.20)
        * np.power(e + eps, 0.10)
    )
    score = np.where(valid & np.isfinite(score), score, 0.0)

    row_sum = score.sum(axis=1, keepdims=True)
    score = np.divide(
        score,
        row_sum,
        out=np.zeros_like(score),
        where=row_sum > 0,
    )

    for i in range(n):
        if not np.any(score[i] > 0):
            candidates = np.flatnonzero(valid[i])
            if candidates.size:
                best = candidates[np.argmin(d[i, candidates])]
                score[i, best] = 1.0

    np.fill_diagonal(score, 0.0)
    return score
