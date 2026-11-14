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
    valid = ~np.eye(n, dtype=bool)

    finite_positive = np.isfinite(distances) & (distances > 0)
    fallback = (
        float(np.max(distances[finite_positive]))
        if np.any(finite_positive)
        else 1.0
    )
    fallback = max(fallback, 1.0)

    d = np.where(
        np.isfinite(distances) & (distances >= 0),
        distances,
        fallback,
    )
    np.fill_diagonal(d, fallback)

    def normalize_rows(signal: np.ndarray) -> np.ndarray:
        signal = np.where(valid & np.isfinite(signal), signal, 0.0)
        maxima = np.max(signal, axis=1, keepdims=True)
        maxima = np.maximum(maxima, eps)
        return signal / maxima

    # Direct preference for short edges.
    inverse_distance = 1.0 / (d + eps)

    # Relative preference within each source node's candidate set.
    source_scale = np.partition(
        np.where(valid, d, np.inf),
        kth=min(2, n - 2),
        axis=1,
    )[:, min(2, n - 2)]
    source_scale = np.maximum(source_scale, eps)
    relative_edge = 1.0 / (1.0 + d / source_scale[:, None])

    # Destination continuation quality: destinations with short exits
    # are preferred because they preserve future construction options.
    continuation_scale = np.partition(
        np.where(valid, d, np.inf),
        kth=min(2, n - 2),
        axis=1,
    )[:, min(2, n - 2)]
    continuation_scale = np.maximum(continuation_scale, eps)
    destination_quality = 1.0 / (continuation_scale + eps)
    destination_quality = np.broadcast_to(
        destination_quality[None, :],
        (n, n),
    )

    # Bidirectional neighborhood compatibility.
    neighborhood_width = min(3, n - 1)
    nearest = np.argsort(d, axis=1, kind="stable")[:, :neighborhood_width]

    neighborhood_mask = np.zeros((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    neighborhood_mask[rows, nearest] = 1.0
    np.fill_diagonal(neighborhood_mask, 0.0)

    incoming_compatibility = neighborhood_mask.T @ neighborhood_mask
    incoming_compatibility /= max(float(neighborhood_width), 1.0)
    incoming_compatibility = 1.0 + incoming_compatibility

    a = normalize_rows(inverse_distance)
    b = normalize_rows(relative_edge)
    c = normalize_rows(destination_quality)
    e = normalize_rows(incoming_compatibility)

    score = (
        np.power(a + eps, 0.45)
        * np.power(b + eps, 0.25)
        * np.power(c + eps, 0.20)
        * np.power(e + eps, 0.10)
    )
    score = np.where(valid & np.isfinite(score), score, 0.0)

    row_sums = score.sum(axis=1, keepdims=True)
    score = np.divide(
        score,
        row_sums,
        out=np.zeros_like(score),
        where=row_sums > 0,
    )

    for i in range(n):
        if not np.any(score[i] > 0):
            candidates = np.flatnonzero(valid[i])
            if candidates.size:
                best = candidates[np.argmin(d[i, candidates])]
                score[i, best] = 1.0

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
