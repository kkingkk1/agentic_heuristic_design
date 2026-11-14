import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros_like(distances, dtype=float)

    positive = distances[valid & (distances > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).tiny)

    eps = np.sqrt(np.finfo(float).eps)
    normalized = np.where(valid, distances / scale, np.inf)

    # Compress the dynamic range so a single extremely short edge
    # does not completely dominate stochastic sampling.
    compressed = np.sqrt(np.maximum(normalized, 0.0) + eps)
    base = np.where(valid, 1.0 / (1.0 + compressed), 0.0)

    row_best = np.max(base, axis=1, keepdims=True)
    col_best = np.max(base, axis=0, keepdims=True)

    row_competitiveness = np.divide(
        base,
        row_best,
        out=np.zeros_like(base),
        where=row_best > 0.0,
    )
    col_competitiveness = np.divide(
        base,
        col_best,
        out=np.zeros_like(base),
        where=col_best > 0.0,
    )

    # Rank bonus favors edges that are consistently near the local best
    # in both their outgoing and incoming neighborhoods.
    row_order = np.argsort(np.where(valid, normalized, np.inf), axis=1)
    col_order = np.argsort(np.where(valid, normalized, np.inf), axis=0)

    row_rank = np.empty_like(row_order, dtype=float)
    col_rank = np.empty_like(col_order, dtype=float)

    row_rank[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n, dtype=float)

    row_rank_score = np.where(
        valid,
        1.0 - row_rank / max(n - 1, 1),
        0.0,
    )
    col_rank_score = np.where(
        valid,
        1.0 - col_rank / max(n - 1, 1),
        0.0,
    )

    priors = base * (
        0.46
        + 0.24 * row_competitiveness
        + 0.18 * col_competitiveness
        + 0.06 * row_rank_score
        + 0.06 * col_rank_score
    )

    priors[~valid] = 0.0

    maximum = np.max(priors)
    if maximum > 0.0 and np.isfinite(maximum):
        priors /= maximum

    return priors
