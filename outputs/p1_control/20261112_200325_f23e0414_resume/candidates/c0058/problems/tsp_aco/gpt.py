import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)

    finite = np.isfinite(d)
    safe = np.where(finite, np.maximum(d, 0.0), np.inf)
    np.fill_diagonal(safe, np.inf)

    valid_values = safe[np.isfinite(safe)]
    if valid_values.size == 0:
        return np.zeros_like(safe, dtype=float)

    positive = valid_values[valid_values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    row_min = np.min(safe, axis=1, keepdims=True)
    row_min = np.where(np.isfinite(row_min), row_min, scale)

    row_median = np.median(
        np.where(np.isfinite(safe), safe, row_min),
        axis=1,
        keepdims=True,
    )
    local_scale = np.maximum(row_median - row_min, scale * 0.05)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    reciprocal = np.divide(
        scale,
        safe + scale * 1e-9,
        out=np.zeros_like(safe),
        where=np.isfinite(safe),
    )

    closeness = np.exp(
        -np.maximum(safe - row_min, 0.0) / local_scale
    )
    closeness[~finite] = 0.0

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    columns = np.broadcast_to(
        np.arange(n, dtype=int),
        (n, n),
    )
    ranks[order, columns] = np.arange(n, dtype=float)[:, None]
    ranks[~finite] = float(n)
    rank_score = 1.0 / (1.0 + ranks)

    nearest = np.argmin(safe, axis=1)
    mutual = (
        (nearest[nearest] == np.arange(n))
        & (nearest != np.arange(n))
    ).astype(float)
    mutual_bonus = np.zeros((n, n), dtype=float)
    mutual_bonus[np.arange(n), nearest] = mutual

    def row_normalize(values: np.ndarray) -> np.ndarray:
        maximum = np.max(values, axis=1, keepdims=True)
        return np.divide(
            values,
            np.where(maximum > 0.0, maximum, 1.0),
        )

    score = (
        0.55 * row_normalize(reciprocal)
        + 0.25 * row_normalize(closeness)
        + 0.12 * row_normalize(rank_score)
        + 0.08 * mutual_bonus
    )

    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
