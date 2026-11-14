import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for a TSP distance matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")

    d = np.maximum(distances, 0.0).copy()
    np.fill_diagonal(d, np.inf)

    finite_positive = d[np.isfinite(d) & (d > 0)]
    global_scale = (
        float(np.median(finite_positive))
        if finite_positive.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    # Use several nearest-neighbor distances to obtain a robust local scale.
    k = min(max(2, int(np.sqrt(n))), n - 1)
    sorted_d = np.sort(d, axis=1)
    local_block = sorted_d[:, :k]

    finite_block = np.where(np.isfinite(local_block), local_block, np.nan)
    local_scale = np.nanmedian(finite_block, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    # Pairwise scale adapts the distance kernel to heterogeneous densities.
    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, np.finfo(float).eps)

    normalized_distance = d / pair_scale
    distance_signal = np.exp(-np.square(normalized_distance))
    distance_signal[~np.isfinite(distance_signal)] = 0.0

    # Smooth rank signal with explicit reciprocal-neighbor agreement.
    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks[np.arange(n), np.arange(n)] = np.inf

    rank_signal = 1.0 / np.power(1.0 + ranks, 0.70)
    rank_signal[~np.isfinite(rank_signal)] = 0.0

    mutual_rank = np.sqrt(rank_signal * rank_signal.T)
    mutual_rank = np.square(mutual_rank)

    # Reciprocal neighborhood membership gives a stronger bonus to shared edges.
    neighborhood_width = min(max(3, k), n - 1)
    knn_mask = ranks <= neighborhood_width
    reciprocal = knn_mask & knn_mask.T
    reciprocal_bonus = reciprocal.astype(float)

    # Clarke-Wright-style savings relative to each endpoint's local baseline.
    nearest = sorted_d[:, 0]
    nearest = np.where(
        np.isfinite(nearest) & (nearest >= 0),
        nearest,
        global_scale,
    )

    savings = nearest[:, None] + nearest[None, :] - d
    savings = np.maximum(savings, 0.0)
    positive_savings = savings[savings > 0]

    savings_scale = (
        float(np.percentile(positive_savings, 70))
        if positive_savings.size
        else global_scale
    )
    savings_scale = max(savings_scale, np.finfo(float).eps)

    savings_signal = np.tanh(savings / (1.25 * savings_scale))
    savings_signal[~np.isfinite(savings_signal)] = 0.0

    # Competition rewards edges that are simultaneously strong in both rows.
    row_strength = rank_signal * distance_signal
    competition = np.sqrt(row_strength * row_strength.T)

    # Mild isolation correction favors useful edges from sparse nodes without
    # overwhelming the local distance and rank evidence.
    isolation = np.sqrt(
        local_scale[:, None] / (local_scale[None, :] + np.finfo(float).eps)
    )
    isolation = 0.5 * (isolation + isolation.T)
    isolation = np.clip(isolation, 0.75, 1.25)

    score = (
        0.42 * distance_signal
        + 0.30 * mutual_rank
        + 0.14 * reciprocal_bonus
        + 0.09 * competition
        + 0.05 * savings_signal
    )
    score *= isolation

    score = np.maximum(score, 0.0)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0,
    )

    # Restore symmetry after row normalization while retaining bounded values.
    score = 0.5 * (score + score.T)
    score = np.clip(score, 0.0, 1.0)
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
