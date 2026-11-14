import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded edge priors for stochastic TSP solution sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = (
        np.isfinite(raw)
        & (raw >= 0.0)
        & ~np.eye(n, dtype=bool)
    )

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_values = raw[valid]
    positive = finite_values[finite_values > 0.0]

    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)

    clean = np.where(valid, raw, np.inf)
    positive_clean = clean[valid]

    # Robustly cap extreme but feasible distances so they do not dominate
    # the normalization of local signals.
    distance_cap = max(
        float(np.quantile(positive_clean, 0.98)),
        4.0 * scale,
    )
    bounded_distance = np.minimum(clean, distance_cap)

    temperature = max(0.90 * scale, eps)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(bounded_distance[valid] / temperature, 40.0)
    )

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (bounded_distance[valid] + eps)

    row_max = np.maximum(inverse.max(axis=1, keepdims=True), eps)
    inverse /= row_max

    # Stable local rank preserves strong nearest-neighbor evidence without
    # introducing hard candidate truncation.
    order = np.argsort(bounded_distance, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank[~valid] = 0.0

    rank_scale = max(1.0, 0.85 * np.sqrt(float(n)))
    rank_signal = np.zeros((n, n), dtype=float)
    rank_signal[valid] = np.exp(
        -rank[valid] / rank_scale
    )

    # Endpoint support scarcity supplies mild diversity pressure. It is
    # bounded so it cannot overwhelm distance affinity.
    support = affinity.sum(axis=1)
    mean_support = max(float(np.mean(support)), eps)

    scarcity = np.sqrt(mean_support / np.maximum(support, eps))
    scarcity = np.clip(scarcity, 0.72, 1.45)

    diversity = np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )
    diversity = np.clip(diversity, 0.72, 1.45)
    diversity[~valid] = 0.0

    # Local rank and affinity are the primary signals. The exploration floor
    # keeps feasible long edges sampleable while retaining a strong short-edge
    # preference.
    local_strength = (
        0.62 * affinity
        + 0.28 * inverse
        + 0.10 * rank_signal
    )

    score = 0.045 + local_strength * (
        1.0 + 0.14 * (diversity - 1.0)
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)

    score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
