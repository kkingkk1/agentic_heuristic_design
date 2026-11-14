import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must contain non-negative values")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros_like(distances, dtype=float)

    eps = np.finfo(float).eps
    d = distances.copy()
    np.fill_diagonal(d, np.inf)

    finite_edges = np.isfinite(d)
    positive_edges = finite_edges & (d > 0)
    global_scale = (
        float(np.median(d[positive_edges]))
        if np.any(positive_edges)
        else 1.0
    )
    global_scale = max(global_scale, eps)

    # Robust endpoint-local distance scales.
    ordered = np.sort(d, axis=1)
    k = max(1, min(n - 1, int(np.sqrt(n))))
    local_scale = np.mean(ordered[:, :k], axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    # Global and endpoint-local shortness.
    local_shortness = np.exp(
        -d / (pair_scale + eps)
    )
    global_shortness = np.exp(
        -d / global_scale
    )

    # Mutual local rank: both endpoints should prefer the edge.
    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks += 1.0
    rank_scale = max(1.0, np.sqrt(float(n)))
    mutual_rank = np.exp(
        -0.5 * (
            ranks / rank_scale
            + ranks.T / rank_scale
        )
    )

    # Endpoint support: connect nodes whose own neighborhoods are compact.
    nearest = np.min(d, axis=1)
    nearest = np.where(
        np.isfinite(nearest),
        nearest,
        global_scale,
    )
    support_scale = nearest[:, None] + nearest[None, :] + global_scale
    neighborhood_support = np.exp(-support_scale / (global_scale + eps))

    # Savings-style signal for edges shorter than two independent nearest links.
    savings = np.maximum(
        nearest[:, None] + nearest[None, :] - d,
        0.0,
    )
    positive_savings = savings[np.isfinite(savings) & (savings > 0)]
    savings_scale = (
        float(np.median(positive_savings))
        if positive_savings.size
        else global_scale
    )
    savings_scale = max(savings_scale, eps)
    savings_signal = np.tanh(savings / savings_scale)

    # Geometric combination preserves agreement across complementary signals.
    signals = np.stack(
        [
            np.maximum(local_shortness, eps),
            np.maximum(global_shortness, eps),
            np.maximum(mutual_rank, eps),
            np.maximum(neighborhood_support, eps),
            np.maximum(0.5 + 0.5 * savings_signal, eps),
        ],
        axis=0,
    )
    weights = np.array([0.30, 0.20, 0.25, 0.15, 0.10])
    score = np.exp(
        np.sum(weights[:, None, None] * np.log(signals), axis=0)
    )

    score[~finite_edges] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
