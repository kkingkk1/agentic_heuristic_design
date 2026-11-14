import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    safe = np.where(valid, d, np.inf)

    positive_edges = d[valid & (d > 0)]
    global_scale = (
        float(np.median(positive_edges))
        if positive_edges.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    sorted_d = np.sort(safe, axis=1)
    k = min(max(2, int(np.sqrt(n))), n - 1)
    neighborhood = sorted_d[:, :k]

    positive_neighborhood = np.where(
        np.isfinite(neighborhood) & (neighborhood > 0),
        neighborhood,
        np.nan,
    )
    local_scale = np.nanmedian(positive_neighborhood, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    nearest = sorted_d[:, 0]
    nearest = np.where(np.isfinite(nearest), nearest, global_scale)

    if n > 2:
        second = sorted_d[:, 1]
        second = np.where(np.isfinite(second), second, nearest)
    else:
        second = nearest

    margin = np.maximum(second - nearest, 0.0)
    margin_scale = np.maximum(
        margin,
        0.15 * local_scale,
    )
    margin_scale = np.maximum(margin_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks[~valid] = np.inf

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    pair_scale = np.maximum(pair_scale, eps)

    local_attraction = np.exp(
        -np.clip(safe / pair_scale, 0.0, 50.0)
    )
    global_attraction = np.exp(
        -np.clip(safe / global_scale, 0.0, 50.0)
    )

    rank_support = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )
    rank_support[~np.isfinite(rank_support)] = 0.0

    mutual_nearest = (
        (ranks == 0)
        & (ranks.T == 0)
        & valid
        & valid.T
    ).astype(float)

    excess_i = np.maximum(safe - nearest[:, None], 0.0)
    excess_j = np.maximum(safe - nearest[None, :], 0.0)

    margin_i = np.exp(
        -np.clip(
            excess_i / margin_scale[:, None],
            0.0,
            50.0,
        )
    )
    margin_j = np.exp(
        -np.clip(
            excess_j / margin_scale[None, :],
            0.0,
            50.0,
        )
    )
    mutual_margin = np.sqrt(margin_i * margin_j)

    savings = np.maximum(
        nearest[:, None] + nearest[None, :] - safe,
        0.0,
    )
    positive_savings = savings[np.isfinite(savings) & (savings > 0)]
    savings_scale = (
        float(np.median(positive_savings))
        if positive_savings.size
        else global_scale
    )
    savings_scale = max(savings_scale, eps)
    savings_score = np.tanh(
        np.clip(savings / (1.25 * savings_scale), 0.0, 20.0)
    )

    mutual_local = np.sqrt(
        local_attraction * local_attraction.T
    )
    mutual_global = np.sqrt(
        global_attraction * global_attraction.T
    )

    score = (
        0.34 * mutual_local
        + 0.20 * mutual_global
        + 0.18 * rank_support
        + 0.12 * mutual_margin
        + 0.10 * savings_score
        + 0.06 * mutual_nearest
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
