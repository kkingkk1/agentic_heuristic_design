import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for a TSP distance matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")

    d = np.maximum(0.5 * (distances + distances.T), 0.0)
    np.fill_diagonal(d, np.inf)

    finite_offdiag = d[np.isfinite(d)]
    positive = finite_offdiag[finite_offdiag > 0]
    base_scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps
    base_scale = max(base_scale, eps)

    k = min(4, n - 1)
    sorted_d = np.sort(d, axis=1)
    local_scale = sorted_d[:, k - 1]
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        base_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    density = base_scale / local_scale
    density = np.clip(density, 0.35, 2.5)
    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, eps)

    temperature = 0.72 * pair_scale
    distance_score = np.exp(-d / temperature)
    distance_score[~np.isfinite(distance_score)] = 0.0

    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks[np.arange(n), np.arange(n)] = n

    rank_score = np.power(1.0 + ranks, -1.15)
    rank_score[np.arange(n), np.arange(n)] = 0.0

    mutual_rank = np.sqrt(rank_score * rank_score.T)
    mutual_distance = np.sqrt(distance_score * distance_score.T)

    neighborhood = distance_score / np.maximum(
        np.sum(distance_score, axis=1, keepdims=True),
        eps,
    )
    shared_neighborhood = neighborhood @ neighborhood.T
    shared_neighborhood /= np.maximum(
        np.sqrt(
            np.sum(neighborhood * neighborhood, axis=1)[:, None]
            * np.sum(neighborhood * neighborhood, axis=1)[None, :]
        ),
        eps,
    )
    np.fill_diagonal(shared_neighborhood, 0.0)

    competition = rank_score * distance_score
    competition = competition / np.maximum(
        np.sum(competition, axis=1, keepdims=True),
        eps,
    )
    mutual_competition = np.sqrt(competition * competition.T)

    nearest = np.min(d, axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, base_scale)

    savings = nearest[:, None] + nearest[None, :] - d
    savings = np.maximum(savings, 0.0)
    positive_savings = savings[savings > 0]
    savings_scale = (
        float(np.percentile(positive_savings, 65))
        if positive_savings.size
        else base_scale
    )
    savings_scale = max(savings_scale, eps)
    savings_score = np.tanh(savings / (1.25 * savings_scale))

    density_pair = np.sqrt(density[:, None] * density[None, :])
    density_pair = np.clip(density_pair, 0.5, 2.0)
    adaptive_distance = np.clip(mutual_distance * density_pair, 0.0, 1.0)

    score = (
        0.38 * adaptive_distance
        + 0.27 * mutual_rank
        + 0.17 * shared_neighborhood
        + 0.10 * mutual_competition
        + 0.08 * savings_score
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0,
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
