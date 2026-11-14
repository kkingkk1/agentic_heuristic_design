import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0
    d = np.where(np.isfinite(d), d, fallback)

    positive = finite[finite > 0.0]
    if positive.size:
        global_scale = float(np.percentile(positive, 60.0))
    else:
        global_scale = fallback
    global_scale = max(global_scale, np.finfo(float).eps)

    k = min(n - 1, max(2, int(np.sqrt(n)) + 1))
    order = np.argsort(d, axis=1, kind="stable")
    nearest = np.take_along_axis(d, order[:, :k], axis=1)

    weights = np.exp(-np.linspace(0.0, 1.4, k))
    local_scale = (
        np.sum(nearest * weights[None, :], axis=1)
        / np.sum(weights)
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative = d / pair_scale

    distance_kernel = np.exp(-np.clip(relative ** 0.9, 0.0, 50.0))

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(float(k - 1), 1.0)
    local_rank_kernel = np.exp(
        -np.minimum(ranks / rank_scale, 50.0)
    )

    endpoint_ratio_i = d / local_scale[:, None]
    endpoint_ratio_j = d / local_scale[None, :]
    endpoint_kernel = np.exp(
        -np.clip(
            0.5 * (
                np.log1p(endpoint_ratio_i)
                + np.log1p(endpoint_ratio_j)
            ),
            0.0,
            50.0,
        )
    )

    temperature = np.maximum(
        np.median(nearest, axis=1, keepdims=True),
        np.finfo(float).eps,
    )
    soft_affinity = np.exp(
        -np.clip(d / temperature, 0.0, 50.0)
    )
    np.fill_diagonal(soft_affinity, 0.0)

    soft_sum = np.sum(soft_affinity, axis=1, keepdims=True)
    soft_probability = soft_affinity / np.maximum(
        soft_sum,
        np.finfo(float).eps,
    )

    two_sided_probability = np.sqrt(
        soft_probability * soft_probability.T
    )

    mutual_neighbor = (
        (ranks < float(k)).astype(float)
        * (ranks.T < float(k)).astype(float)
    )
    mutual_bonus = 1.0 + 0.3 * mutual_neighbor

    global_kernel = np.exp(
        -np.clip((d / global_scale) ** 0.75, 0.0, 50.0)
    )
    global_adjustment = 0.65 + 0.35 * global_kernel

    score = (
        distance_kernel ** 1.45
        * local_rank_kernel ** 0.7
        * endpoint_kernel ** 1.15
        * (0.55 + 0.45 * two_sided_probability)
        * mutual_bonus
        * global_adjustment
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
