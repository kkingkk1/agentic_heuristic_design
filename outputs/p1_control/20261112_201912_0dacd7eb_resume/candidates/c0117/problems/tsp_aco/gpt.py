import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust multi-scale edge-priority indicators for stochastic TSP sampling."""
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
    np.fill_diagonal(d, np.inf)

    positive = finite[finite > 0.0]
    if positive.size:
        global_scale = float(np.percentile(positive, 50.0))
        global_q75 = float(np.percentile(positive, 75.0))
    else:
        global_scale = fallback
        global_q75 = fallback

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)
    global_q75 = max(global_q75, global_scale)

    k = min(n - 1, max(3, int(np.sqrt(n)) + 2))
    sorted_d = np.sort(d, axis=1)
    nearest = sorted_d[:, :k]

    weights = np.linspace(1.0, 0.45, k, dtype=float)
    weights /= np.sum(weights)

    local_scale = np.sum(nearest * weights[None, :], axis=1)
    local_scale = np.maximum(local_scale, eps)

    local_median = np.median(nearest, axis=1)
    local_median = np.maximum(local_median, eps)

    local_pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    median_pair_scale = np.sqrt(
        local_median[:, None] * local_median[None, :]
    )

    ratio_i = d / local_scale[:, None]
    ratio_j = d / local_scale[None, :]
    pair_ratio = d / local_pair_scale

    adaptive_decay = np.exp(
        -np.clip(pair_ratio ** 0.82, 0.0, 45.0)
    )

    median_decay = np.exp(
        -np.clip((d / median_pair_scale) ** 0.95, 0.0, 45.0)
    )

    row_order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(row_order, dtype=float)
    ranks[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    normalized_rank = ranks / max(float(n - 1), 1.0)
    rank_i = 1.0 / (1.0 + 2.2 * normalized_rank)
    rank_j = 1.0 / (1.0 + 2.2 * normalized_rank.T)

    mutual_nearest = (
        (normalized_rank <= 0.20)
        & (normalized_rank.T <= 0.20)
    ).astype(float)

    mutual_local = (
        (normalized_rank <= 0.40)
        & (normalized_rank.T <= 0.40)
    ).astype(float)

    rank_term = (
        np.sqrt(rank_i * rank_j)
        * (1.0 + 0.28 * mutual_local + 0.22 * mutual_nearest)
    )

    mismatch = np.abs(np.log1p(ratio_i) - np.log1p(ratio_j))
    compatibility = np.exp(
        -np.clip(
            0.72 * (np.log1p(ratio_i) + np.log1p(ratio_j))
            + 0.45 * mismatch,
            0.0,
            45.0,
        )
    )

    global_term = np.exp(
        -np.clip((d / global_scale) ** 0.88, 0.0, 45.0)
    )
    global_term = 0.30 + 0.70 * global_term

    density_i = local_scale / global_scale
    density_j = local_scale / global_scale
    density_balance = np.sqrt(
        (1.0 + density_i[:, None]) / (1.0 + density_j[None, :])
    )
    density_balance = np.exp(
        -0.35 * np.abs(np.log(np.maximum(density_balance, eps)))
    )

    robust_scale = max(global_q75, eps)
    tail_guard = 1.0 / (
        1.0 + 0.25 * np.maximum(d / robust_scale - 1.0, 0.0) ** 1.35
    )

    score = (
        adaptive_decay ** 1.20
        * median_decay ** 0.45
        * rank_term ** 1.10
        * compatibility ** 1.25
        * global_term
        * density_balance
        * tail_guard
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score = score / maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
