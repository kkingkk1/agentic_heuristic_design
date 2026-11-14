import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust endpoint-aware edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite_values = d[off_diagonal & np.isfinite(d)]

    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_values = finite_values[finite_values > 0.0]
    if positive_values.size:
        fallback = float(np.percentile(positive_values, 90.0))
        global_scale = float(np.median(positive_values))
    else:
        fallback = float(np.max(finite_values))
        global_scale = fallback

    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = fallback

    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, np.inf)

    k = max(1, min(n - 1, int(np.ceil(1.25 * np.sqrt(n)))))

    partitioned = np.partition(d, k - 1, axis=1)
    nearest = partitioned[:, :k]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = d / (pair_scale + eps)
    local_affinity = np.exp(
        -np.clip(relative_length ** 0.88, 0.0, 50.0)
    )

    ratio_i = d / (local_scale[:, None] + eps)
    ratio_j = d / (local_scale[None, :] + eps)

    endpoint_compatibility = np.exp(
        -np.clip(
            0.55 * (
                np.log1p(np.maximum(ratio_i, 0.0))
                + np.log1p(np.maximum(ratio_j, 0.0))
            )
            + 0.20 * np.abs(
                np.log1p(np.maximum(ratio_i, 0.0))
                - np.log1p(np.maximum(ratio_j, 0.0))
            ),
            0.0,
            50.0,
        )
    )

    order = np.argsort(d, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_denominator = max(float(n - 1), 1.0)
    normalized_rank = ranks / rank_denominator

    mutual_rank = np.exp(
        -1.25 * np.sqrt(
            np.clip(normalized_rank * normalized_rank.T, 0.0, None)
        )
    )

    candidate_width = float(max(k, 1))
    candidate_support = np.sqrt(
        np.exp(
            -np.clip(ranks / candidate_width, 0.0, 50.0)
        )
        * np.exp(
            -np.clip(ranks.T / candidate_width, 0.0, 50.0)
        )
    )
    soft_candidate_gate = 0.55 + 0.45 * candidate_support

    alternative = partitioned[:, k - 1]
    alternative = np.maximum(alternative, eps)
    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )

    margin = alternative_scale / (
        d + alternative_scale + eps
    )
    margin_bonus = 1.0 + 0.08 * np.clip(margin, 0.0, 1.0)

    global_affinity = np.exp(
        -np.clip((d / max(global_scale, eps)) ** 0.80, 0.0, 50.0)
    )
    global_guard = 0.35 + 0.65 * global_affinity

    score = (
        local_affinity ** 1.15
        * endpoint_compatibility
        * mutual_rank ** 1.05
        * soft_candidate_gate
        * margin_bonus
        * global_guard
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
