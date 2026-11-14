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
    diagonal = np.eye(n, dtype=bool)

    finite_offdiag = np.isfinite(distances) & ~diagonal
    positive_offdiag = finite_offdiag & (distances > 0.0)

    if not np.any(finite_offdiag):
        return np.zeros((n, n), dtype=float)

    finite_values = distances[finite_offdiag]
    positive_values = distances[positive_offdiag]

    fallback = (
        float(np.percentile(positive_values, 90.0))
        if positive_values.size
        else float(np.max(finite_values))
    )
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.nan_to_num(
        distances,
        nan=fallback,
        posinf=fallback,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))

    partitioned = np.partition(d, min(k, n - 1), axis=1)
    neighbors = partitioned[:, :k]
    alternative = partitioned[:, min(k, n - 1)]

    local_scale = np.mean(neighbors, axis=1)
    local_scale = np.maximum(local_scale, eps)

    if n > 2:
        nearest = partitioned[:, 0]
        second_nearest = partitioned[:, 1]
    else:
        nearest = partitioned[:, 0]
        second_nearest = nearest

    local_gap = np.maximum(second_nearest - nearest, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = d / (pair_scale + eps)
    geometric_affinity = np.exp(
        -np.clip(relative_distance ** 0.88, 0.0, 50.0)
    )

    ratio_i = d / (local_scale[:, None] + eps)
    ratio_j = d / (local_scale[None, :] + eps)

    endpoint_affinity = np.exp(
        -np.clip(
            0.5 * (
                ratio_i ** 0.82
                + ratio_j ** 0.82
            ),
            0.0,
            50.0,
        )
    )

    worst_endpoint_affinity = np.exp(
        -np.clip(
            0.35 * np.minimum(ratio_i, ratio_j)
            + 0.65 * np.maximum(ratio_i, ratio_j),
            0.0,
            50.0,
        )
    )

    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    normalized_rank_i = ranks / max(float(n - 1), 1.0)
    normalized_rank_j = normalized_rank_i.T

    mutual_rank = np.exp(
        -1.35 * np.sqrt(
            np.clip(normalized_rank_i * normalized_rank_j, 0.0, None)
        )
    )

    candidate_width = float(max(k, 1))
    candidate_i = np.exp(
        -np.clip(ranks / candidate_width, 0.0, 50.0)
    )
    candidate_j = np.exp(
        -np.clip(ranks.T / candidate_width, 0.0, 50.0)
    )

    candidate_gate = 0.58 + 0.42 * np.sqrt(
        candidate_i * candidate_j
    )

    alternative_scale = np.sqrt(
        np.maximum(alternative[:, None], eps)
        * np.maximum(alternative[None, :], eps)
    )

    relative_margin = (
        alternative_scale - d
    ) / (
        alternative_scale + d + eps
    )
    relative_margin = np.clip(relative_margin, 0.0, 1.0)

    gap_scale_i = local_gap[:, None]
    gap_scale_j = local_gap[None, :]

    gap_margin_i = np.clip(
        (alternative[:, None] - d)
        / (alternative[:, None] + gap_scale_i + eps),
        0.0,
        1.0,
    )
    gap_margin_j = np.clip(
        (alternative[None, :] - d)
        / (alternative[None, :] + gap_scale_j + eps),
        0.0,
        1.0,
    )

    margin_support = np.sqrt(
        relative_margin
        * gap_margin_i
        * gap_margin_j
    )
    margin_bonus = 1.0 + 0.08 * margin_support

    scale_balance = np.sqrt(
        local_scale[:, None] / (local_scale[None, :] + eps)
    )
    density_balance = np.exp(
        -0.22 * np.abs(np.log(np.maximum(scale_balance, eps)))
    )

    score = (
        geometric_affinity
        * endpoint_affinity
        * worst_endpoint_affinity
        * mutual_rank
        * candidate_gate
        * margin_bonus
        * density_balance
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
