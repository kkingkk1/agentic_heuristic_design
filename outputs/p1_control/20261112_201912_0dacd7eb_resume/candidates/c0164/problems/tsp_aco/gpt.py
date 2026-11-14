import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust reciprocal multi-scale edge-priority indicators."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)

    diagonal = np.eye(n, dtype=bool)
    finite_offdiag = d[~diagonal & np.isfinite(d)]

    if finite_offdiag.size == 0:
        return np.zeros((n, n), dtype=float)

    positive = finite_offdiag[finite_offdiag > 0.0]
    fallback = float(np.max(positive)) if positive.size else 1.0
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    positive = d[np.isfinite(d) & (d > 0.0)]
    global_scale = (
        float(np.median(positive)) if positive.size else fallback
    )
    global_scale = max(global_scale, eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))
    ordered = np.sort(d, axis=1)
    nearest = ordered[:, :k]

    weights = np.linspace(1.0, 0.5, k, dtype=float)
    weights /= weights.sum()

    local_scale = np.maximum(
        np.sum(nearest * weights[None, :], axis=1),
        eps,
    )
    local_median = np.maximum(np.median(nearest, axis=1), eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    median_pair_scale = np.sqrt(
        local_median[:, None] * local_median[None, :]
    )

    pair_ratio = d / (pair_scale + eps)
    endpoint_ratio_i = d / (local_scale[:, None] + eps)
    endpoint_ratio_j = d / (local_scale[None, :] + eps)

    geometric_affinity = np.exp(
        -np.clip(pair_ratio ** 0.88, 0.0, 50.0)
    )
    median_affinity = np.exp(
        -np.clip((d / (median_pair_scale + eps)) ** 0.95, 0.0, 50.0)
    )

    endpoint_affinity_i = np.exp(
        -np.clip(endpoint_ratio_i ** 0.82, 0.0, 50.0)
    )
    endpoint_affinity_j = np.exp(
        -np.clip(endpoint_ratio_j ** 0.82, 0.0, 50.0)
    )
    bottleneck_affinity = np.minimum(
        endpoint_affinity_i,
        endpoint_affinity_j,
    )

    row_order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)

    rank_scale = max(float(n - 1), 1.0)
    normalized_i = ranks / rank_scale
    normalized_j = ranks.T / rank_scale

    reciprocal_rank = np.exp(
        -1.55 * 0.5 * (normalized_i + normalized_j)
    )

    candidate_width = max(float(k), 1.0)
    candidate_i = np.exp(
        -np.clip(ranks / candidate_width, 0.0, 50.0)
    )
    candidate_j = np.exp(
        -np.clip(ranks.T / candidate_width, 0.0, 50.0)
    )
    reciprocal_support = np.sqrt(candidate_i * candidate_j)

    mutual_local = (
        (ranks <= float(min(n - 1, 2 * k)))
        & (ranks.T <= float(min(n - 1, 2 * k)))
    )
    mutual_bonus = 1.0 + 0.14 * reciprocal_support * mutual_local

    second_nearest = (
        ordered[:, 1] if n > 2 else ordered[:, 0]
    )
    margin_i = np.clip(
        (second_nearest[:, None] - d)
        / (second_nearest[:, None] + local_scale[:, None] + eps),
        0.0,
        1.0,
    )
    margin_j = np.clip(
        (second_nearest[None, :] - d)
        / (second_nearest[None, :] + local_scale[None, :] + eps),
        0.0,
        1.0,
    )
    margin_bonus = 1.0 + 0.10 * np.sqrt(margin_i * margin_j)

    density = local_scale / global_scale
    density_balance = np.exp(
        -0.30
        * np.abs(
            np.log(
                (1.0 + density[:, None])
                / (1.0 + density[None, :])
            )
        )
    )

    global_guard = 0.35 + 0.65 * np.exp(
        -np.clip((d / global_scale) ** 0.9, 0.0, 50.0)
    )

    score = (
        geometric_affinity ** 1.15
        * median_affinity ** 0.35
        * (0.45 + 0.55 * bottleneck_affinity) ** 1.20
        * reciprocal_rank
        * (0.55 + 0.45 * reciprocal_support)
        * mutual_bonus
        * margin_bonus
        * density_balance
        * global_guard
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
