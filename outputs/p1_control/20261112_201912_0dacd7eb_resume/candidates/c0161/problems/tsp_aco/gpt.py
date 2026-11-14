import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust endpoint-aware edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    eps = np.finfo(float).eps

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(d), d, fallback)
    np.fill_diagonal(d, np.inf)

    positive_values = finite_values[finite_values > 0.0]
    if positive_values.size:
        global_scale = float(np.percentile(positive_values, 60.0))
        global_q75 = float(np.percentile(positive_values, 75.0))
    else:
        global_scale = fallback
        global_q75 = fallback

    global_scale = max(global_scale, eps)
    global_q75 = max(global_q75, global_scale)

    k = max(2, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))
    ordered = np.sort(d, axis=1)
    neighbors = ordered[:, :k]

    weights = np.linspace(1.0, 0.5, k, dtype=float)
    weights /= np.sum(weights)

    local_scale = np.sum(neighbors * weights[None, :], axis=1)
    local_scale = np.maximum(local_scale, eps)

    nearest = ordered[:, 0]
    second_nearest = ordered[:, 1] if n > 2 else nearest
    local_gap = np.maximum(second_nearest - nearest, 0.0)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = d / pair_scale
    endpoint_ratio_i = d / local_scale[:, None]
    endpoint_ratio_j = d / local_scale[None, :]

    geometric_term = np.exp(
        -np.clip(relative_distance ** 0.9, 0.0, 50.0)
    )

    endpoint_penalty = (
        np.log1p(endpoint_ratio_i) ** 1.2
        + np.log1p(endpoint_ratio_j) ** 1.2
    )
    endpoint_compatibility = np.exp(
        -np.clip(0.75 * endpoint_penalty, 0.0, 50.0)
    )

    row_order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(row_order, dtype=float)
    ranks[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    normalized_rank_i = ranks / max(float(n - 1), 1.0)
    normalized_rank_j = normalized_rank_i.T

    rank_term = (
        1.0
        + 1.5 * normalized_rank_i
        + 1.5 * normalized_rank_j
    ) ** -0.9

    candidate_width = max(float(k), 1.0)
    candidate_i = np.exp(
        -np.clip(normalized_rank_i * (n - 1) / candidate_width, 0.0, 50.0)
    )
    candidate_j = np.exp(
        -np.clip(normalized_rank_j * (n - 1) / candidate_width, 0.0, 50.0)
    )

    mutual_support = np.sqrt(candidate_i * candidate_j)
    soft_mutual_gate = 0.50 + 0.50 * mutual_support

    local_density_i = local_scale / global_scale
    local_density_j = local_scale / global_scale
    density_balance = np.exp(
        -0.30
        * np.abs(
            np.log(
                np.maximum(
                    (1.0 + local_density_i[:, None])
                    / (1.0 + local_density_j[None, :]),
                    eps,
                )
            )
        )
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

    margin_strength = np.sqrt(margin_i * margin_j)
    margin_bonus = 1.0 + 0.08 * margin_strength

    tail_guard = 1.0 / (
        1.0
        + 0.25
        * np.maximum(d / global_q75 - 1.0, 0.0) ** 1.25
    )

    score = (
        geometric_term ** 1.15
        * endpoint_compatibility ** 1.20
        * rank_term
        * soft_mutual_gate
        * density_balance
        * margin_bonus
        * tail_guard
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
