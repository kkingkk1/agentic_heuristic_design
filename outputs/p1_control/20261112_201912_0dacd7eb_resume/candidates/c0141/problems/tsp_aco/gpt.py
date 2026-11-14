import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return endpoint-aware edge-priority indicators for stochastic TSP sampling."""
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
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_values))
    d = np.where(np.isfinite(d), d, fallback)

    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else max(fallback, 1.0)
    )
    global_scale = max(global_scale, eps)

    k = max(1, min(n - 1, int(np.sqrt(n)) + 1))
    ordered = np.sort(d, axis=1)

    local_scale = np.mean(ordered[:, :k], axis=1)
    local_scale = np.maximum(local_scale, eps)

    nearest = ordered[:, 0]
    second_nearest = (
        ordered[:, 1] if n > 2 else ordered[:, 0]
    )
    local_gap = np.maximum(second_nearest - nearest, eps)

    geometric_distance = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    endpoint_distance = np.minimum(
        d / local_scale[:, None],
        d / local_scale[None, :],
    )

    geometric_affinity = np.exp(
        -np.clip(geometric_distance, 0.0, 50.0)
    )
    bottleneck_affinity = np.exp(
        -np.clip(endpoint_distance, 0.0, 50.0)
    )

    row_order = np.argsort(d, axis=1)
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    rank_i = row_rank / max(float(n - 1), 1.0)
    rank_j = row_rank.T / max(float(n - 1), 1.0)
    mean_rank = 0.5 * (rank_i + rank_j)

    mutual_rank_affinity = np.exp(
        -1.6 * np.clip(mean_rank, 0.0, 1.0)
    )

    candidate_width = max(float(k), 1.0)
    candidate_i = np.exp(
        -np.clip(row_rank / candidate_width, 0.0, 50.0)
    )
    candidate_j = np.exp(
        -np.clip(row_rank.T / candidate_width, 0.0, 50.0)
    )
    mutual_candidate_support = np.sqrt(candidate_i * candidate_j)

    endpoint_margin_i = np.clip(
        (second_nearest[:, None] - d)
        / (second_nearest[:, None] + local_scale[:, None] + eps),
        0.0,
        1.0,
    )
    endpoint_margin_j = np.clip(
        (second_nearest[None, :] - d)
        / (second_nearest[None, :] + local_scale[None, :] + eps),
        0.0,
        1.0,
    )
    margin_bonus = 1.0 + 0.12 * np.sqrt(
        endpoint_margin_i * endpoint_margin_j
    )

    reciprocal_reinforcement = 0.85 + 0.15 * np.sqrt(
        geometric_affinity * bottleneck_affinity
    )

    soft_candidate_gate = 0.55 + 0.45 * mutual_candidate_support

    score = (
        geometric_affinity
        * bottleneck_affinity
        * mutual_rank_affinity
        * soft_candidate_gate
        * margin_bonus
        * reciprocal_reinforcement
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
