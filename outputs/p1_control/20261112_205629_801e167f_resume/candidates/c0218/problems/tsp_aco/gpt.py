import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric endpoint-aware edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite_input = np.isfinite(distances)
    positive_input = finite_input & (distances > 0.0)
    np.fill_diagonal(positive_input, False)

    if not np.any(positive_input):
        return np.zeros((n, n), dtype=float)

    positive_values = distances[positive_input]
    fallback = float(np.max(positive_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(finite_input, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, fallback)

    positive_values = d[d > 0.0]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else fallback
    )
    global_scale = max(global_scale, eps)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    k = max(1, min(n - 1, int(np.rint(np.sqrt(n))) + 1))

    ordered = np.sort(work, axis=1)
    neighbors = ordered[:, :k]

    weights = np.linspace(1.0, 0.6, k, dtype=float)
    weights /= np.sum(weights)

    local_scale = neighbors @ weights
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = d / np.maximum(pair_scale, eps)
    geometric_affinity = np.exp(
        -np.clip(relative_distance ** 0.9, 0.0, 50.0)
    )

    endpoint_ratio_i = d / local_scale[:, None]
    endpoint_ratio_j = d / local_scale[None, :]

    endpoint_i = np.exp(
        -np.clip(endpoint_ratio_i ** 0.85, 0.0, 50.0)
    )
    endpoint_j = np.exp(
        -np.clip(endpoint_ratio_j ** 0.85, 0.0, 50.0)
    )

    bottleneck_affinity = np.minimum(endpoint_i, endpoint_j)
    endpoint_compatibility = np.sqrt(endpoint_i * endpoint_j)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    normalized_i = ranks / max(float(n - 1), 1.0)
    normalized_j = normalized_i.T

    rank_affinity = np.exp(
        -1.35 * np.clip(
            0.5 * (normalized_i + normalized_j),
            0.0,
            1.0,
        )
    )

    candidate_width = max(float(k), 1.0)
    candidate_i = np.exp(
        -np.clip(ranks / candidate_width, 0.0, 50.0)
    )
    candidate_j = np.exp(
        -np.clip(ranks.T / candidate_width, 0.0, 50.0)
    )

    mutual_candidate_support = np.sqrt(candidate_i * candidate_j)
    soft_candidate_gate = (
        0.55 + 0.45 * mutual_candidate_support
    )

    nearest = ordered[:, 0]
    second_nearest = ordered[:, 1] if n > 2 else nearest
    alternative_scale = np.sqrt(
        np.maximum(second_nearest[:, None], eps)
        * np.maximum(second_nearest[None, :], eps)
    )

    margin = alternative_scale / (
        d + alternative_scale + eps
    )
    margin_bonus = 1.0 + 0.12 * np.clip(margin, 0.0, 1.0)

    density_balance = np.exp(
        -0.25
        * np.abs(
            np.log(
                np.maximum(
                    local_scale[:, None]
                    / np.maximum(local_scale[None, :], eps),
                    eps,
                )
            )
        )
    )

    global_guard = 0.88 + 0.12 * np.exp(
        -np.clip((d / global_scale) ** 0.9, 0.0, 50.0)
    )

    score = (
        geometric_affinity
        * bottleneck_affinity
        * endpoint_compatibility
        * rank_affinity
        * soft_candidate_gate
        * margin_bonus
        * density_balance
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
