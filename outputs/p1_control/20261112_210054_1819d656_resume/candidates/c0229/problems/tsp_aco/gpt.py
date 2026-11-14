import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return endpoint-aware edge-priority indicators for TSP sampling."""

    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    valid_values = distances[valid]
    fallback = float(np.max(valid_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    values = np.where(valid, distances, 0.0)
    support = valid.astype(float) + valid.T.astype(float)

    symmetric = np.divide(
        values + values.T,
        support,
        out=np.full((n, n), fallback, dtype=float),
        where=support > 0.0,
    )

    symmetric = np.maximum(symmetric, 0.0)
    np.fill_diagonal(symmetric, np.inf)

    finite_values = symmetric[np.isfinite(symmetric)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(symmetric), symmetric, fallback)
    np.fill_diagonal(d, fallback)

    eps = np.finfo(float).eps

    # Use a small adaptive neighborhood to estimate endpoint-specific scale.
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n))) + 1))
    order = np.argsort(d, axis=1, kind="stable")
    ordered = np.take_along_axis(d, order, axis=1)

    local_neighbors = ordered[:, :k]
    weights = np.linspace(1.0, 0.55, k, dtype=float)
    weights /= np.sum(weights)

    local_scale = local_neighbors @ weights
    local_scale = np.maximum(local_scale, eps)

    nearest = ordered[:, 0]
    if n > 2:
        second = ordered[:, 1]
    else:
        second = nearest

    # Stable ranks, with rank zero denoting the closest available neighbor.
    rank = np.empty((n, n), dtype=float)
    rank[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)

    normalized_rank = rank / max(float(n - 1), 1.0)

    # Endpoint-relative distance. The maximum endpoint penalty prevents
    # an edge from being favored when it is poor for either endpoint.
    relative_i = d / np.maximum(local_scale[:, None], eps)
    relative_j = d / np.maximum(local_scale[None, :], eps)

    worst_relative = np.maximum(relative_i, relative_j)
    mean_relative = 0.5 * (relative_i + relative_j)

    endpoint_affinity = np.exp(
        -0.62 * np.clip(worst_relative ** 0.78, 0.0, 50.0)
        -0.18 * np.clip(mean_relative ** 0.88, 0.0, 50.0)
    )

    # Geometric pair scale preserves useful edges between endpoints
    # with different local densities.
    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )
    pair_relative = d / pair_scale

    pair_affinity = np.exp(
        -0.42 * np.clip(pair_relative ** 0.82, 0.0, 50.0)
    )

    # Bidirectional rank and mutual-candidate support.
    rank_i = np.exp(
        -0.72 * np.clip(rank / max(float(k), 1.0), 0.0, 50.0)
    )
    rank_j = rank_i.T
    mutual_rank = np.sqrt(rank_i * rank_j)

    candidate_i = np.exp(
        -np.clip(rank / max(float(k), 1.0), 0.0, 50.0)
    )
    candidate_j = candidate_i.T
    mutual_candidate = np.sqrt(candidate_i * candidate_j)

    candidate_support = 0.72 + 0.28 * mutual_candidate

    # Reward an edge that is substantially better than the endpoint's
    # second-best local alternative, but keep the reward deliberately weak.
    gap_i = np.maximum(second - nearest, eps)
    gap_j = np.maximum(second - nearest, eps)

    advantage_i = np.clip(
        (second[:, None] - d) / np.maximum(second[:, None] + gap_i[:, None], eps),
        0.0,
        1.0,
    )
    advantage_j = np.clip(
        (second[None, :] - d) / np.maximum(second[None, :] + gap_j[None, :], eps),
        0.0,
        1.0,
    )

    selective_bonus = 1.0 + 0.12 * np.sqrt(advantage_i * advantage_j)

    # Mild density-balance correction avoids systematically preferring
    # edges solely because one endpoint lies in a dense region.
    density_ratio = np.maximum(
        local_scale[:, None] / np.maximum(local_scale[None, :], eps),
        eps,
    )
    density_balance = np.exp(
        -0.12 * np.abs(np.log(density_ratio))
    )

    score = (
        endpoint_affinity
        * pair_affinity
        * mutual_rank
        * candidate_support
        * selective_bonus
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
