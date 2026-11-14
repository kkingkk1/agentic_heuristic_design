import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    raw_valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    safe = np.where(raw_valid, distances, np.inf)

    # Use the best mutually valid direction while enforcing symmetric scores.
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.full((n, n), float(n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)[None, :]

    # Because safe is symmetric, row and column ranks are equivalent.
    col_rank = row_rank.T

    valid_counts = np.sum(valid, axis=1)
    local_k = min(4, n - 1)

    nearest_values = np.take_along_axis(
        safe,
        order[:, :local_k],
        axis=1,
    )
    nearest_valid = np.isfinite(nearest_values)

    local_scale = np.sum(
        np.where(nearest_valid, nearest_values, 0.0),
        axis=1,
    ) / np.maximum(np.sum(nearest_valid, axis=1), 1)

    local_scale = np.where(
        valid_counts > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    normalized_distance = safe / np.maximum(pair_scale, eps)
    distance_score = np.exp(-np.clip(normalized_distance, 0.0, 50.0))

    rank_score = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )

    nearest = order[:, 0]
    mutual_nearest = (
        valid
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    top_k = min(2, n - 1)
    top_neighbors = valid & (row_rank < float(top_k))
    mutual_top_neighbors = (
        top_neighbors
        & top_neighbors.T
    ).astype(float)

    mutual_bonus = (
        1.0
        + 0.22 * mutual_nearest
        + 0.08 * mutual_top_neighbors
    )

    # Penalize edges that are poor for at least one endpoint, but keep the
    # correction bounded so distance remains the dominant signal.
    endpoint_scale = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )
    opportunity_ratio = safe / np.maximum(endpoint_scale, eps)
    opportunity_penalty = 1.0 / (
        1.0
        + 0.08 * np.clip(opportunity_ratio - 1.0, 0.0, 5.0)
    )

    # Estimate whether each endpoint has a useful alternative to its
    # strongest edge. This adds only a weak reward to non-dominant
    # competitive edges, reducing premature local lock-in.
    local_strength = np.where(
        valid,
        np.exp(-np.clip(safe / local_scale[:, None], 0.0, 50.0))
        / np.sqrt(1.0 + row_rank),
        0.0,
    )

    sorted_strength = np.sort(local_strength, axis=1)
    strongest = sorted_strength[:, -1]
    second_strongest = sorted_strength[:, -2]

    alternative_ratio = second_strongest / np.maximum(strongest, eps)
    alternative_ratio = np.clip(alternative_ratio, 0.0, 1.0)

    competitive = (
        valid
        & (row_rank < float(min(4, n - 1)))
        & (col_rank < float(min(4, n - 1)))
    )

    edge_topness = np.sqrt(
        1.0 / (1.0 + row_rank)
        * 1.0 / (1.0 + col_rank)
    )

    complementary_bonus = (
        1.0
        + 0.05
        * np.sqrt(
            alternative_ratio[:, None]
            * alternative_ratio[None, :]
        )
        * competitive
        * (1.0 - edge_topness)
    )

    scores = (
        distance_score
        * (0.55 + 0.45 * rank_score)
        * mutual_bonus
        * opportunity_penalty
        * complementary_bonus
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
