import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank- and neighborhood-based edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)

    valid = (~diagonal) & (finite | finite.T)
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.zeros((n, n), dtype=float)

    both = finite & finite.T
    only_left = finite & ~finite.T
    only_right = ~finite & finite.T

    cleaned[both] = 0.5 * (raw[both] + raw.T[both])
    cleaned[only_left] = raw[only_left]
    cleaned[only_right] = raw.T[only_right]

    cleaned = np.maximum(cleaned, 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)

    observed = cleaned[valid]
    positive = observed[observed > 0.0]

    if positive.size:
        median = float(np.median(positive))
        mad = float(np.median(np.abs(positive - median)))
        q25, q75 = np.quantile(positive, [0.25, 0.75])
        spread = max(float(q75 - q25), mad)
    else:
        median = 1.0
        spread = 1.0

    if not np.isfinite(median) or median <= 0.0:
        median = 1.0
    if not np.isfinite(spread) or spread <= 0.0:
        spread = 0.35 * median

    eps = np.finfo(float).eps
    scale = max(spread, 0.08 * median, eps)

    largest = float(np.max(observed)) if observed.size else median
    if not np.isfinite(largest):
        largest = median

    replacement = largest + 8.0 * max(scale, median, eps)

    distances = np.full((n, n), replacement, dtype=float)
    distances[valid] = cleaned[valid]
    np.fill_diagonal(distances, replacement)

    # Robust global distance preference based on a logistic quantile contrast.
    sorted_edges = np.sort(observed)
    q20, q50, q80 = np.quantile(sorted_edges, [0.20, 0.50, 0.80])
    global_width = max(float(q80 - q20), 0.25 * scale, eps)

    centered = (q50 - distances) / global_width
    global_preference = np.zeros((n, n), dtype=float)
    global_preference[valid] = 1.0 / (
        1.0 + np.exp(-np.clip(centered[valid], -30.0, 30.0))
    )

    # Stable node-wise ranks. Lower distance receives higher rank score.
    order = np.argsort(distances, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)

    valid_count = np.maximum(valid.sum(axis=1).astype(float), 1.0)
    rank_width = np.maximum(
        0.55 * np.sqrt(valid_count),
        1.0,
    )

    row_rank_score = np.exp(
        -ranks / rank_width[:, None]
    )
    row_rank_score[~valid] = 0.0

    # Reciprocity rewards edges that are preferred by both endpoints.
    reciprocal_rank = np.sqrt(
        np.maximum(row_rank_score * row_rank_score.T, 0.0)
    )
    reciprocal_rank[~valid] = 0.0

    # Local scarcity: an endpoint with few competitive alternatives gives
    # extra weight to its strongest edges.
    row_best = np.min(distances, axis=1)
    row_second = np.partition(distances, 1, axis=1)[:, 1]

    local_gap = np.maximum(
        row_second - row_best,
        0.18 * scale,
    )
    local_gap = np.maximum(local_gap, eps)

    local_contrast = np.exp(
        -np.maximum(distances - row_best[:, None], 0.0)
        / local_gap[:, None]
    )
    local_contrast[~valid] = 0.0

    row_mass = np.maximum(
        local_contrast.sum(axis=1, keepdims=True),
        eps,
    )
    scarcity = 1.0 / row_mass
    scarcity = scarcity / max(float(np.max(scarcity)), eps)

    endpoint_scarcity = np.sqrt(
        np.maximum(
            local_contrast * scarcity,
            0.0,
        )
        * np.maximum(
            local_contrast.T * scarcity.T,
            0.0,
        )
    )
    endpoint_scarcity[~valid] = 0.0

    # Triangle closure: reward edges whose endpoints share nearby neighbors.
    # The calculation is bounded by a small adaptive neighborhood.
    neighborhood_size = int(
        np.clip(
            4.0 + 1.5 * np.sqrt(float(n)),
            4,
            24,
        )
    )
    neighborhood_size = min(neighborhood_size, max(1, n - 1))

    neighbor_ids = np.argpartition(
        distances,
        kth=neighborhood_size - 1,
        axis=1,
    )[:, :neighborhood_size]

    neighborhood_mask = np.zeros((n, n), dtype=float)
    neighborhood_mask[
        np.arange(n)[:, None],
        neighbor_ids,
    ] = 1.0
    neighborhood_mask[~valid] = 0.0

    shared_neighbors = (
        neighborhood_mask @ neighborhood_mask.T
    )

    denom = np.sqrt(
        np.maximum(
            neighborhood_mask.sum(axis=1)[:, None]
            * neighborhood_mask.sum(axis=1)[None, :],
            1.0,
        )
    )
    closure = shared_neighbors / denom
    closure[~valid] = 0.0

    # Edges that are simultaneously short and bridge locally coherent
    # neighborhoods receive a mild multiplicative bonus.
    edge_scale = max(0.45 * scale, eps)
    local_shortness = np.exp(
        -np.minimum(distances, replacement) / edge_scale
    )
    local_shortness[~valid] = 0.0

    score = (
        np.maximum(global_preference, eps) ** 0.30
        * np.maximum(reciprocal_rank, eps) ** 0.28
        * np.maximum(endpoint_scarcity, eps) ** 0.22
        * np.maximum(0.35 + 0.65 * closure, eps) ** 0.12
        * np.maximum(0.25 + 0.75 * local_shortness, eps) ** 0.08
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    # Contrast compression prevents one unusually short edge from
    # eliminating all other feasible alternatives.
    positive_score = score[valid]
    if positive_score.size:
        reference = float(np.quantile(positive_score, 0.90))
    else:
        reference = 1.0

    reference = max(reference, eps)
    result = score / (score + reference)
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
