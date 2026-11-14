import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, scale-invariant edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return scores

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    valid_values = d[valid]
    positive_values = valid_values[valid_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.max(valid_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    ordered = np.argsort(d, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(d, ordered, axis=1)

    nearest = sorted_distances[:, 0]
    second_nearest = (
        sorted_distances[:, 1] if n > 2 else sorted_distances[:, 0]
    )

    nearest_finite = np.isfinite(nearest)
    second_finite = np.isfinite(second_nearest)

    nearest_safe = np.where(nearest_finite, nearest, global_scale)
    second_safe = np.where(second_finite, second_nearest, nearest_safe)

    local_scale = 0.5 * (nearest_safe + second_safe)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    rank = np.full((n, n), float(n), dtype=float)
    rows = np.repeat(np.arange(n), n)
    cols = ordered.reshape(-1)
    flat_distances = sorted_distances.reshape(-1)
    flat_rank = np.tile(np.arange(n, dtype=float), n)
    flat_valid = np.isfinite(flat_distances)
    rank[rows[flat_valid], cols[flat_valid]] = flat_rank[flat_valid]

    rank_temperature = max(1.0, 0.72 * np.sqrt(float(n)))
    rank_score = np.exp(-rank / rank_temperature)
    bidirectional_rank = np.sqrt(rank_score * rank_score.T)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        d,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (geometric_scale > 0.0),
    )

    distance_score = np.exp(
        -np.clip(1.12 * relative_distance, 0.0, 60.0)
    )

    endpoint_ratio = (
        np.divide(
            d,
            local_scale[:, None],
            out=np.full((n, n), np.inf, dtype=float),
            where=valid,
        )
        + np.divide(
            d,
            local_scale[None, :],
            out=np.full((n, n), np.inf, dtype=float),
            where=valid,
        )
    )

    endpoint_score = np.exp(
        -np.clip(0.34 * endpoint_ratio, 0.0, 60.0)
    )

    endpoint_saving = (
        np.maximum(second_safe[:, None] - d, 0.0)
        / np.maximum(second_safe[:, None], global_scale)
        + np.maximum(second_safe[None, :] - d, 0.0)
        / np.maximum(second_safe[None, :], global_scale)
    )

    saving_bonus = 1.0 + 0.38 * np.clip(endpoint_saving, 0.0, 2.0)

    nearest_index = np.full(n, -1, dtype=int)
    has_nearest = np.isfinite(sorted_distances[:, 0])
    nearest_index[has_nearest] = ordered[has_nearest, 0]

    indices = np.arange(n)
    mutual_nearest = (
        (nearest_index[:, None] == indices[None, :])
        & (nearest_index[None, :] == indices[:, None])
        & (nearest_index[:, None] >= 0)
        & (nearest_index[None, :] >= 0)
    )
    mutual_bonus = np.where(mutual_nearest, 1.42, 1.0)

    long_edge_penalty = np.exp(
        -0.58 * np.maximum(relative_distance - 1.0, 0.0)
    )

    scores = (
        distance_score
        * (
            0.42
            + 0.31 * bidirectional_rank
            + 0.27 * endpoint_score
        )
        * saving_bonus
        * mutual_bonus
        * long_edge_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
