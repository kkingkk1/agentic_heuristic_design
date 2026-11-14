import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    diagonal = np.eye(n, dtype=bool)

    finite = np.isfinite(distances)
    raw = np.where(
        finite,
        np.maximum(distances, 0.0),
        np.inf,
    )
    np.fill_diagonal(raw, np.inf)

    transpose = raw.T
    finite_forward = np.isfinite(raw)
    finite_backward = np.isfinite(transpose)

    both = finite_forward & finite_backward
    valid = finite_forward | finite_backward

    symmetric_distance = np.full((n, n), np.inf, dtype=float)
    symmetric_distance[both] = (
        raw[both] + transpose[both]
    ) * 0.5
    symmetric_distance[finite_forward & ~finite_backward] = raw[
        finite_forward & ~finite_backward
    ]
    symmetric_distance[~finite_forward & finite_backward] = transpose[
        ~finite_forward & finite_backward
    ]
    np.fill_diagonal(symmetric_distance, np.inf)

    valid = valid & ~diagonal
    if not np.any(valid):
        return scores

    finite_values = symmetric_distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    global_scale = max(global_scale, np.finfo(float).eps)

    order = np.argsort(symmetric_distance, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(
        symmetric_distance,
        order,
        axis=1,
    )

    neighbor_count = min(
        n - 1,
        max(3, int(np.ceil(np.sqrt(n)))),
    )

    local_values = sorted_distances[:, :neighbor_count]
    finite_local = np.isfinite(local_values)

    local_sum = np.where(
        finite_local,
        local_values,
        0.0,
    ).sum(axis=1)
    local_count = finite_local.sum(axis=1)

    local_scale = np.divide(
        local_sum,
        np.maximum(local_count, 1),
        out=np.full(n, global_scale, dtype=float),
        where=local_count > 0,
    )
    local_scale = np.maximum(
        local_scale,
        global_scale * 0.05,
    )

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    pair_scale = np.maximum(
        pair_scale,
        np.finfo(float).eps,
    )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        normalized = symmetric_distance / pair_scale
        distance_preference = np.exp(
            -np.clip(normalized, 0.0, 60.0)
        )

        endpoint_preference = np.sqrt(
            np.exp(
                -np.clip(
                    symmetric_distance / local_scale[:, None],
                    0.0,
                    60.0,
                )
            )
            * np.exp(
                -np.clip(
                    symmetric_distance / local_scale[None, :],
                    0.0,
                    60.0,
                )
            )
        )

    rank = np.empty((n, n), dtype=float)
    rank_values = np.arange(n, dtype=float)
    rank[np.arange(n)[:, None], order] = rank_values[None, :]

    rank_continuity = 1.0 / np.sqrt(
        (1.0 + rank) * (1.0 + rank.T)
    )

    nearest = order[:, 0]
    mutual_nearest = (
        np.arange(n)[:, None] == nearest[None, :]
    ) & (
        np.arange(n)[None, :] == nearest[:, None]
    )

    top_mask = np.zeros((n, n), dtype=bool)
    top_order = order[:, :neighbor_count]
    top_mask[
        np.arange(n)[:, None],
        top_order,
    ] = True
    mutual_top = top_mask & top_mask.T

    second_distance = sorted_distances[:, 1]
    second_distance = np.where(
        np.isfinite(second_distance),
        second_distance,
        global_scale,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        scarcity = second_distance[:, None] / np.maximum(
            symmetric_distance,
            np.finfo(float).eps,
        )

    scarcity = 1.0 + 0.25 * np.clip(
        scarcity - 1.0,
        0.0,
        4.0,
    )
    scarcity = np.sqrt(scarcity * scarcity.T)

    logits = -np.clip(
        symmetric_distance / pair_scale,
        0.0,
        60.0,
    )
    logits[~valid] = -np.inf

    row_max = np.max(logits, axis=1, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore"):
        weights = np.exp(logits - row_max)

    weights[~valid] = 0.0
    row_total = weights.sum(axis=1, keepdims=True)

    preferences = np.divide(
        weights,
        row_total,
        out=np.zeros_like(weights),
        where=row_total > 0.0,
    )

    endpoint_importance = np.maximum(
        preferences,
        preferences.T,
    )

    scores = (
        distance_preference
        * endpoint_preference
        * (0.55 + 0.45 * rank_continuity)
        * (0.75 + 0.25 * endpoint_importance)
        * scarcity
        * (
            1.0
            + 0.35 * mutual_nearest.astype(float)
            + 0.20 * mutual_top.astype(float)
        )
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    scores = np.maximum(scores, 0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores, initial=0.0))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
