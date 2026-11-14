import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed, row-normalized edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros_like(distances, dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    invalid_distance = np.inf
    work = np.where(valid, distances, invalid_distance)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(work, order, axis=1)

    counts = valid.sum(axis=1).astype(int)
    row_indices = np.arange(n)

    median_index = np.maximum((counts - 1) // 2, 0)
    lower_index = np.maximum((counts - 1) // 4, 0)
    upper_index = np.maximum(
        np.minimum((3 * counts - 1) // 4, n - 1),
        0,
    )

    row_median = sorted_distances[row_indices, median_index]
    row_lower = sorted_distances[row_indices, lower_index]
    row_upper = sorted_distances[row_indices, upper_index]

    finite_rows = counts > 0
    positive_values = distances[valid & (distances > 0.0)]

    if positive_values.size:
        fallback_scale = float(np.median(positive_values))
    else:
        fallback_scale = 1.0

    if not np.isfinite(fallback_scale) or fallback_scale <= 0.0:
        fallback_scale = 1.0

    row_median = np.where(
        np.isfinite(row_median) & (row_median >= 0.0),
        row_median,
        fallback_scale,
    )
    row_lower = np.where(
        np.isfinite(row_lower) & (row_lower >= 0.0),
        row_lower,
        row_median,
    )
    row_upper = np.where(
        np.isfinite(row_upper) & (row_upper >= 0.0),
        row_upper,
        row_median,
    )

    row_scale = np.maximum(row_median, eps)
    relative_spread = np.clip(
        (row_upper - row_lower) / row_scale,
        0.0,
        4.0,
    )

    # Convert each valid edge to an empirical row-wise rank.
    inverse_order = np.empty_like(order)
    inverse_order[row_indices[:, None], order] = np.arange(n, dtype=int)

    rank_fraction = np.divide(
        inverse_order.astype(float),
        np.maximum(counts[:, None] - 1, 1),
    )
    rank_fraction = np.clip(rank_fraction, 0.0, 1.0)

    # High-spread rows receive a flatter rank profile and explore more edges.
    temperature = 0.58 + 0.22 * np.minimum(relative_spread, 2.0)
    rank_weight = np.power(
        np.maximum(1.0 - rank_fraction, 0.0),
        1.0 / temperature[:, None],
    )

    # Use the row median as a scale reference without coupling opposite
    # directions or forcing symmetry.
    distance_ratio = np.divide(
        work,
        row_scale[:, None],
        out=np.full_like(work, np.inf, dtype=float),
        where=row_scale[:, None] > 0.0,
    )
    distance_weight = np.power(
        1.0 + np.maximum(distance_ratio, 0.0),
        -0.72,
    )

    # Slightly reward edges that are rank-distinct from the rest of their row.
    rank_margin = np.maximum(
        np.minimum(rank_fraction, 1.0 - rank_fraction),
        0.0,
    )
    distinctiveness = 1.0 + 0.08 * (
        1.0 - 2.0 * rank_margin
    )

    weights = (
        rank_weight
        * distance_weight
        * distinctiveness
    )
    weights[~valid] = 0.0

    # Iteratively reduce destination congestion. This creates a directed
    # competition signal instead of a mutual-nearest-neighbor bonus.
    for _ in range(4):
        row_totals = weights.sum(axis=1, keepdims=True)
        normalized = np.divide(
            weights,
            row_totals,
            out=np.zeros_like(weights),
            where=row_totals > 0.0,
        )

        destination_load = normalized.sum(axis=0)
        active_rows = float(np.count_nonzero(row_totals[:, 0] > 0.0))

        if active_rows > 0.0:
            mean_load = max(float(destination_load.sum()) / active_rows, eps)
        else:
            mean_load = 1.0

        congestion = 1.0 / np.sqrt(
            1.0 + destination_load / mean_load
        )
        congestion /= max(float(np.max(congestion)), eps)

        weights *= congestion[None, :]
        weights[~valid] = 0.0

    row_totals = weights.sum(axis=1, keepdims=True)
    normalized = np.divide(
        weights,
        row_totals,
        out=np.zeros_like(weights),
        where=row_totals > 0.0,
    )

    valid_counts = np.maximum(counts.astype(float), 1.0)
    uniform_valid = valid.astype(float) / valid_counts[:, None]

    # Preserve exploration on valid edges while retaining the learned ranking.
    result = 0.94 * normalized + 0.06 * uniform_valid
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    final_totals = result.sum(axis=1, keepdims=True)
    return np.divide(
        result,
        final_totals,
        out=np.zeros_like(result),
        where=final_totals > 0.0,
    )
