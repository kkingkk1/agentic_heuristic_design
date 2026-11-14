import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    n = distance_matrix.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n == 0:
        return result

    eps = np.finfo(float).eps

    valid = np.isfinite(distance_matrix) & (distance_matrix >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    positive_mask = valid & (distance_matrix > 0.0)
    positive = distance_matrix[positive_mask]

    if positive.size:
        q05, q50, q95 = np.quantile(positive, [0.05, 0.50, 0.95])
        base_scale = max(float(q50), eps)
        lower = max(float(q05) * 0.25, base_scale * 1e-3, eps)
        upper = max(float(q95), lower)
        spread = max(float(q95 - q05), base_scale * 0.1, eps)
    else:
        base_scale = 1.0
        lower = eps
        upper = 1.0
        spread = 1.0

    zero_edge = valid & (distance_matrix <= eps)

    safe_distance = np.where(
        valid,
        np.maximum(distance_matrix, lower),
        base_scale,
    )
    clipped_distance = np.clip(safe_distance, lower, upper)

    def normalize_valid(values: np.ndarray) -> np.ndarray:
        masked = values[valid]
        output = np.zeros_like(values, dtype=float)

        if masked.size == 0:
            return output

        low, high = np.quantile(masked, [0.05, 0.95])
        if high - low <= eps:
            output[valid] = 1.0
        else:
            output[valid] = np.clip(
                (masked - low) / (high - low + eps),
                0.0,
                1.0,
            )

        return output

    def row_normalize(values: np.ndarray) -> np.ndarray:
        output = np.zeros_like(values, dtype=float)
        masked = np.where(valid, np.maximum(values, 0.0), 0.0)
        scale = np.max(masked, axis=1, keepdims=True)

        np.divide(
            masked,
            scale,
            out=output,
            where=scale > eps,
        )
        output[valid & (scale <= eps)] = 1.0
        return output

    row_scale = np.full(n, base_scale, dtype=float)
    column_scale = np.full(n, base_scale, dtype=float)
    second_nearest = np.full(n, base_scale, dtype=float)

    for i in range(n):
        row_values = distance_matrix[i, valid[i]]
        row_positive = row_values[row_values > 0.0]

        if row_positive.size:
            clipped_row = np.clip(row_positive, lower, upper)
            row_scale[i] = max(float(np.median(clipped_row)), lower)

            sorted_row = np.sort(row_positive)
            second_nearest[i] = float(
                sorted_row[1] if sorted_row.size >= 2 else sorted_row[0]
            )

        column_values = distance_matrix[valid[:, i], i]
        column_positive = column_values[column_values > 0.0]

        if column_positive.size:
            clipped_column = np.clip(column_positive, lower, upper)
            column_scale[i] = max(float(np.median(clipped_column)), lower)

    local_scale = np.sqrt(
        np.maximum(row_scale[:, None], lower)
        * np.maximum(column_scale[None, :], lower)
    )

    inverse = 1.0 / clipped_distance
    inverse = np.where(valid, inverse, 0.0)
    inverse = normalize_valid(inverse)
    inverse[zero_edge] = 1.0

    local_affinity = np.exp(
        -clipped_distance / (local_scale + eps)
    )
    local_affinity = np.where(valid, local_affinity, 0.0)
    local_affinity[zero_edge] = 1.0

    row_preference = np.exp(
        -clipped_distance / (row_scale[:, None] + eps)
    )
    column_preference = np.exp(
        -clipped_distance / (column_scale[None, :] + eps)
    )

    mutual_preference = np.sqrt(
        np.maximum(row_preference, 0.0)
        * np.maximum(column_preference, 0.0)
    )
    mutual_preference = np.where(valid, mutual_preference, 0.0)
    mutual_preference[zero_edge] = 1.0

    row_order = np.argsort(
        np.where(valid, clipped_distance, np.inf),
        axis=1,
        kind="mergesort",
    )
    row_rank = np.full((n, n), n, dtype=float)

    for i in range(n):
        indices = row_order[i][valid[i, row_order[i]]]
        row_rank[i, indices] = np.arange(1, indices.size + 1)

    column_valid = valid.T
    column_order = np.argsort(
        np.where(column_valid, clipped_distance.T, np.inf),
        axis=1,
        kind="mergesort",
    )
    column_rank = np.full((n, n), n, dtype=float)

    for j in range(n):
        indices = column_order[j][column_valid[j, column_order[j]]]
        column_rank[j, indices] = np.arange(1, indices.size + 1)

    row_counts = np.sum(valid, axis=1, keepdims=True)
    column_counts = np.sum(column_valid, axis=1, keepdims=True)

    outgoing_rank = np.where(
        valid,
        1.0 - (row_rank - 1.0) / np.maximum(row_counts, 1.0),
        0.0,
    )
    incoming_rank = np.where(
        valid,
        1.0 - (column_rank.T - 1.0)
        / np.maximum(column_counts.T, 1.0),
        0.0,
    )

    endpoint_agreement = np.sqrt(
        np.maximum(outgoing_rank, 0.0)
        * np.maximum(incoming_rank, 0.0)
    )
    endpoint_agreement[zero_edge] = 1.0

    row_opportunity = np.divide(
        np.maximum(
            second_nearest[:, None] - clipped_distance,
            0.0,
        ),
        second_nearest[:, None] + spread + eps,
    )
    row_opportunity = np.clip(row_opportunity, 0.0, 1.0)
    row_opportunity = np.where(valid, row_opportunity, 0.0)

    opportunity = normalize_valid(row_opportunity)
    opportunity[zero_edge] = 1.0

    rank_rarity = np.divide(
        1.0,
        1.0 + row_rank + column_rank.T,
        out=np.zeros((n, n), dtype=float),
        where=valid,
    )
    rank_rarity = normalize_valid(rank_rarity)
    rank_rarity[zero_edge] = 1.0

    score = (
        0.36 * inverse
        + 0.24 * local_affinity
        + 0.18 * mutual_preference
        + 0.10 * endpoint_agreement
        + 0.07 * opportunity
        + 0.05 * rank_rarity
    )

    score = np.where(valid, np.maximum(score, 0.0), 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > eps:
        score /= maximum

    return score.astype(float, copy=False)
