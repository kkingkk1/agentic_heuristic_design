import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    distances = distance_matrix.copy()
    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    result = np.zeros((n, n), dtype=float)
    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    masked = np.where(valid, distances, np.nan)

    def masked_row_min(values: np.ndarray) -> np.ndarray:
        out = np.full((values.shape[0], 1), np.nan, dtype=float)
        has_value = np.any(np.isfinite(values), axis=1)
        if np.any(has_value):
            out[has_value, 0] = np.nanmin(values[has_value], axis=1)
        return out

    def masked_row_max(values: np.ndarray) -> np.ndarray:
        out = np.full((values.shape[0], 1), np.nan, dtype=float)
        has_value = np.any(np.isfinite(values), axis=1)
        if np.any(has_value):
            out[has_value, 0] = np.nanmax(values[has_value], axis=1)
        return out

    def normalize_rows(values: np.ndarray) -> np.ndarray:
        values = np.where(valid, np.maximum(values, 0.0), 0.0)
        scale = np.max(values, axis=1, keepdims=True)
        return np.divide(
            values,
            scale,
            out=np.zeros_like(values, dtype=float),
            where=scale > eps,
        )

    def normalize_global(values: np.ndarray) -> np.ndarray:
        values = np.where(valid, np.maximum(values, 0.0), 0.0)
        positive = values[valid]
        if positive.size == 0:
            return np.zeros_like(values, dtype=float)

        low = float(np.quantile(positive, 0.05))
        high = float(np.quantile(positive, 0.95))
        if high - low <= eps:
            return np.where(valid, 1.0, 0.0)

        clipped = np.clip(values, low, high)
        return np.where(
            valid,
            (clipped - low) / (high - low + eps),
            0.0,
        )

    positive = distances[valid & (distances > 0.0)]
    if positive.size:
        positive_floor = float(np.quantile(positive, 0.05))
        positive_floor = max(positive_floor, eps)
    else:
        positive_floor = eps

    safe_distance = np.where(
        valid,
        np.maximum(distances, positive_floor),
        np.nan,
    )

    zero_bonus = np.where(valid & (distances <= eps), 1.0, 0.0)

    log_distance = np.where(
        valid,
        np.log1p(safe_distance / positive_floor),
        np.nan,
    )

    row_min = masked_row_min(log_distance)
    row_max = masked_row_max(log_distance)
    local_closeness = np.divide(
        row_max - np.nan_to_num(log_distance, nan=0.0),
        row_max - row_min + eps,
        out=np.zeros_like(distances, dtype=float),
        where=np.isfinite(row_min),
    )
    local_closeness = np.where(valid, local_closeness, 0.0)
    local_closeness = np.maximum(local_closeness, zero_bonus)

    global_closeness = normalize_global(
        -np.nan_to_num(log_distance, nan=0.0)
    )
    global_closeness = np.maximum(global_closeness, zero_bonus)

    row_order = np.argsort(
        np.where(valid, safe_distance, np.inf),
        axis=1,
        kind="mergesort",
    )
    row_rank = np.full((n, n), n, dtype=float)
    for i in range(n):
        valid_indices = row_order[i][valid[i, row_order[i]]]
        row_rank[i, valid_indices] = np.arange(1, valid_indices.size + 1)

    column_valid = valid.T
    column_distances = safe_distance.T
    column_order = np.argsort(
        np.where(column_valid, column_distances, np.inf),
        axis=1,
        kind="mergesort",
    )
    column_rank = np.full((n, n), n, dtype=float)
    for j in range(n):
        valid_indices = column_order[j][column_valid[j, column_order[j]]]
        column_rank[j, valid_indices] = np.arange(1, valid_indices.size + 1)

    row_counts = np.sum(valid, axis=1, keepdims=True)
    column_counts = np.sum(column_valid, axis=1, keepdims=True)

    outgoing_rank = np.where(
        valid,
        1.0 - (row_rank - 1.0) / np.maximum(row_counts, 1.0),
        0.0,
    )
    incoming_rank = np.where(
        valid,
        1.0 - (column_rank.T - 1.0) / np.maximum(column_counts.T, 1.0),
        0.0,
    )

    endpoint_agreement = np.sqrt(
        np.maximum(outgoing_rank, 0.0)
        * np.maximum(incoming_rank, 0.0)
    )

    row_sorted = np.sort(
        np.where(valid, safe_distance, np.inf),
        axis=1,
    )
    row_best = row_sorted[:, 0:1]
    row_second = row_sorted[:, 1:2] if n > 1 else row_best

    column_sorted = np.sort(
        np.where(column_valid, column_distances, np.inf),
        axis=1,
    )
    column_best = column_sorted[:, 0:1]
    column_second = column_sorted[:, 1:2] if n > 1 else column_best

    row_regret = np.divide(
        np.maximum(row_second - row_best, 0.0),
        row_second + eps,
        out=np.zeros_like(row_best),
        where=np.isfinite(row_second),
    )
    column_regret = np.divide(
        np.maximum(column_second - column_best, 0.0),
        column_second + eps,
        out=np.zeros_like(column_best),
        where=np.isfinite(column_second),
    )

    row_opportunity = np.clip(
        0.55 + 0.45 * row_regret,
        0.0,
        1.0,
    )
    column_opportunity = np.clip(
        0.55 + 0.45 * column_regret,
        0.0,
        1.0,
    )

    long_term_potential = np.sqrt(
        row_opportunity * column_opportunity.T
    )

    if n > 2:
        two_step = np.full((n, n), np.nan, dtype=float)

        for i in range(n):
            for j in range(n):
                if not valid[i, j]:
                    continue

                intermediate_valid = valid[i] & valid[:, j]
                intermediate_valid[i] = False
                intermediate_valid[j] = False

                if np.any(intermediate_valid):
                    candidate = (
                        safe_distance[i, intermediate_valid]
                        + safe_distance[intermediate_valid, j]
                    )
                    two_step[i, j] = np.nanmin(candidate)

        direct = np.where(valid, safe_distance, np.nan)
        savings = np.divide(
            two_step - direct,
            two_step + direct + eps,
            out=np.zeros_like(distances, dtype=float),
            where=np.isfinite(two_step) & np.isfinite(direct),
        )
        savings = np.clip(savings, 0.0, 1.0)
    else:
        savings = np.zeros_like(distances, dtype=float)

    edge_rarity = np.divide(
        1.0,
        1.0 + row_rank + column_rank.T,
        out=np.zeros_like(distances, dtype=float),
        where=valid,
    )
    edge_rarity = normalize_global(edge_rarity)

    score = (
        0.28 * normalize_rows(local_closeness)
        + 0.18 * normalize_rows(global_closeness)
        + 0.16 * endpoint_agreement
        + 0.18 * long_term_potential
        + 0.12 * normalize_rows(savings)
        + 0.08 * edge_rarity
    )

    score = np.where(valid, np.maximum(score, 0.0), 0.0)
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
