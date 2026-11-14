import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    work = np.where(valid, d, np.inf)
    sorted_rows = np.sort(work, axis=1)

    k = min(4, n - 1)
    neighbors = sorted_rows[:, :k]
    finite_neighbors = np.isfinite(neighbors)
    neighbor_count = finite_neighbors.sum(axis=1)

    neighbor_sum = np.where(finite_neighbors, neighbors, 0.0).sum(axis=1)
    local_scale = np.divide(
        neighbor_sum,
        np.maximum(neighbor_count, 1),
        out=np.ones(n, dtype=float),
        where=neighbor_count > 0,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    finite_values = d[valid]
    global_scale = float(np.median(finite_values))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        positive = finite_values[finite_values > 0.0]
        global_scale = (
            float(np.median(positive))
            if positive.size
            else 1.0
        )
    global_scale = max(global_scale, np.finfo(float).eps)

    best = sorted_rows[:, 0]
    second = sorted_rows[:, 1] if n > 2 else best.copy()

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(np.isfinite(second), second, 2.0 * safe_best)

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, np.finfo(float).eps),
        0.0,
        1.0,
    )

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    local_distance = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    global_distance = np.divide(
        d,
        global_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    row_best = safe_best[:, None]
    row_second = safe_second[:, None]
    col_best = safe_best[None, :]
    col_second = safe_second[None, :]

    alternative_i = np.where(d <= row_best, row_second, row_best)
    alternative_j = np.where(d <= col_best, col_second, col_best)

    lookahead = d + 0.5 * (alternative_i + alternative_j)
    lookahead_scale = max(2.0 * global_scale, np.finfo(float).eps)
    normalized_lookahead = lookahead / lookahead_scale

    local_quality = np.exp(
        -np.clip(local_distance, 0.0, 50.0) / 0.92
    )
    global_quality = np.exp(
        -np.clip(global_distance, 0.0, 50.0) / 1.45
    )
    continuation_quality = np.exp(
        -np.clip(normalized_lookahead, 0.0, 50.0) / 1.20
    )

    row_competition = np.sum(
        valid & (d <= row_second),
        axis=1,
        dtype=float,
    )
    col_competition = np.sum(
        valid & (d <= col_second),
        axis=0,
        dtype=float,
    )

    endpoint_bonus = 1.0 + 0.06 / np.sqrt(
        np.maximum(row_competition[:, None], 1.0)
        * np.maximum(col_competition[None, :], 1.0)
    )

    urgency_bonus = 1.0 + 0.10 * regret[:, None]
    urgency_bonus = urgency_bonus * (1.0 + 0.10 * regret[None, :])

    score = (
        0.78 * local_quality
        + 0.22 * global_quality
    )
    score *= continuation_quality
    score *= endpoint_bonus
    score *= urgency_bonus
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    symmetric_input = (
        np.array_equal(valid, valid.T)
        and np.allclose(
            np.where(valid, d, 0.0),
            np.where(valid.T, d.T, 0.0),
            rtol=1e-10,
            atol=1e-12,
        )
    )

    if symmetric_input:
        score = np.sqrt(score * score.T)
        for _ in range(4):
            row_mass = score.sum(axis=1)
            target_mass = float(np.mean(row_mass[row_mass > 0.0]))
            factors = np.divide(
                target_mass,
                np.maximum(row_mass, np.finfo(float).eps),
            )
            factors = np.power(factors, 0.18)
            score *= factors[:, None] * factors[None, :]
            score[~valid] = 0.0
    else:
        for _ in range(3):
            column_mass = score.sum(axis=0)
            positive_mass = column_mass[column_mass > 0.0]
            if positive_mass.size == 0:
                break
            target_mass = float(np.mean(positive_mass))
            correction = np.divide(
                target_mass,
                np.maximum(column_mass, np.finfo(float).eps),
            )
            score *= np.power(correction[None, :], 0.16)
            score[~valid] = 0.0

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=result,
        where=totals > 0.0,
    )
