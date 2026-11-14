import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diverse, long-term-aware edge priorities for TSP sampling."""
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

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)
    sorted_rows = np.sort(work, axis=1)

    best = sorted_rows[:, 0]
    second = sorted_rows[:, 1] if n > 2 else best.copy()

    safe_best = np.where(np.isfinite(best), best, 1.0)
    safe_second = np.where(
        np.isfinite(second),
        second,
        np.maximum(2.0 * safe_best, safe_best + eps),
    )

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    local_scale = np.minimum(
        safe_best + 0.5 * safe_second,
        np.maximum(safe_second, eps),
    )
    local_scale = np.maximum(local_scale, eps)

    row_distance = d / np.maximum(local_scale[:, None], eps)
    global_distance = d / global_scale

    local_quality = np.exp(
        -np.clip(row_distance, 0.0, 60.0) / 0.82
    )
    global_quality = np.exp(
        -np.clip(global_distance, 0.0, 60.0) / 1.65
    )

    alternative_i = np.where(
        d <= safe_best[:, None],
        safe_second[:, None],
        safe_best[:, None],
    )
    alternative_j = np.where(
        d <= safe_best[None, :],
        safe_second[None, :],
        safe_best[None, :],
    )

    long_term_cost = d + 0.42 * (alternative_i + alternative_j)
    long_term_quality = np.exp(
        -np.clip(long_term_cost / (2.0 * global_scale), 0.0, 60.0)
        / 1.08
    )

    row_rank = np.sum(
        valid & (d <= safe_second[:, None]),
        axis=1,
        dtype=float,
    )
    col_rank = np.sum(
        valid & (d <= safe_second[None, :]),
        axis=0,
        dtype=float,
    )

    row_scarcity = 1.0 / np.sqrt(np.maximum(row_rank, 1.0))
    col_scarcity = 1.0 / np.sqrt(np.maximum(col_rank, 1.0))
    scarcity_bonus = 1.0 + 0.16 * (
        row_scarcity[:, None] + col_scarcity[None, :]
    )

    row_regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, eps),
        0.0,
        1.0,
    )
    urgency_bonus = (
        1.0
        + 0.08 * row_regret[:, None]
        + 0.08 * row_regret[None, :]
    )

    reverse_distance = np.where(valid.T, d.T, np.inf)
    reverse_quality = np.exp(
        -np.clip(reverse_distance / global_scale, 0.0, 60.0) / 1.35
    )
    mutual_quality = np.sqrt(
        np.maximum(local_quality, 0.0)
        * np.maximum(reverse_quality, 0.0)
    )
    mutual_quality = np.where(valid, mutual_quality, 0.0)

    score = (
        0.68 * local_quality
        + 0.16 * global_quality
        + 0.16 * long_term_quality
    )
    score *= 0.90 + 0.10 * mutual_quality
    score *= scarcity_bonus
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
        score = np.sqrt(np.maximum(score, 0.0) * np.maximum(score.T, 0.0))

        for _ in range(5):
            row_mass = score.sum(axis=1)
            positive = row_mass > 0.0
            if not np.any(positive):
                break

            target = float(np.mean(row_mass[positive]))
            factors = np.divide(
                target,
                np.maximum(row_mass, eps),
            )
            factors = np.power(factors, 0.16)
            score *= factors[:, None] * factors[None, :]
            score[~valid] = 0.0
    else:
        for _ in range(4):
            row_mass = score.sum(axis=1, keepdims=True)
            score = np.divide(
                score,
                np.maximum(row_mass, eps),
                out=np.zeros_like(score),
                where=row_mass > 0.0,
            )

            column_mass = score.sum(axis=0)
            positive = column_mass > 0.0
            if not np.any(positive):
                break

            target = float(np.mean(column_mass[positive]))
            correction = np.divide(
                target,
                np.maximum(column_mass, eps),
            )
            score *= np.power(correction[None, :], 0.12)
            score[~valid] = 0.0

    degree = valid.sum(axis=1, keepdims=True).astype(float)
    uniform_prior = np.divide(
        valid.astype(float),
        np.maximum(degree, 1.0),
    )

    score = 0.92 * score + 0.08 * uniform_prior
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=result,
        where=totals > 0.0,
    )
