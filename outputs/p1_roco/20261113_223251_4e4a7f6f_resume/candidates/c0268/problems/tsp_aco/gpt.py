import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return entropy-regularized soft 2-regular TSP edge priors."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(raw)
        & (raw >= 0.0)
        & ~diagonal
    )
    valid = directed_valid & directed_valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distance = np.full((n, n), np.inf, dtype=float)
    distance[valid] = 0.5 * (
        raw[valid] + raw.T[valid]
    )
    np.fill_diagonal(distance, np.inf)

    values = distance[valid]
    positive = values[values > 0.0]

    if positive.size:
        median = float(np.median(positive))
        mad = float(np.median(np.abs(positive - median)))
        q10, q90 = np.quantile(positive, [0.10, 0.90])
        spread = max(
            float(q90 - q10),
            2.0 * mad,
            median * 1e-3,
        )
    else:
        median = 1.0
        spread = 1.0

    if not np.isfinite(spread) or spread <= 0.0:
        spread = 1.0

    eps = np.finfo(float).eps
    safe_distance = np.where(valid, distance, 0.0)

    temperature = max(
        0.55 * spread,
        0.05 * max(median, eps),
        eps,
    )

    kernel = np.zeros((n, n), dtype=float)
    kernel[valid] = np.exp(
        -np.clip(
            safe_distance[valid] / temperature,
            0.0,
            60.0,
        )
    )

    positive_kernel = kernel[valid]
    if positive_kernel.size:
        kernel_floor = max(
            float(np.quantile(positive_kernel, 0.20)) * 0.015,
            1e-12,
        )
    else:
        kernel_floor = 1e-12

    kernel[valid] += kernel_floor
    kernel[~valid] = 0.0
    kernel = 0.5 * (kernel + kernel.T)
    np.fill_diagonal(kernel, 0.0)

    row_mass = kernel.sum(axis=1)
    active = row_mass > eps

    if not np.any(active):
        return np.zeros((n, n), dtype=float)

    scaled = kernel.copy()

    target_degree = 2.0
    for _ in range(80):
        current = scaled.sum(axis=1)
        factors = np.ones(n, dtype=float)
        factors[active] = np.sqrt(
            target_degree
            / np.maximum(current[active], eps)
        )

        scaled *= factors[:, None]
        scaled *= factors[None, :]
        scaled[~valid] = 0.0
        scaled = 0.5 * (scaled + scaled.T)
        np.fill_diagonal(scaled, 0.0)

        current = scaled.sum(axis=1)
        if np.any(active):
            error = np.max(
                np.abs(current[active] - target_degree)
            )
            if error <= 1e-8:
                break

    degree = scaled.sum(axis=1)

    common_support = scaled @ scaled.T
    support_scale = np.sqrt(
        np.maximum(degree[:, None] * degree[None, :], eps)
    )
    normalized_common = common_support / support_scale
    normalized_common[~np.isfinite(normalized_common)] = 0.0

    isolation = 1.0 / np.sqrt(
        np.maximum(degree, eps)
    )
    isolation /= max(float(np.mean(isolation)), eps)
    isolation = np.clip(isolation, 0.65, 1.55)
    endpoint_factor = np.sqrt(
        isolation[:, None] * isolation[None, :]
    )

    local_competition = np.zeros((n, n), dtype=float)
    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        total = float(np.sum(kernel[i, neighbors]))
        if total <= eps:
            continue

        local_competition[i, neighbors] = (
            kernel[i, neighbors] / total
        )

    reciprocal = np.sqrt(
        np.maximum(
            local_competition * local_competition.T,
            0.0,
        )
    )

    baseline = np.zeros((n, n), dtype=float)
    baseline[valid] = 1.0 / (
        1.0 + safe_distance[valid] / max(spread, eps)
    )

    cycle_penalty = 1.0 / (
        1.0 + 0.75 * normalized_common
    )

    score = scaled.copy()
    score *= 0.72 + 0.28 * reciprocal
    score *= 0.74 + 0.26 * baseline
    score *= 0.78 + 0.22 * endpoint_factor
    score *= cycle_penalty

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
