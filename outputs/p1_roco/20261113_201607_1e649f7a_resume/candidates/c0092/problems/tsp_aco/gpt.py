import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return entropy-regularized 2-factor edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(raw)
    nonnegative = raw >= 0.0
    valid = finite & nonnegative
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_valid = valid & valid.T
    if not np.any(symmetric_valid):
        symmetric_valid = valid | valid.T

    cleaned = np.where(valid, raw, np.inf)
    reverse = np.where(valid.T, raw.T, np.inf)

    both = valid & valid.T
    symmetric_cost = np.where(
        both,
        0.5 * (cleaned + reverse),
        np.minimum(cleaned, reverse),
    )
    symmetric_cost[~symmetric_valid] = np.inf
    np.fill_diagonal(symmetric_cost, np.inf)

    observed = symmetric_cost[np.isfinite(symmetric_cost)]
    positive = observed[observed > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    elif observed.size:
        scale = float(np.mean(observed))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_cost = observed[np.isfinite(observed)]
    upper = float(np.max(finite_cost)) if finite_cost.size else scale
    penalty = upper + 8.0 * scale

    cost = np.where(np.isfinite(symmetric_cost), symmetric_cost, penalty)
    np.fill_diagonal(cost, penalty)

    spread = float(np.median(np.abs(finite_cost - np.median(finite_cost))))
    temperature = max(
        0.25 * scale + 0.75 * spread,
        np.sqrt(np.finfo(float).eps) * max(scale, 1.0),
    )

    affinity = np.exp(
        -np.clip(cost / temperature, 0.0, 60.0)
    )
    affinity[~symmetric_valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    if not np.any(affinity > 0.0):
        return symmetric_valid.astype(float)

    scaled = affinity.copy()

    # Entropy-regularized matrix balancing with forbidden diagonal entries.
    for _ in range(80):
        row_sum = scaled.sum(axis=1)
        row_scale = np.divide(
            1.0,
            row_sum,
            out=np.zeros_like(row_sum),
            where=row_sum > 0.0,
        )
        scaled *= row_scale[:, None]

        col_sum = scaled.sum(axis=0)
        col_scale = np.divide(
            1.0,
            col_sum,
            out=np.zeros_like(col_sum),
            where=col_sum > 0.0,
        )
        scaled *= col_scale[None, :]

        scaled[~symmetric_valid] = 0.0
        np.fill_diagonal(scaled, 0.0)

    reciprocal_support = np.sqrt(
        np.maximum(scaled * scaled.T, 0.0)
    )

    affinity_max = float(np.max(affinity))
    if np.isfinite(affinity_max) and affinity_max > 0.0:
        affinity /= affinity_max

    support_max = float(np.max(reciprocal_support))
    if np.isfinite(support_max) and support_max > 0.0:
        reciprocal_support /= support_max

    result = 0.35 * affinity + 0.65 * reciprocal_support
    result = 0.5 * (result + result.T)
    result[~symmetric_valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum
    else:
        result.fill(0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
