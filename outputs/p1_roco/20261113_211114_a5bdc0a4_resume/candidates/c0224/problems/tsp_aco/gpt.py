import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(raw)
    offdiag = ~np.eye(n, dtype=bool)
    valid = offdiag & (finite | finite.T)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    both_finite = finite & finite.T
    cleaned = np.where(
        both_finite,
        0.5 * (raw + raw.T),
        np.where(finite, raw, raw.T),
    )
    cleaned = np.maximum(cleaned, 0.0)

    observed = cleaned[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    maximum_distance = float(np.max(observed))
    penalty = maximum_distance + 10.0 * scale

    distances = np.where(valid, cleaned, penalty)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, penalty)

    temperature = float(np.median(distances[valid]))
    temperature = max(0.30 * temperature, np.finfo(float).eps)

    affinity = np.exp(-distances / temperature)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    maximum_affinity = float(np.max(affinity))
    if not np.isfinite(maximum_affinity) or maximum_affinity <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    base = affinity / maximum_affinity

    work = distances.copy()
    nearest = np.min(work, axis=1)
    second_nearest = np.partition(work, 1, axis=1)[:, 1]

    local_scale = np.maximum(
        second_nearest - nearest,
        0.20 * scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    nearest_edge = work <= nearest[:, None]
    endpoint_gap = np.where(
        nearest_edge,
        second_nearest[:, None] - work,
        nearest[:, None] - work,
    )
    endpoint_gap = np.exp(
        np.clip(endpoint_gap / local_scale[:, None], -25.0, 25.0)
    )
    endpoint_gap[~valid] = 0.0
    endpoint_gap = np.sqrt(endpoint_gap * endpoint_gap.T)

    best_bypass = np.full((n, n), penalty, dtype=float)

    # Keep the temporary tensor below roughly two million elements.
    block_size = max(1, min(n, 2_000_000 // max(n * n, 1)))

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        candidate = (
            distances[:, start:stop, None]
            + distances[None, start:stop, :]
        )
        block_minimum = np.min(candidate, axis=1)
        np.minimum(best_bypass, block_minimum, out=best_bypass)

    np.fill_diagonal(best_bypass, penalty)

    bypass_advantage = np.maximum(best_bypass - distances, 0.0)
    bypass_scale = max(0.85 * scale, np.finfo(float).eps)
    irreplaceability = 1.0 - np.exp(
        -bypass_advantage / bypass_scale
    )
    irreplaceability[~valid] = 0.0

    score = (
        0.45 * base
        + 0.32 * endpoint_gap
        + 0.23 * irreplaceability
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    epsilon = np.finfo(float).eps

    for _ in range(32):
        row_sum = np.maximum(
            balanced.sum(axis=1, keepdims=True),
            epsilon,
        )
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)

    result = 0.62 * score + 0.38 * balanced
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
    )
