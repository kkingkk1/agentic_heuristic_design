import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric normalized edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    offdiag = ~np.eye(n, dtype=bool)
    valid = finite_nonnegative & finite_nonnegative.T & offdiag

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = 0.5 * (raw + raw.T)
    observed = symmetric[valid]

    positive = observed[observed > 0.0]
    q25, q50, q75 = np.quantile(observed, [0.25, 0.50, 0.75])
    spread = float(q75 - q25)

    if np.isfinite(spread) and spread > 0.0:
        scale = spread
    elif np.isfinite(q50) and q50 > 0.0:
        scale = float(q50)
    elif positive.size:
        scale = float(np.mean(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    upper = float(np.quantile(observed, 0.95))
    if not np.isfinite(upper):
        upper = float(np.max(observed))

    penalty = max(upper, scale) + 18.0 * scale
    distances = np.where(valid, symmetric, penalty).astype(float)
    np.fill_diagonal(distances, penalty)

    eps = max(np.finfo(float).eps * scale, 1e-12)

    reciprocal = 1.0 / (distances + eps)
    reciprocal[~valid] = 0.0

    row_best = np.maximum(
        np.max(reciprocal, axis=1, keepdims=True),
        eps,
    )
    col_best = np.maximum(
        np.max(reciprocal, axis=0, keepdims=True),
        eps,
    )

    relative_closeness = np.sqrt(
        (reciprocal / row_best) * (reciprocal / col_best)
    )

    row_mass = np.maximum(
        reciprocal.sum(axis=1, keepdims=True),
        eps,
    )
    col_mass = np.maximum(
        reciprocal.sum(axis=0, keepdims=True),
        eps,
    )

    mutual_competition = np.sqrt(
        (reciprocal / row_mass) * (reciprocal / col_mass)
    )

    order = np.argsort(distances, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.80 * np.sqrt(float(n)))
    rank_score = np.exp(-ranks / rank_scale)
    rank_score[~valid] = 0.0

    temperature = max(0.70 * scale, eps)
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(distances[valid] / temperature, 60.0)
    )

    affinity_mass = np.maximum(
        affinity.sum(axis=1, keepdims=True),
        eps,
    )
    local_probability = affinity / affinity_mass

    mutual_affinity = np.sqrt(
        np.maximum(local_probability * local_probability.T, 0.0)
    )

    scarcity = 1.0 / np.sqrt(
        np.maximum(affinity_mass[:, 0], eps)
    )
    scarcity /= max(float(np.mean(scarcity)), eps)
    scarcity = np.clip(scarcity, 0.5, 2.0)

    endpoint_scarcity = np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], 0.0)
    )

    nearest = np.min(distances, axis=1)
    second_nearest = np.partition(distances, 1, axis=1)[:, 1]
    local_gap = np.maximum(
        second_nearest - nearest,
        0.18 * scale,
    )
    local_gap = np.maximum(local_gap, eps)

    nearest_mask = distances <= nearest[:, None] + eps
    endpoint_gap = np.where(
        nearest_mask,
        second_nearest[:, None] - distances,
        nearest[:, None] - distances,
    )
    endpoint_gap = np.exp(
        np.clip(endpoint_gap / local_gap[:, None], -24.0, 24.0)
    )
    endpoint_gap[~valid] = 0.0
    endpoint_gap = np.sqrt(
        np.maximum(endpoint_gap * endpoint_gap.T, 0.0)
    )

    bypass = np.full((n, n), penalty, dtype=float)
    block_size = max(1, min(n, 2_000_000 // max(n, 1)))

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block = (
            distances[:, start:stop, None]
            + distances[None, start:stop, :]
        )
        bypass = np.minimum(
            bypass,
            np.min(block, axis=1),
        )

    np.fill_diagonal(bypass, penalty)

    bypass_gain = np.maximum(bypass - distances, 0.0)
    bypass_scale = max(0.80 * scale, eps)
    irreplaceability = 1.0 - np.exp(
        -bypass_gain / bypass_scale
    )
    irreplaceability[~valid] = 0.0

    affinity_max = max(float(np.max(affinity)), eps)
    affinity_score = affinity / affinity_max

    score = (
        0.28 * affinity_score
        + 0.22 * relative_closeness
        + 0.18 * mutual_competition
        + 0.15 * rank_score
        + 0.10 * endpoint_gap
        + 0.07 * irreplaceability
    )

    score *= 0.88 + 0.12 * mutual_affinity
    score *= 0.90 + 0.10 * endpoint_scarcity
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)

    balanced = score.copy()
    for _ in range(32):
        row_sum = np.maximum(
            balanced.sum(axis=1, keepdims=True),
            eps,
        )
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)

    result = 0.64 * score + 0.36 * balanced
    result[~valid] = 0.0
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
