import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic edge-selection priors for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(d) & (d >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    values = d[valid]
    q25, q50, q75, q90, q99 = np.quantile(
        values,
        [0.25, 0.50, 0.75, 0.90, 0.99],
    )

    spread = float(q75 - q25)
    positive = values[values > 0.0]

    scale = spread if np.isfinite(spread) and spread > 0.0 else float(q50)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.mean(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    upper = max(float(q99), float(q90), scale)
    penalty = upper + 20.0 * scale

    clean = np.where(valid, d, penalty).astype(float, copy=True)
    np.fill_diagonal(clean, penalty)

    eps = max(np.sqrt(np.finfo(float).eps) * scale, 1e-12)

    normalized = np.minimum(clean / max(scale, eps), 64.0)
    reciprocal = 1.0 / (clean + eps)
    reciprocal[~valid] = 0.0

    row_max = np.maximum(reciprocal.max(axis=1, keepdims=True), eps)
    col_max = np.maximum(reciprocal.max(axis=0, keepdims=True), eps)

    row_strength = reciprocal / row_max
    col_strength = reciprocal / col_max

    row_mass = np.maximum(reciprocal.sum(axis=1, keepdims=True), eps)
    col_mass = np.maximum(reciprocal.sum(axis=0, keepdims=True), eps)

    row_competition = reciprocal / row_mass
    col_competition = reciprocal / col_mass

    order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.95 * np.sqrt(float(n)))
    rank_score = np.exp(-ranks / rank_scale)
    rank_score[~valid] = 0.0

    affinity = np.exp(-normalized)
    affinity[~valid] = 0.0

    nearest = clean.min(axis=1)
    second = np.partition(clean, 1, axis=1)[:, 1]
    local_scale = np.maximum(second - nearest, 0.14 * scale)
    local_scale = np.maximum(local_scale, eps)

    nearest_mask = clean <= nearest[:, None]
    row_gap = np.where(
        nearest_mask,
        second[:, None] - clean,
        nearest[:, None] - clean,
    )

    endpoint_row = np.exp(
        np.clip(row_gap / local_scale[:, None], -26.0, 26.0)
    )
    endpoint_row[~valid] = 0.0

    endpoint_col = endpoint_row.T
    endpoint_gap = np.sqrt(endpoint_row * endpoint_col)
    endpoint_gap[~valid] = 0.0

    affinity_max = float(affinity.max())
    if affinity_max > 0.0 and np.isfinite(affinity_max):
        affinity /= affinity_max

    base = (
        0.25 * np.sqrt(row_strength * col_strength)
        + 0.20 * np.sqrt(row_competition * col_competition)
        + 0.13 * rank_score
        + 0.22 * affinity
        + 0.20 * endpoint_gap
    )
    base[~valid] = 0.0

    best_bypass = np.full((n, n), penalty, dtype=float)

    block_size = max(1, min(n, 2_000_000 // max(n, 1)))
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        via_block = (
            clean[:, start:stop, None]
            + clean[None, start:stop, :]
        )
        best_bypass = np.minimum(
            best_bypass,
            np.min(via_block, axis=1),
        )

    np.fill_diagonal(best_bypass, penalty)

    bypass_gain = np.maximum(best_bypass - clean, 0.0)
    bypass_scale = max(0.65 * scale, eps)
    irreplaceability = 1.0 - np.exp(
        -bypass_gain / bypass_scale
    )
    irreplaceability[~valid] = 0.0

    score = (
        0.40 * base
        + 0.36 * endpoint_gap
        + 0.24 * irreplaceability
    )
    score[~valid] = 0.0

    balanced = score.copy()
    for _ in range(36):
        row_sum = np.maximum(
            balanced.sum(axis=1, keepdims=True),
            eps,
        )
        balanced /= row_sum

        col_sum = np.maximum(
            balanced.sum(axis=0, keepdims=True),
            eps,
        )
        balanced /= col_sum

    result = 0.58 * score + 0.42 * balanced
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(result.max())
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
