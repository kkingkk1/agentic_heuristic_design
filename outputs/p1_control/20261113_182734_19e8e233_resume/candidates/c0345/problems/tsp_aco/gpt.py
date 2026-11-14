import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced, regret-aware priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    eps = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, eps)

    masked = np.where(valid, d, np.inf)

    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_min = np.min(masked, axis=1)
    col_min = np.min(masked, axis=0)

    row_min = np.where(row_count > 0, row_min, 0.0)
    col_min = np.where(col_count > 0, col_min, 0.0)

    row_delta = np.maximum(d - row_min[:, None], 0.0)
    col_delta = np.maximum(d - col_min[None, :], 0.0)
    row_delta[~valid] = 0.0
    col_delta[~valid] = 0.0

    row_second = np.full(n, scale, dtype=float)
    col_second = np.full(n, scale, dtype=float)

    row_has_two = row_count >= 2
    col_has_two = col_count >= 2

    if np.any(row_has_two):
        row_values = np.where(valid[row_has_two], d[row_has_two], np.inf)
        row_second[row_has_two] = np.partition(
            row_values, kth=1, axis=1
        )[:, 1]

    if np.any(col_has_two):
        col_values = np.where(valid[:, col_has_two], d[:, col_has_two], np.inf)
        col_second[col_has_two] = np.partition(
            col_values, kth=1, axis=0
        )[1, :]

    row_gap = np.maximum(row_second - row_min, 0.0)
    col_gap = np.maximum(col_second - col_min, 0.0)

    row_gap_values = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_gap_values = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(row_gap_values))
        if row_gap_values.size
        else scale
    )
    col_gap_scale = (
        float(np.median(col_gap_values))
        if col_gap_values.size
        else scale
    )

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    regret = np.exp(
        -0.5
        * (
            row_delta / row_gap_scale
            + col_delta / col_gap_scale
        )
    )
    regret[~valid] = 0.0

    temperature = max(0.12 * scale, eps)

    row_logits = np.clip(row_delta / temperature, 0.0, 60.0)
    col_logits = np.clip(col_delta / temperature, 0.0, 60.0)

    row_affinity = np.exp(-row_logits)
    col_affinity = np.exp(-col_logits)

    affinity = np.sqrt(row_affinity * col_affinity)
    affinity[~valid] = 0.0

    balanced = affinity
    for _ in range(96):
        previous = balanced

        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced = np.divide(
            balanced,
            row_sum,
            out=np.zeros_like(balanced),
            where=row_sum > 0.0,
        )

        col_sum = balanced.sum(axis=0, keepdims=True)
        balanced = np.divide(
            balanced,
            col_sum,
            out=np.zeros_like(balanced),
            where=col_sum > 0.0,
        )

        difference = np.max(np.abs(balanced - previous))
        if difference < 1e-12:
            break

    score = balanced * np.power(regret + eps, 1.35)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = np.max(score)
    if maximum > 0.0:
        score = score / maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
