import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric soft edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = 0.5 * (raw + raw.T)
    distances[~valid] = np.inf
    np.fill_diagonal(distances, np.inf)

    observed = distances[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)
    maximum = float(np.max(observed))
    penalty = maximum + 12.0 * safe_scale

    finite_distances = np.where(valid, distances, penalty)
    np.fill_diagonal(finite_distances, penalty)

    temperature = max(
        0.65 * float(np.median(observed)),
        0.25 * safe_scale,
        eps,
    )

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(finite_distances[valid], penalty) / temperature
    )

    row_mass = affinity.sum(axis=1, keepdims=True)
    col_mass = affinity.sum(axis=0, keepdims=True)

    row_share = affinity / np.maximum(row_mass, eps)
    col_share = affinity / np.maximum(col_mass, eps)
    mutual_support = np.sqrt(np.maximum(row_share * col_share, 0.0))

    row_best = np.min(finite_distances, axis=1)
    row_second = np.partition(finite_distances, 1, axis=1)[:, 1]
    col_best = np.min(finite_distances, axis=0)
    col_second = np.partition(finite_distances.T, 1, axis=1)[:, 1]

    row_gap = np.maximum(row_second[:, None] - finite_distances, 0.0)
    col_gap = np.maximum(col_second[None, :] - finite_distances, 0.0)

    gap_values = np.concatenate(
        [
            (row_second - row_best)[row_second > row_best],
            (col_second - col_best)[col_second > col_best],
        ]
    )
    gap_scale = float(np.median(gap_values)) if gap_values.size else safe_scale
    gap_scale = max(gap_scale, 0.1 * safe_scale, eps)

    exclusivity = 1.0 - np.exp(
        -0.5 * (row_gap + col_gap) / gap_scale
    )
    exclusivity[~valid] = 0.0

    bypass = np.full((n, n), penalty, dtype=float)
    for k in range(n):
        candidate = (
            finite_distances[:, k, None]
            + finite_distances[None, k, :]
        )
        bypass = np.minimum(bypass, candidate)

    bypass[diagonal] = penalty
    bypass_advantage = np.maximum(bypass - finite_distances, 0.0)

    bridge_scale = max(0.85 * safe_scale, eps)
    bridge = 1.0 - np.exp(-bypass_advantage / bridge_scale)
    bridge[~valid] = 0.0

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (finite_distances[valid] + safe_scale * 1e-9)
    inverse /= max(float(np.max(inverse)), eps)

    scarcity = 1.0 / np.maximum(row_mass[:, 0], eps)
    scarcity /= max(float(np.mean(scarcity)), eps)
    endpoint_support = np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], 0.0)
    )
    endpoint_support = np.clip(endpoint_support, 0.0, 3.0)

    score = (
        np.power(np.maximum(inverse, eps), 0.46)
        * np.power(np.maximum(mutual_support, eps), 0.22)
        * np.power(np.maximum(0.25 + exclusivity, eps), 0.16)
        * np.power(np.maximum(0.25 + bridge, eps), 0.16)
        * np.power(np.maximum(0.5 + 0.2 * endpoint_support, eps), 0.12)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    for _ in range(16):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced /= np.maximum(row_sum, eps)
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0

    exploration = 0.06
    valid_count = float(np.sum(valid))
    exploratory = valid.astype(float) / max(valid_count, 1.0)

    result = (
        0.68 * score
        + 0.24 * balanced
        + 0.08 * exploratory
    )
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
