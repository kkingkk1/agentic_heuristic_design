import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diffusion-based soft edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    nonnegative = raw >= 0.0
    valid = finite & nonnegative & ~diagonal
    valid &= valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(symmetric, np.inf)

    observed = symmetric[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    spread = float(np.median(np.abs(observed - np.median(observed))))
    spread = max(spread, 0.15 * scale, np.finfo(float).eps)
    temperature = 0.8 * scale + 0.7 * spread
    temperature = max(temperature, np.finfo(float).eps)

    kernel = np.zeros((n, n), dtype=float)
    kernel[valid] = np.exp(
        -np.clip(symmetric[valid] / temperature, 0.0, 60.0)
    )

    degree = kernel.sum(axis=1)
    active = degree > np.finfo(float).eps

    transition = np.zeros((n, n), dtype=float)
    transition[active] = kernel[active] / degree[active, None]

    two_step = transition @ transition
    three_step = two_step @ transition

    diffusion = np.sqrt(
        np.maximum(two_step * three_step.T, 0.0)
    )
    diffusion = 0.5 * (diffusion + diffusion.T)

    activity = np.sqrt(
        np.maximum(degree[:, None] * degree[None, :], 0.0)
    )
    activity_scale = max(float(np.max(activity)), np.finfo(float).eps)
    activity /= activity_scale

    direct = np.zeros((n, n), dtype=float)
    direct[valid] = kernel[valid]
    direct_max = max(float(np.max(direct)), np.finfo(float).eps)
    direct /= direct_max

    diffusion_max = max(float(np.max(diffusion)), np.finfo(float).eps)
    diffusion /= diffusion_max

    rank_values = np.zeros((n, n), dtype=float)
    sorted_values = np.sort(observed)
    ranks = np.searchsorted(sorted_values, observed, side="right")
    rank_values[valid] = 1.0 - (
        ranks.astype(float) / max(float(sorted_values.size), 1.0)
    )

    rank_values[valid] = 0.12 + 0.88 * rank_values[valid]

    # Correct for overly active nodes while preserving their structural signal.
    activity_factor = 1.0 / np.sqrt(
        np.maximum(activity, np.finfo(float).eps)
    )
    activity_factor = np.clip(activity_factor, 0.15, 2.0)

    score = (
        0.38 * direct
        + 0.37 * diffusion
        + 0.15 * rank_values
        + 0.10 * activity_factor
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    # A small diffusion floor prevents complete concentration on one edge family.
    positive_score = score[valid]
    floor = 0.025 * float(np.median(positive_score))
    if np.isfinite(floor) and floor > 0.0:
        score[valid] = score[valid] + floor

    balanced = score.copy()
    eps = np.finfo(float).eps

    for _ in range(10):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced /= np.maximum(row_sum, eps)
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0

    result = 0.68 * score + 0.32 * balanced
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)
    result = 0.5 * (result + result.T)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
