import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diffusion-based edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    valid = finite & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    symmetric_valid = valid & valid.T
    if not np.any(symmetric_valid):
        return np.zeros((n, n), dtype=float)

    valid_values = distances[symmetric_valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_values = valid_values[np.isfinite(valid_values)]
    if finite_values.size:
        fallback = float(np.max(finite_values))
    else:
        fallback = scale

    if not np.isfinite(fallback) or fallback < 0.0:
        fallback = scale

    symmetric_distance = 0.5 * (
        distances + distances.T
    )
    work_distance = np.where(
        symmetric_valid,
        symmetric_distance,
        fallback,
    )
    work_distance = np.maximum(work_distance, 0.0)

    np.fill_diagonal(work_distance, fallback)

    eps = np.finfo(float).eps
    normalized = work_distance / max(scale, eps)

    kernel = np.exp(
        -np.clip(np.sqrt(normalized), 0.0, 40.0)
    )
    kernel = np.nan_to_num(
        kernel,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(kernel, 0.0)

    row_mass = np.sum(kernel, axis=1)
    row_mass = np.maximum(row_mass, eps)

    transition = kernel / row_mass[:, None]

    two_step = transition @ transition
    two_step = 0.5 * (two_step + two_step.T)
    two_step = np.nan_to_num(
        two_step,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    two_step_max = float(np.max(two_step))
    if two_step_max > 0.0 and np.isfinite(two_step_max):
        diffusion = two_step / two_step_max
    else:
        diffusion = np.zeros_like(two_step)

    flow_balance = np.sqrt(
        row_mass[:, None] * row_mass[None, :]
    )
    flow_balance /= max(float(np.max(flow_balance)), eps)
    flow_balance = np.clip(flow_balance, 0.0, 1.0)

    direct = kernel
    direct_max = float(np.max(direct))
    if direct_max > 0.0 and np.isfinite(direct_max):
        direct = direct / direct_max

    score = direct * (
        0.62
        + 0.28 * diffusion
        + 0.10 * flow_balance
    )

    score = 0.5 * (score + score.T)
    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
