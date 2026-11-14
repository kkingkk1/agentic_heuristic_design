import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d) & (d > 0)
    valid &= valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    fallback = max(float(np.max(positive_values)), global_scale)

    safe = np.where(valid, d, np.inf)

    first_k = 0
    second_k = min(1, n - 2)

    nearest = np.partition(safe, first_k, axis=1)[:, first_k]
    second_nearest = np.partition(safe, second_k, axis=1)[:, second_k]

    nearest = np.where(np.isfinite(nearest), nearest, global_scale)
    second_nearest = np.where(
        np.isfinite(second_nearest),
        second_nearest,
        np.maximum(nearest, global_scale),
    )

    nearest = np.maximum(nearest, eps)
    second_nearest = np.maximum(second_nearest, nearest)

    safe_distance = np.where(
        np.isfinite(safe),
        safe,
        fallback * 10.0,
    )
    safe_distance = np.maximum(safe_distance, eps)

    row_scale = np.sqrt(nearest[:, None] * nearest[None, :])
    second_scale = np.sqrt(
        second_nearest[:, None] * second_nearest[None, :]
    )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        relative_length = safe_distance / np.maximum(row_scale, eps)
        global_length = safe_distance / global_scale

        short_edge = 1.0 / (1.0 + 0.85 * relative_length)
        global_preference = 0.72 + 0.28 * np.exp(-0.22 * global_length)

        endpoint_rank = 0.5 * (
            safe_distance / nearest[:, None]
            + safe_distance / nearest[None, :]
        )
        second_neighbor_rank = 0.5 * (
            safe_distance / second_nearest[:, None]
            + safe_distance / second_nearest[None, :]
        )

        mutual_neighborhood = np.exp(
            -0.62 * endpoint_rank - 0.24 * second_neighbor_rank
        )

        local_difference = np.abs(
            np.log((nearest[:, None] + eps) / (nearest[None, :] + eps))
        )
        density_balance = 1.0 / (1.0 + 0.55 * local_difference)
        density_factor = 0.84 + 0.16 * density_balance

        score = (
            short_edge
            * mutual_neighborhood
            * global_preference
            * density_factor
        )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score.astype(float, copy=False)
