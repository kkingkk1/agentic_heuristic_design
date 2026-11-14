import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed edge desirability indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(distances) & (distances > 0)
    valid &= ~np.eye(n, dtype=bool)

    positive = distances[valid]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    outgoing = np.where(valid, distances, np.nan)
    incoming = np.where(valid, distances, np.nan).T

    out_scale = np.nanmedian(outgoing, axis=1)
    in_scale = np.nanmedian(incoming, axis=1)

    global_scale = float(np.median(positive))
    out_scale = np.where(
        np.isfinite(out_scale) & (out_scale > 0),
        out_scale,
        global_scale,
    )
    in_scale = np.where(
        np.isfinite(in_scale) & (in_scale > 0),
        in_scale,
        global_scale,
    )

    endpoint_scale = np.sqrt(
        np.maximum(out_scale[:, None] * in_scale[None, :], eps)
    )
    relative_distance = distances / endpoint_scale

    relative_distance = np.where(
        valid,
        np.maximum(relative_distance, eps),
        np.inf,
    )

    base = 1.0 / relative_distance

    out_valid_count = np.maximum(valid.sum(axis=1), 1)
    in_valid_count = np.maximum(valid.sum(axis=0), 1)

    out_best = np.where(valid, distances, np.inf).min(axis=1)
    in_best = np.where(valid, distances, np.inf).min(axis=0)

    out_regret = np.where(
        valid,
        out_best[:, None] / np.maximum(distances, eps),
        0.0,
    )
    in_regret = np.where(
        valid,
        in_best[None, :] / np.maximum(distances, eps),
        0.0,
    )

    out_affinity = np.exp(
        -0.5 * np.maximum(relative_distance - 1.0, 0.0)
    )
    in_relative = distances / np.sqrt(
        np.maximum(in_scale[:, None] * out_scale[None, :], eps)
    )
    in_affinity = np.exp(
        -0.5 * np.maximum(in_relative.T - 1.0, 0.0)
    )

    score = base
    score *= 0.65 + 0.35 * np.sqrt(
        np.maximum(out_regret, 0.0) * np.maximum(in_regret, 0.0)
    )
    score *= 0.75 + 0.25 * np.sqrt(out_affinity * in_affinity)
    score *= valid

    row_max = np.max(score, axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    score /= row_max

    density = np.sqrt(
        out_valid_count[:, None] / float(n - 1)
        * in_valid_count[None, :] / float(n - 1)
    )
    score *= 0.9 + 0.1 * density
    score *= valid

    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
