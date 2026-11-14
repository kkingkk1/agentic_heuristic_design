import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diffusion-supported edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    positive = distances[finite & (distances > 0.0)]

    if positive.size:
        fallback = float(np.max(positive))
        scale_floor = max(float(np.median(positive)), np.finfo(float).eps)
    else:
        fallback = 1.0
        scale_floor = 1.0

    d = np.nan_to_num(
        distances,
        nan=fallback,
        posinf=fallback,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    positive_d = d[d > 0.0]
    if positive_d.size:
        fallback = max(float(np.max(positive_d)), np.finfo(float).eps)
    else:
        fallback = 1.0

    invalid = ~np.isfinite(d)
    d = np.where(invalid, fallback, d)

    off_diagonal = ~np.eye(n, dtype=bool)
    nonzero_rows = np.where(
        off_diagonal & (d > 0.0),
        d,
        np.nan,
    )

    row_scale = np.nanmedian(nonzero_rows, axis=1)
    row_scale = np.nan_to_num(
        row_scale,
        nan=scale_floor,
        posinf=scale_floor,
        neginf=scale_floor,
    )
    row_scale = np.maximum(row_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(
        row_scale[:, None] * row_scale[None, :]
    )
    pair_scale = np.maximum(pair_scale, np.finfo(float).eps)

    normalized_distance = d / pair_scale
    direct = np.exp(
        -np.clip(normalized_distance ** 2, 0.0, 50.0)
    )
    direct[~off_diagonal] = 0.0

    density = np.sum(direct, axis=1)
    density = np.maximum(density, np.finfo(float).eps)

    transition = direct / density[:, None]

    shared_support = transition @ transition.T
    shared_support = 0.5 * (
        shared_support + shared_support.T
    )
    shared_support[~off_diagonal] = 0.0

    support_max = float(np.max(shared_support))
    if support_max > 0.0 and np.isfinite(support_max):
        normalized_support = shared_support / support_max
    else:
        normalized_support = np.zeros_like(shared_support)

    density_balance = np.sqrt(
        density[:, None] * density[None, :]
    )
    density_balance = density_balance / max(
        float(np.max(density_balance)),
        np.finfo(float).eps,
    )

    score = (
        direct
        * (0.45 + 0.55 * normalized_support)
        * (0.85 + 0.15 * density_balance)
    )

    score = 0.5 * (score + score.T)
    score[~off_diagonal] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
