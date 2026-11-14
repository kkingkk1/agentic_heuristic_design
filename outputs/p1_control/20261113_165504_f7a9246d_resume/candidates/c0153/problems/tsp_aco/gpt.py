import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return graph-diffusion and triangle-closure priors for TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]

    if positive.size:
        global_temperature = float(np.median(positive))
    else:
        finite = d[valid]
        global_temperature = (
            float(np.mean(finite)) if finite.size else 1.0
        )

    global_temperature = max(global_temperature, eps)

    safe = np.where(valid, d, np.inf)

    row_mean = np.zeros(n, dtype=float)
    row_count = np.sum(valid, axis=1)
    np.sum(np.where(valid, d, 0.0), axis=1, out=row_mean)
    row_mean = np.divide(
        row_mean,
        np.maximum(row_count, 1),
        out=np.full(n, global_temperature, dtype=float),
        where=row_count > 0,
    )

    row_sq = np.zeros(n, dtype=float)
    np.sum(
        np.where(valid, (d - row_mean[:, None]) ** 2, 0.0),
        axis=1,
        out=row_sq,
    )
    row_std = np.sqrt(
        np.divide(
            row_sq,
            np.maximum(row_count, 1),
            out=np.zeros(n, dtype=float),
            where=row_count > 0,
        )
    )

    temperature = np.maximum(
        0.5 * row_std + 0.5 * row_mean,
        0.15 * global_temperature,
    )
    temperature = np.maximum(temperature, eps)

    scaled = np.zeros_like(d, dtype=float)
    np.divide(
        d,
        np.sqrt(temperature[:, None] * temperature[None, :]),
        out=scaled,
        where=valid,
    )

    affinity = np.exp(-np.clip(scaled, 0.0, 50.0))
    affinity[~valid] = 0.0

    row_mass = affinity.sum(axis=1, keepdims=True)
    transition = np.divide(
        affinity,
        row_mass,
        out=np.zeros_like(affinity),
        where=row_mass > 0.0,
    )

    column_mass = affinity.sum(axis=0, keepdims=True)
    reverse_transition = np.divide(
        affinity,
        column_mass,
        out=np.zeros_like(affinity),
        where=column_mass > 0.0,
    )

    two_step = transition @ reverse_transition.T
    two_step[~valid] = 0.0

    closure_scale = np.maximum(
        np.max(two_step, axis=1, keepdims=True),
        eps,
    )
    closure = np.divide(
        two_step,
        np.sqrt(closure_scale * closure_scale.T),
        out=np.zeros_like(two_step),
        where=(closure_scale * closure_scale.T) > 0.0,
    )
    closure = np.clip(closure, 0.0, 1.0)

    bidirectional = np.sqrt(
        np.maximum(transition * reverse_transition, 0.0)
    )
    bidirectional[~valid] = 0.0

    direct_scale = np.maximum(
        np.max(affinity, axis=1, keepdims=True),
        eps,
    )
    direct = np.divide(
        affinity,
        np.sqrt(direct_scale * direct_scale.T),
        out=np.zeros_like(affinity),
        where=(direct_scale * direct_scale.T) > 0.0,
    )
    direct = np.clip(direct, 0.0, 1.0)

    bridge = direct * (1.0 - closure)
    score = (
        0.46 * closure
        + 0.34 * bidirectional
        + 0.15 * direct
        + 0.05 * bridge
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
