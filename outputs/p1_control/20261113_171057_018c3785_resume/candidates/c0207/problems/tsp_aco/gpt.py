import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return structural shortcut priors for directed TSP edges."""
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

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe = np.where(valid, d, np.inf)

    outgoing_count = valid.sum(axis=1).astype(float)
    incoming_count = valid.sum(axis=0).astype(float)

    outgoing_mass = np.where(valid, d, 0.0).sum(axis=1)
    incoming_mass = np.where(valid, d, 0.0).sum(axis=0)

    outgoing_mean = np.divide(
        outgoing_mass,
        outgoing_count,
        out=np.full(n, scale, dtype=float),
        where=outgoing_count > 0.0,
    )
    incoming_mean = np.divide(
        incoming_mass,
        incoming_count,
        out=np.full(n, scale, dtype=float),
        where=incoming_count > 0.0,
    )

    endpoint_scale = np.sqrt(
        np.maximum(outgoing_mean[:, None], scale * 1e-12)
        * np.maximum(incoming_mean[None, :], scale * 1e-12)
    )

    distance_term = np.divide(
        endpoint_scale,
        endpoint_scale + np.where(valid, d, 0.0) + scale,
        out=np.zeros_like(d),
        where=valid,
    )

    # Best valid two-edge detour for every directed pair.
    best_detour = np.full((n, n), np.inf, dtype=float)

    for k in range(n):
        left_valid = valid[:, k]
        right_valid = valid[k, :]
        usable = left_valid[:, None] & right_valid[None, :]

        detour = safe[:, k, None] + safe[None, k, :]
        best_detour = np.minimum(
            best_detour,
            np.where(usable, detour, np.inf),
        )

    advantage = np.zeros_like(d)
    np.subtract(
        best_detour,
        d,
        out=advantage,
        where=valid & np.isfinite(best_detour),
    )
    advantage = np.maximum(advantage, 0.0)

    positive_advantage = advantage[valid & (advantage > 0.0)]
    advantage_scale = (
        float(np.median(positive_advantage))
        if positive_advantage.size
        else scale
    )
    advantage_scale = max(advantage_scale, np.finfo(float).eps)

    shortcut_term = np.divide(
        advantage,
        advantage + advantage_scale,
        out=np.zeros_like(d),
        where=valid,
    )

    # Favor edges whose endpoints have fewer cheap alternatives.
    row_pressure = np.divide(
        scale,
        scale + outgoing_mean,
        out=np.zeros(n, dtype=float),
    )
    col_pressure = np.divide(
        scale,
        scale + incoming_mean,
        out=np.zeros(n, dtype=float),
    )
    endpoint_term = np.sqrt(
        row_pressure[:, None] * col_pressure[None, :]
    )

    score = distance_term * (
        0.35 + 0.65 * shortcut_term
    ) * (
        0.55 + 0.45 * endpoint_term
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score
