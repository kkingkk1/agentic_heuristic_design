import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite_positive = np.isfinite(d) & (d > 0)
    symmetric_valid = finite_positive & finite_positive.T

    if not np.any(symmetric_valid):
        return result

    effective = np.where(
        symmetric_valid,
        0.5 * (d + d.T),
        np.inf,
    )
    np.fill_diagonal(effective, np.inf)

    valid_edges = effective[np.isfinite(effective)]
    if valid_edges.size == 0:
        return result

    scale = float(np.median(valid_edges))
    scale = max(scale, np.finfo(float).eps)

    detour_quality = np.zeros((n, n), dtype=float)

    for k in range(n):
        detour = effective[:, k, None] + effective[None, k, :]
        usable = np.isfinite(detour) & np.isfinite(effective)
        ratio = np.divide(
            detour,
            effective,
            out=np.zeros_like(effective),
            where=usable & (effective > 0),
        )
        detour_quality = np.maximum(detour_quality, np.where(usable, ratio, 0.0))

    direct_rank = np.zeros((n, n), dtype=float)
    edge_values = effective[np.isfinite(effective)]

    for i in range(n):
        row = effective[i]
        valid = np.isfinite(row)
        if np.any(valid):
            direct_rank[i, valid] = np.searchsorted(
                np.sort(edge_values),
                row[valid],
                side="right",
            ) / max(edge_values.size, 1)

    detour_strength = np.clip(detour_quality - 1.0, 0.0, None)
    detour_strength = detour_strength / (
        np.max(detour_strength) + np.finfo(float).eps
    )

    competitive = 1.0 - direct_rank
    competitive = np.clip(competitive, 0.0, 1.0)

    score = (
        0.78 * detour_strength
        + 0.22 * competitive
    )

    score = 0.5 * (score + score.T)
    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
