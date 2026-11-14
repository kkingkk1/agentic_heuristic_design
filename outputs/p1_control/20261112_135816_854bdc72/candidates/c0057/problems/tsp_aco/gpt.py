import warnings

import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid, False)

    symmetric_valid = valid & valid.T
    symmetric_distances = 0.5 * (distances + distances.T)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        local_scale = np.nanmedian(
            np.where(symmetric_valid, symmetric_distances, np.nan),
            axis=1,
        )

    finite_scales = local_scale[np.isfinite(local_scale) & (local_scale > 0)]
    fallback = float(np.median(finite_scales)) if finite_scales.size else 1.0
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        fallback,
    )

    safe_distances = np.where(symmetric_valid, symmetric_distances, np.inf)

    endpoint_score = np.sqrt(
        np.divide(
            local_scale[:, None] * local_scale[None, :],
            safe_distances * safe_distances,
            out=np.zeros_like(safe_distances),
            where=symmetric_valid,
        )
    )

    positive_distances = symmetric_distances[symmetric_valid]
    global_scale = (
        float(np.median(positive_distances))
        if positive_distances.size
        else fallback
    )

    global_score = np.divide(
        global_scale,
        safe_distances,
        out=np.zeros_like(safe_distances),
        where=symmetric_valid,
    )

    score = 0.8 * endpoint_score + 0.2 * global_score
    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
