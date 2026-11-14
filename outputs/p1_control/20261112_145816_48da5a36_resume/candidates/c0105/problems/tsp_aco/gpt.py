import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, normalized edge-priority matrix for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    distances = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
    distances = 0.5 * (distances + distances.T)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(distances) & (distances > 0.0) & ~diagonal
    valid &= valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_values = distances[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    replacement = max(float(np.max(positive_values)), global_scale) * 10.0

    work = np.where(valid, distances, replacement)
    work = np.maximum(work, eps)
    np.fill_diagonal(work, replacement)

    # Use the third-nearest valid neighbor when available.
    sorted_work = np.sort(np.where(valid, distances, np.inf), axis=1)
    kth = min(2, n - 2)
    local_scale = sorted_work[:, kth]

    fallback = global_scale
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        fallback,
    )
    local_scale = np.maximum(local_scale, eps)

    normalized_length = work / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    # Reciprocal endpoint ranks favor edges preferred by both endpoints.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    ranks /= max(n - 1, 1)

    rank_sum = ranks + ranks.T
    mutual_rank = 1.0 / (1.0 + rank_sum)

    # Penalize joining endpoints with strongly different local scales.
    scale_difference = np.abs(
        np.log(local_scale[:, None]) - np.log(local_scale[None, :])
    )
    scale_compatibility = np.exp(-0.25 * scale_difference)

    # Softly reduce concentration around unusually dense endpoints.
    local_density = np.sum(
        np.exp(-work / local_scale[:, None]) * valid,
        axis=1,
    )
    density_factor = np.sqrt(
        (1.0 + local_density[:, None])
        * (1.0 + local_density[None, :])
    )
    crowding_penalty = 1.0 / density_factor

    # Combine all signals in the log domain for numerical stability.
    log_score = (
        -1.0 * normalized_length
        -0.35 * (work / global_scale)
        + 1.6 * np.log(np.maximum(mutual_rank, eps))
        + np.log(np.maximum(scale_compatibility, eps))
        + 0.35 * np.log(np.maximum(crowding_penalty, eps))
    )

    finite_scores = valid & np.isfinite(log_score)
    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(
        np.clip(log_score[finite_scores] - maximum, -745.0, 0.0)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
