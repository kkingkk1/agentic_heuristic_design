import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")

    if np.any(distances < 0):
        raise ValueError("distance_matrix must contain non-negative distances")

    result = np.zeros((n, n), dtype=float)
    if n == 1:
        return result

    work = distances.copy()
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps

    row_sorted = np.sort(work, axis=1)
    k = min(max(2, int(np.sqrt(n))), n - 1)
    local_scale = row_sorted[:, k - 1]
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        fallback,
    )

    normalized = work / np.maximum(local_scale[:, None], eps)
    distance_score = 1.0 / (1.0 + normalized)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = np.exp(-ranks / max(1.0, 0.35 * k))

    finite_values = work[np.isfinite(work)]
    global_scale = np.percentile(finite_values, 35) if finite_values.size else fallback
    global_scale = max(float(global_scale), eps)
    global_score = np.exp(-work / global_scale)

    local_score = (
        0.50 * distance_score
        + 0.35 * rank_score
        + 0.15 * global_score
    )
    local_score[~np.isfinite(local_score)] = 0.0
    np.fill_diagonal(local_score, 0.0)

    reverse_score = local_score.T
    mutual_score = np.sqrt(np.maximum(local_score, 0.0) *
                           np.maximum(reverse_score, 0.0))
    agreement = 1.0 - np.abs(local_score - reverse_score)
    agreement = np.clip(agreement, 0.0, 1.0)

    result = 0.65 * mutual_score + 0.25 * np.minimum(local_score, reverse_score)
    result += 0.10 * agreement * mutual_score
    result[~np.isfinite(result)] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(result.max())
    if maximum > 0.0:
        result /= maximum

    return result
