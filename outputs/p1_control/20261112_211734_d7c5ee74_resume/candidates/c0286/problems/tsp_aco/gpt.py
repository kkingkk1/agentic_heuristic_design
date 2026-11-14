import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric local-neighborhood edge priorities for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = distances[np.isfinite(distances) & (distances > 0.0)]
    if finite_positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(finite_positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(finite_positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(
        np.isfinite(distances) & (distances > 0.0),
        distances,
        fallback,
    )
    d = 0.5 * (d + d.T)
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.sqrt(n))))

    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.maximum(nearest.mean(axis=1), eps)

    if n > 2:
        second_scale = np.partition(work, 1, axis=1)[:, 1]
    else:
        second_scale = nearest[:, 0]
    second_scale = np.maximum(second_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_distance = d / (pair_scale + eps)

    local_i = d / (local_scale[:, None] + eps)
    local_j = d / (local_scale[None, :] + eps)

    distance_signal = np.exp(
        -np.clip(relative_distance ** 0.9, 0.0, 50.0)
    )
    endpoint_signal = np.exp(
        -0.5 * np.clip(local_i + local_j, 0.0, 100.0)
    )

    competition = 1.0 / (1.0 + relative_distance)
    competition = np.sqrt(competition * competition.T)

    substitute_scale = np.sqrt(
        second_scale[:, None] * second_scale[None, :]
    )
    margin = substitute_scale / (d + substitute_scale + eps)
    margin = np.clip(margin, 0.0, 1.0)

    score = (
        distance_signal
        * endpoint_signal
        * (0.72 + 0.28 * competition)
        * (1.0 + 0.06 * margin)
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    values = score[off_diagonal]

    if values.size:
        cutoff = float(np.percentile(values, 85.0))
        score = np.where(
            off_diagonal & (score >= cutoff),
            score * 1.03,
            score,
        )

    score = np.nan_to_num(
        0.5 * (score + score.T),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
