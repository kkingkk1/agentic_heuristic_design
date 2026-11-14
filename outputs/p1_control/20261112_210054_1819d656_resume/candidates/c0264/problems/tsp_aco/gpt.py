import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return detour-advantage edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    positive_values = distances[valid & (distances > 0.0)]
    if positive_values.size == 0:
        return np.zeros((n, n), dtype=float)

    reference = float(np.median(positive_values))
    spread = float(np.percentile(positive_values, 75.0))
    lower = float(np.percentile(positive_values, 25.0))

    if not np.isfinite(reference) or reference <= 0.0:
        reference = float(np.mean(positive_values))
    if not np.isfinite(reference) or reference <= 0.0:
        reference = 1.0

    scale = max(spread - lower, reference * 0.1, eps)

    fallback = float(np.percentile(positive_values, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = reference

    d = np.where(valid, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)

    symmetric_valid = valid & valid.T
    np.fill_diagonal(d, np.inf)

    if n > 2:
        through = d[:, :, None] + d[None, :, :]
        np.fill_diagonal(through.reshape(n * n, n), np.inf)
        for intermediary in range(n):
            through[intermediary, :, intermediary] = np.inf
            through[:, intermediary, intermediary] = np.inf

        best_detour = np.min(through, axis=2)
        has_detour = np.isfinite(best_detour)
    else:
        best_detour = np.full((n, n), fallback, dtype=float)
        has_detour = np.zeros((n, n), dtype=bool)

    direct = np.where(np.isfinite(d), d, fallback)

    detour_gap = best_detour - direct
    detour_signal = np.tanh(detour_gap / (2.0 * scale))

    length_signal = np.tanh(
        (reference - direct) / (2.0 * scale)
    )

    absolute_scale = np.maximum(
        best_detour + direct,
        eps,
    )
    relative_gap = detour_gap / absolute_scale
    relative_signal = np.tanh(3.0 * relative_gap)

    score = (
        0.55 * detour_signal
        + 0.30 * relative_signal
        + 0.15 * length_signal
    )

    score = 0.5 * (score + score.T)
    score = 0.5 + 0.5 * score

    score[~symmetric_valid] = 0.0
    if n > 2:
        score[~has_detour] = 0.0

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    minimum = float(np.min(score[symmetric_valid])) if np.any(symmetric_valid) else 0.0

    if maximum > minimum and np.isfinite(maximum) and np.isfinite(minimum):
        score[symmetric_valid] = (
            score[symmetric_valid] - minimum
        ) / (maximum - minimum)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
