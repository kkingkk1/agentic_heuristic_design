import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust probability-consensus edge priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite_nonnegative, False)

    if not np.any(finite_nonnegative):
        return result

    reverse_valid = finite_nonnegative.T
    both_valid = finite_nonnegative & reverse_valid
    forward_only = finite_nonnegative & ~reverse_valid
    backward_only = reverse_valid & ~finite_nonnegative

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[both_valid] = 0.5 * (
        distances[both_valid] + distances.T[both_valid]
    )
    symmetric[forward_only] = distances[forward_only]
    symmetric[backward_only] = distances.T[backward_only]
    np.fill_diagonal(symmetric, np.inf)

    valid = np.isfinite(symmetric)
    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    probabilities = np.zeros((n, n), dtype=float)
    concentration = np.zeros(n, dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        values = symmetric[i, neighbors]
        center = float(np.median(values))
        deviations = np.abs(values - center)
        mad = float(np.median(deviations))
        mean_deviation = float(np.mean(deviations))

        scale = 0.5 * (mad + mean_deviation)
        if not np.isfinite(scale) or scale <= eps:
            positive = values[values > 0.0]
            if positive.size:
                scale = float(np.median(positive))
            else:
                scale = 1.0

        scale = max(scale, eps)

        standardized = (values - center) / scale
        standardized = np.clip(standardized, -40.0, 40.0)

        logits = -standardized
        logits -= float(np.max(logits))

        weights = np.exp(logits)
        total = float(np.sum(weights))

        if not np.isfinite(total) or total <= 0.0:
            weights = np.full(
                neighbors.size,
                1.0 / float(neighbors.size),
                dtype=float,
            )
        else:
            weights /= total

        probabilities[i, neighbors] = weights
        concentration[i] = float(np.sum(weights * weights))

    consensus = np.sqrt(probabilities * probabilities.T)

    concentration_pair = np.sqrt(
        concentration[:, None] * concentration[None, :]
    )
    concentration_pair = np.clip(concentration_pair, 0.0, 1.0)

    scores = consensus * (
        0.78 + 0.44 * concentration_pair
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
