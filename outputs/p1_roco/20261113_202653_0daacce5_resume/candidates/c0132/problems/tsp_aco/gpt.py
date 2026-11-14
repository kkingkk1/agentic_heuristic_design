import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust, structurally informed edge-selection priors for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    values = distances[valid]
    positive = values[values > 0.0]

    if positive.size:
        median = float(np.median(positive))
        mad = float(np.median(np.abs(positive - median)))
        scale = max(median, 1.4826 * mad, np.finfo(float).eps)
    else:
        scale = 1.0

    d = np.where(valid, np.maximum(distances, 0.0), np.inf)
    normalized = np.minimum(d / scale, 40.0)

    # Smooth distance preference. The reciprocal term protects very short edges.
    thermal = np.exp(-normalized)
    reciprocal = 1.0 / (1.0 + normalized)
    attraction = 0.65 * thermal + 0.35 * reciprocal
    attraction[~valid] = 0.0

    # Bidirectional rank competition: favor edges that are locally competitive
    # at both their source and destination.
    row_counts = np.sum(valid, axis=1, keepdims=True)
    col_counts = np.sum(valid, axis=0, keepdims=True)

    row_best = np.min(d, axis=1, keepdims=True)
    col_best = np.min(d, axis=0, keepdims=True)

    row_gap = np.maximum(d - row_best, 0.0) / (scale + np.finfo(float).eps)
    col_gap = np.maximum(d - col_best, 0.0) / (scale + np.finfo(float).eps)

    competition = 1.0 / (
        1.0
        + 0.45 * np.minimum(row_gap, 8.0)
        + 0.45 * np.minimum(col_gap, 8.0)
    )
    competition[~valid] = 0.0

    # Mild endpoint balancing prevents one node from dominating the sample
    # without overriding strong distance evidence.
    local_strength = np.sum(attraction, axis=1, keepdims=True)
    local_strength = np.maximum(local_strength, np.finfo(float).eps)
    row_balance = 1.0 / np.sqrt(local_strength)
    col_strength = np.sum(attraction, axis=0, keepdims=True)
    col_strength = np.maximum(col_strength, np.finfo(float).eps)
    col_balance = 1.0 / np.sqrt(col_strength)

    balance = row_balance * col_balance
    positive_balance = balance[valid]
    reference = float(np.median(positive_balance)) if positive_balance.size else 1.0
    balance = np.clip(balance / max(reference, np.finfo(float).eps), 0.75, 1.35)

    # Small regret bonus preserves edges that are substantially better than
    # the second-best available option at either endpoint.
    if n >= 3:
        second_out = np.partition(d, 1, axis=1)[:, 1][:, None]
        second_in = np.partition(d, 1, axis=0)[1, :][None, :]
        regret_out = np.maximum(second_out - d, 0.0) / scale
        regret_in = np.maximum(second_in - d, 0.0) / scale
        regret = 1.0 + 0.12 * np.tanh(
            np.minimum(regret_out, 8.0) + np.minimum(regret_in, 8.0)
        )
    else:
        regret = np.ones((n, n), dtype=float)

    priors = attraction * competition * balance * regret
    priors[~valid] = 0.0
    np.fill_diagonal(priors, 0.0)

    priors = np.nan_to_num(priors, nan=0.0, posinf=0.0, neginf=0.0)
    maximum = float(np.max(priors))
    if maximum > 0.0:
        priors /= maximum

    return priors
