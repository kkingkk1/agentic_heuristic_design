import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return scale-free savings indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    eps = np.finfo(float).eps
    finite = np.isfinite(distances)
    positive = finite & (distances > 0.0)
    positive &= ~np.eye(n, dtype=bool)

    if not np.any(positive):
        return np.zeros((n, n), dtype=float)

    positive_values = distances[positive]
    fallback = float(np.max(positive_values))

    d = np.where(finite, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)

    valid = np.isfinite(d) & (d > 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(valid, d, fallback)
    np.fill_diagonal(safe, np.inf)

    ordered = np.sort(safe, axis=1)
    finite_counts = np.sum(np.isfinite(safe), axis=1)
    finite_counts = np.maximum(finite_counts, 1)

    median_index = np.maximum((finite_counts - 1) // 2, 0)
    reference = ordered[
        np.arange(n),
        np.minimum(median_index, n - 1),
    ]
    reference = np.maximum(reference, eps)

    lower_index = np.maximum((finite_counts - 1) // 4, 0)
    lower_reference = ordered[
        np.arange(n),
        np.minimum(lower_index, n - 1),
    ]
    lower_reference = np.maximum(lower_reference, eps)

    endpoint_budget = reference[:, None] + reference[None, :]

    savings = (endpoint_budget - d) / (endpoint_budget + d + eps)
    savings = np.clip(savings, -1.0, 1.0)

    positive_savings = np.maximum(savings, 0.0)
    savings_score = positive_savings ** 1.5

    relative_cost = d / (
        lower_reference[:, None]
        + lower_reference[None, :]
        + eps
    )
    reciprocal_score = 1.0 / (1.0 + relative_cost ** 1.25)

    endpoint_tightness = np.sqrt(
        (lower_reference / (reference + eps))[:, None]
        * (lower_reference / (reference + eps))[None, :]
    )
    endpoint_tightness = np.clip(endpoint_tightness, 0.0, 1.0)

    row_mean = np.mean(
        np.where(np.isfinite(safe), safe, reference[:, None]),
        axis=1,
    )
    centrality = np.sqrt(
        (reference / (row_mean + eps))[:, None]
        * (reference / (row_mean + eps))[None, :]
    )
    centrality = np.clip(centrality, 0.25, 4.0)

    score = (
        0.55 * savings_score
        + 0.30 * reciprocal_score
        + 0.15 * endpoint_tightness
    )
    score *= np.sqrt(centrality)
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
