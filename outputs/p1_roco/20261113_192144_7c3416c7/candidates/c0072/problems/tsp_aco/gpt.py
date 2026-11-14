import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust edge-selection priors for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    valid = np.isfinite(distances)
    positive = distances[valid & (distances > 0)]

    if positive.size:
        center = float(np.median(positive))
        spread = float(np.median(np.abs(positive - center)))
        replacement = center + max(spread, np.finfo(float).eps)
    else:
        replacement = 1.0

    safe = np.where(valid, np.maximum(distances, 0.0), replacement)
    np.fill_diagonal(safe, np.inf)

    finite_edges = np.isfinite(safe)
    if not np.any(finite_edges):
        return np.zeros((n, n), dtype=float)

    finite_values = safe[finite_edges]
    fallback = float(np.max(finite_values)) if finite_values.size else replacement
    safe = np.where(np.isfinite(safe), safe, fallback)

    positive_scale = max(float(np.median(finite_values)), np.finfo(float).eps)
    normalized = safe / positive_scale

    inverse = 1.0 / (normalized + 1e-12)
    inverse[~finite_edges] = 0.0

    row_order = np.argsort(np.where(finite_edges, safe, np.inf), axis=1)
    col_order = np.argsort(np.where(finite_edges, safe, np.inf), axis=0)

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    for i in range(n):
        row_rank[i, row_order[i]] = np.arange(n, dtype=float)
        col_rank[col_order[:, i], i] = np.arange(n, dtype=float)

    denominator = max(n - 1, 1)
    row_rank_score = np.exp(-row_rank / denominator)
    col_rank_score = np.exp(-col_rank / denominator)

    row_best = np.max(inverse, axis=1, keepdims=True)
    col_best = np.max(inverse, axis=0, keepdims=True)

    row_competitiveness = inverse / (row_best + 1e-12)
    col_competitiveness = inverse / (col_best + 1e-12)

    reciprocal = 0.5 * (
        (row_rank <= 1).astype(float) +
        (col_rank <= 1).astype(float)
    )

    priors = (
        0.45 * inverse +
        0.20 * row_rank_score +
        0.20 * col_rank_score +
        0.10 * row_competitiveness +
        0.05 * col_competitiveness
    )

    priors *= 1.0 + 0.25 * reciprocal

    candidate_count = min(max(4, int(np.ceil(np.sqrt(max(n, 1))))), max(n - 1, 1))
    candidate_mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        candidates = row_order[i, :candidate_count]
        candidate_mask[i, candidates] = True

    priors = np.where(candidate_mask, priors, 0.02 * priors)
    priors[~finite_edges] = 0.0
    np.fill_diagonal(priors, 0.0)

    maximum = float(np.max(priors))
    if maximum > 0.0 and np.isfinite(maximum):
        priors /= maximum

    return np.nan_to_num(priors, nan=0.0, posinf=0.0, neginf=0.0)
