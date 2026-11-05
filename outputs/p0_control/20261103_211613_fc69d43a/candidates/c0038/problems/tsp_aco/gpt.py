import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric mutual-rank edge-score matrix for a TSP distance matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    finite = np.isfinite(distances)
    if np.any(distances[finite] < 0):
        raise ValueError("distance_matrix must contain non-negative distances")

    valid = finite & ~np.eye(n, dtype=bool)
    counts = valid.sum(axis=1)

    if not np.any(valid):
        return scores

    work = np.where(valid, distances, np.inf)

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.zeros((n, n), dtype=float)

    row_ids = np.arange(n)[:, None]
    rank_values = np.arange(1, n + 1, dtype=float)[None, :]
    ranks[row_ids, order] = rank_values
    ranks[~valid] = np.inf

    denominator = np.maximum(counts[:, None] - 1, 1)
    normalized_ranks = (ranks - 1.0) / denominator
    normalized_ranks[~valid] = np.inf

    forward = normalized_ranks
    backward = normalized_ranks.T

    mutual_rank = 0.5 * (forward + backward)
    rank_disagreement = np.abs(forward - backward)

    finite_counts = counts[valid.any(axis=1)]
    typical_count = float(np.median(finite_counts)) if finite_counts.size else 1.0

    endpoint_balance = np.sqrt(
        np.minimum(counts[:, None], counts[None, :])
        / np.maximum(np.maximum(counts[:, None], counts[None, :]), 1)
    )
    endpoint_balance *= 0.75 + 0.25 * np.minimum(
        1.0, np.sqrt(np.maximum(counts[:, None], 1) / max(typical_count, 1.0))
    )
    endpoint_balance *= 0.75 + 0.25 * np.minimum(
        1.0, np.sqrt(np.maximum(counts[None, :], 1) / max(typical_count, 1.0))
    )

    scores = np.exp(-3.0 * mutual_rank - 2.0 * rank_disagreement)
    scores *= endpoint_balance
    scores[~valid] = 0.0

    scores = 0.5 * (scores + scores.T)
    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
