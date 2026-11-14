import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a globally degree-balanced stochastic prior for TSP edges."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must contain nonnegative values")

    # Use an undirected distance model for the global balancing step.
    d = 0.5 * (distances + distances.T)
    np.fill_diagonal(d, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    positive = d[off_diagonal & (d > 0)]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    scale = max(scale, np.finfo(float).eps)

    # Temperature adapts to the overall distance dispersion.
    spread = float(np.std(d[off_diagonal]))
    temperature = max(scale * 0.35, spread * 0.75, np.finfo(float).eps)

    shifted = d - np.min(d[off_diagonal])
    kernel = np.exp(-np.clip(shifted / temperature, 0.0, 60.0))
    kernel *= off_diagonal

    # Symmetric Sinkhorn scaling: solve diag(s) K diag(s) 1 = 1.
    scaling = np.ones(n, dtype=float)

    for _ in range(2000):
        updated = 1.0 / np.maximum(kernel @ scaling, np.finfo(float).eps)
        updated = np.sqrt(np.maximum(updated * scaling, 0.0))

        if np.max(np.abs(updated - scaling)) <= 1e-10:
            scaling = updated
            break

        scaling = updated

    score = scaling[:, None] * kernel * scaling[None, :]
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    row_sum = score.sum(axis=1, keepdims=True)
    score = np.divide(
        score,
        row_sum,
        out=np.zeros_like(score),
        where=row_sum > 0,
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
