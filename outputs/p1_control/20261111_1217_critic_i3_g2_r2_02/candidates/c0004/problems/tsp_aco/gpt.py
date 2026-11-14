import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    off_diagonal = ~np.eye(n, dtype=bool)

    if not np.any(finite & off_diagonal):
        return np.zeros((n, n), dtype=float)

    if np.any((~finite) & off_diagonal):
        finite_values = distances[finite & off_diagonal]
        fallback = float(np.max(finite_values)) if finite_values.size else 1.0
        work = np.where(finite, distances, fallback)
    else:
        work = distances.copy()

    work = np.maximum(work, 0.0)
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    # A broad temperature preserves globally useful alternatives.
    temperature = scale * (0.8 + 0.2 * np.log1p(n))

    affinity = np.zeros((n, n), dtype=float)
    valid = np.isfinite(work)
    affinity[valid] = np.exp(-work[valid] / temperature)
    np.fill_diagonal(affinity, 0.0)

    # Entropy-regularized balancing suppresses edges connected to hubs.
    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    for _ in range(80):
        row_mass = affinity @ col_scale
        row_mass = np.maximum(row_mass, np.finfo(float).eps)
        row_scale = 1.0 / row_mass

        col_mass = affinity.T @ row_scale
        col_mass = np.maximum(col_mass, np.finfo(float).eps)
        col_scale = 1.0 / col_mass

    balanced = row_scale[:, None] * affinity * col_scale[None, :]
    balanced = (balanced + balanced.T) / 2.0

    # Compare global connection profiles rather than only direct distances.
    norms = np.linalg.norm(balanced, axis=1)
    profile_similarity = balanced @ balanced.T
    profile_similarity /= np.maximum(
        norms[:, None] * norms[None, :],
        np.finfo(float).eps,
    )
    profile_similarity = np.clip(profile_similarity, 0.0, 1.0)

    score = balanced * (0.7 + 0.3 * profile_similarity)
    score[~(finite | np.eye(n, dtype=bool))] = 0.0
    np.fill_diagonal(score, 0.0)

    score = (score + score.T) / 2.0
    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
