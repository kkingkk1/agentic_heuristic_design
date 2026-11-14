import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0.0)
    valid = finite_positive & finite_positive.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    symmetric = np.zeros_like(d, dtype=float)
    symmetric[valid] = 0.5 * (d[valid] + d.T[valid])

    positive_values = symmetric[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    maximum_distance = max(float(np.max(positive_values)), global_scale)

    work = np.where(valid, symmetric, maximum_distance * 4.0)
    np.fill_diagonal(work, maximum_distance * 4.0)

    # Global affinity graph used for two-step neighborhood diffusion.
    scaled = np.minimum(work / global_scale, 60.0)
    affinity = np.exp(-scaled)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    row_mass = affinity.sum(axis=1)
    row_mass = np.maximum(row_mass, eps)
    transition = affinity / row_mass[:, None]

    shared_neighborhood = transition @ transition.T
    shared_neighborhood = np.maximum(shared_neighborhood, eps)

    # Each node chooses a nearest valid anchor.
    nearest_index = np.argmin(work, axis=1)
    nearest_distance = work[np.arange(n), nearest_index]

    anchor_budget = nearest_distance[:, None] + nearest_distance[None, :]
    savings = anchor_budget - work
    savings = np.maximum(savings, 0.0)
    savings = np.minimum(savings / global_scale, 40.0)

    # Mutual-anchor edges indicate direct compatibility of two endpoint basins.
    mutual_anchor = (
        (nearest_index[:, None] == np.arange(n)[None, :])
        & (nearest_index[None, :] == np.arange(n)[:, None])
        & valid
    )

    # Penalize edges that are globally expensive, while retaining structural
    # information from the diffusion graph.
    normalized_distance = np.minimum(work / global_scale, 60.0)

    # Nodes with little affinity mass are structurally isolated; edges incident
    # to them receive a mild compensation instead of being removed outright.
    accessibility = np.sqrt(row_mass[:, None] * row_mass[None, :])
    accessibility = accessibility / max(float(np.max(accessibility)), eps)
    accessibility = np.maximum(accessibility, eps)

    log_score = (
        1.35 * np.log(shared_neighborhood)
        + 0.80 * np.log1p(savings)
        + 0.55 * np.log(accessibility)
        - 0.48 * normalized_distance
        + 0.22 * mutual_anchor.astype(float)
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(
        np.clip(log_score[finite_scores] - maximum, -745.0, 0.0)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
