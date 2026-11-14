import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    valid = finite & finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = 0.5 * (distances + distances.T)
    symmetric = np.maximum(symmetric, 0.0)

    valid_values = symmetric[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.sqrt(np.finfo(float).eps) * max(scale, 1.0)

    safe = np.where(valid, symmetric, np.inf)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (safe[valid] + eps)

    inverse_max = float(np.max(inverse))
    if inverse_max > 0.0 and np.isfinite(inverse_max):
        inverse_scaled = inverse / inverse_max
    else:
        inverse_scaled = np.zeros_like(inverse)

    temperature = max(0.75 * scale, eps)
    soft_affinity = np.zeros_like(inverse)
    soft_affinity[valid] = np.exp(
        -np.minimum(safe[valid], 50.0 * scale) / temperature
    )

    row_mass = soft_affinity.sum(axis=1)
    safe_mass = np.maximum(row_mass, eps)

    row_competitiveness = soft_affinity / safe_mass[:, None]
    col_competitiveness = row_competitiveness.T
    mutual_support = np.sqrt(
        np.maximum(row_competitiveness * col_competitiveness, 0.0)
    )

    row_max = np.maximum(inverse_scaled.max(axis=1), eps)
    col_max = np.maximum(inverse_scaled.max(axis=0), eps)

    endpoint_competitiveness = np.sqrt(
        (inverse_scaled / row_max[:, None])
        * (inverse_scaled / col_max[None, :])
    )

    neighborhood_size = min(3, n - 1)
    inverse_candidates = np.where(valid, inverse_scaled, 0.0)

    row_top = np.partition(
        inverse_candidates,
        n - neighborhood_size,
        axis=1,
    )[:, n - neighborhood_size:]

    col_top = np.partition(
        inverse_candidates,
        n - neighborhood_size,
        axis=0,
    )[n - neighborhood_size:, :]

    row_potential = row_top.mean(axis=1)
    col_potential = col_top.mean(axis=0)

    node_potential = np.sqrt(
        np.maximum(row_potential, eps)
        * np.maximum(col_potential, eps)
    )
    node_max = float(np.max(node_potential))
    if node_max > 0.0 and np.isfinite(node_max):
        node_potential /= node_max

    endpoint_potential = np.sqrt(
        np.maximum(node_potential[:, None] * node_potential[None, :], eps)
    )

    score = (
        np.maximum(inverse_scaled, eps) ** 0.58
        * np.maximum(endpoint_competitiveness, eps) ** 0.18
        * np.maximum(mutual_support, eps) ** 0.14
        * np.maximum(endpoint_potential, eps) ** 0.10
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    valid_count = float(np.sum(valid))
    exploration = 0.06
    if valid_count > 0.0:
        score = (
            (1.0 - exploration) * score
            + exploration * valid.astype(float) / valid_count
        )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
