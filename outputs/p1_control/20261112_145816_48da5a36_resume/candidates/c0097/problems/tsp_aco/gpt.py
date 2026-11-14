import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric diffusion-supported edge-priority matrix for TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0)
    d = 0.5 * (d + d.T)
    valid = np.isfinite(d) & (d > 0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_values = d[valid]
    eps = np.finfo(float).eps
    scale = max(float(np.median(positive_values)), eps)

    safe_d = np.where(valid, d, scale * 32.0)
    safe_d = np.maximum(safe_d, eps)

    # Robustly compress extreme distances before constructing affinities.
    transformed = np.log1p(safe_d / scale)
    affinity = np.exp(-transformed * transformed)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    row_mass = np.sum(affinity, axis=1, keepdims=True)
    transition = affinity / np.maximum(row_mass, eps)

    # Mutual one-step accessibility.
    mutual_transition = np.sqrt(
        np.maximum(transition, 0.0)
        * np.maximum(transition.T, 0.0)
    )

    # Support supplied by paths through an intermediate node.
    diffusion = transition @ transition
    diffusion = np.maximum(0.5 * (diffusion + diffusion.T), 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    diffusion_values = diffusion[off_diagonal]
    diffusion_scale = (
        float(np.quantile(diffusion_values, 0.75))
        if diffusion_values.size
        else 0.0
    )
    diffusion_scale = max(diffusion_scale, eps)
    diffusion_score = np.clip(diffusion / diffusion_scale, 0.0, 1.0)

    direct_score = np.exp(-safe_d / scale)
    direct_score[~valid] = 0.0

    if n > 2:
        score = (
            np.power(np.maximum(mutual_transition, 0.0), 0.55)
            * np.power(diffusion_score, 0.75)
            * np.power(direct_score, 0.20)
        )
    else:
        score = direct_score

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
