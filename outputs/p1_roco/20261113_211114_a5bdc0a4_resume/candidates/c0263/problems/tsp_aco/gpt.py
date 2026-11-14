import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return random-walk structural priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    offdiag = ~np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    valid = offdiag & (finite | finite.T)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.zeros((n, n), dtype=float)
    both = finite & finite.T
    forward_only = finite & ~finite.T
    backward_only = ~finite & finite.T

    cleaned[both] = 0.5 * (raw[both] + raw.T[both])
    cleaned[forward_only] = raw[forward_only]
    cleaned[backward_only] = raw.T[backward_only]
    cleaned = np.maximum(cleaned, 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)

    observed = cleaned[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps

    # Reciprocal edge conductance with a robust floor for zero distances.
    conductance = np.zeros((n, n), dtype=float)
    conductance[valid] = 1.0 / (
        np.maximum(cleaned[valid], 0.03 * scale) + 0.08 * scale
    )
    conductance = np.minimum(conductance, 1.0 / eps)
    conductance[~valid] = 0.0
    np.fill_diagonal(conductance, 0.0)

    degree = conductance.sum(axis=1)
    safe_degree = np.maximum(degree, eps)
    transition = conductance / safe_degree[:, None]

    # Direct preference corrected by endpoint accessibility.
    direct = np.sqrt(
        np.maximum(transition, 0.0) * np.maximum(transition.T, 0.0)
    )

    # Common-neighbor support from a two-step random walk.
    two_step = transition @ transition
    walk_support = np.sqrt(
        np.maximum(two_step, 0.0) * np.maximum(two_step.T, 0.0)
    )
    support_max = np.max(walk_support)
    if np.isfinite(support_max) and support_max > 0.0:
        walk_support /= support_max

    # Penalize edges attached to highly diffuse neighborhoods.
    probabilities = transition.copy()
    probabilities[probabilities <= 0.0] = 1.0
    entropy = -np.sum(transition * np.log(probabilities), axis=1)
    max_entropy = np.log(max(float(n - 1), 1.0))
    concentration = 1.0 - entropy / max(max_entropy, eps)
    concentration = np.clip(concentration, 0.0, 1.0)
    endpoint_concentration = np.sqrt(
        concentration[:, None] * concentration[None, :]
    )

    # A local contrast term: edge conductance relative to the geometric mean
    # of the incident node's available conductance.
    geometric_degree = np.sqrt(safe_degree[:, None] * safe_degree[None, :])
    contrast = conductance / np.maximum(geometric_degree, eps)
    contrast[~valid] = 0.0
    contrast_max = float(np.max(contrast))
    if np.isfinite(contrast_max) and contrast_max > 0.0:
        contrast /= contrast_max

    # Log compression avoids letting a few zero/near-zero edges dominate.
    score = (
        0.34 * np.sqrt(np.maximum(direct, 0.0))
        + 0.30 * np.sqrt(np.maximum(walk_support, 0.0))
        + 0.20 * np.sqrt(np.maximum(contrast, 0.0))
        + 0.16 * endpoint_concentration
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.log1p(5.0 * np.maximum(score, 0.0))
    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
