import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
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

    symmetric = np.zeros_like(d, dtype=float)
    symmetric[valid] = 0.5 * (d[valid] + d.T[valid])

    values = symmetric[valid]
    eps = np.finfo(float).eps

    q25, q50, q75 = np.percentile(values, [25.0, 50.0, 75.0])
    spread = max(float(q75 - q25), eps)
    scale = max(float(q50), eps)

    fill_value = max(
        float(np.max(values)) + 4.0 * spread,
        scale + spread,
    )

    work = np.where(valid, symmetric, fill_value)
    np.fill_diagonal(work, fill_value)

    # Global radial-basis similarity; the bandwidth is robust to outliers.
    bandwidth = max(scale + 0.75 * spread, eps)
    centered = work / bandwidth
    with np.errstate(over="ignore", under="ignore"):
        direct = np.exp(-0.5 * np.minimum(centered, 60.0) ** 2)

    direct[~valid] = 0.0
    np.fill_diagonal(direct, 0.0)

    # Normalize outgoing affinities without using hard candidate sets.
    row_mass = np.sum(direct, axis=1)
    transition = direct / np.maximum(row_mass[:, None], eps)

    # Two-hop support: edges supported by multiple short indirect routes
    # receive additional prior probability.
    closure = transition @ transition
    closure = 0.5 * (closure + closure.T)
    closure[~valid] = 0.0
    np.fill_diagonal(closure, 0.0)

    closure_max = float(np.max(closure))
    if closure_max > 0.0:
        closure = closure / closure_max

    # Endpoint replacement regret. An edge is valuable when removing it
    # exposes a relatively expensive alternative at either endpoint.
    ordered = np.sort(np.where(valid, symmetric, np.inf), axis=1)
    first = ordered[:, 0]
    second = ordered[:, 1] if n > 2 else ordered[:, 0]

    first = np.where(np.isfinite(first), first, scale)
    second = np.where(np.isfinite(second), second, first)

    regret = np.maximum(second - first, 0.0)
    regret_scale = max(float(np.median(regret[regret > 0.0]))
                       if np.any(regret > 0.0) else spread, eps)

    endpoint_regret = (
        regret[:, None] + regret[None, :]
    ) / regret_scale
    endpoint_regret = np.minimum(endpoint_regret, 20.0)

    # Penalize edges that are long relative to the global robust distance
    # while retaining a nonzero probability for structurally necessary edges.
    length_term = np.exp(
        -np.minimum((work / max(scale + spread, eps)) ** 1.35, 50.0)
    )

    # Combine independent structural signals in ordinary positive space.
    score = (
        (0.55 + 0.45 * direct)
        * (0.70 + 0.30 * closure)
        * (1.0 + 0.28 * np.tanh(endpoint_regret))
        * length_term
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
