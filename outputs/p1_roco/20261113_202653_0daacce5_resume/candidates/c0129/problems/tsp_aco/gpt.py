import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diversified edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite = np.isfinite(distances)
    valid = finite.copy()
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    clean = np.where(finite, np.maximum(distances, 0.0), np.inf)
    np.fill_diagonal(clean, np.inf)

    positive = clean[np.isfinite(clean) & (clean > 0.0)]
    if positive.size:
        center = float(np.median(positive))
        mad = float(np.median(np.abs(positive - center)))
        scale = max(center, 1.4826 * mad, 1.0)
    else:
        scale = 1.0

    eps = np.sqrt(np.finfo(float).eps) * scale

    finite_values = clean[valid]
    finite_values = np.asarray(finite_values, dtype=float)

    # Temperature-calibrated distance desirability.
    temperature = max(scale, eps)
    desirability = np.zeros((n, n), dtype=float)
    desirability[valid] = np.exp(
        -np.minimum(finite_values / temperature, 50.0)
    )

    # Row and column regret: reward edges that are distinctly better
    # than the corresponding node's alternatives.
    row_sorted = np.sort(np.where(valid, clean, np.inf), axis=1)
    col_sorted = np.sort(np.where(valid, clean, np.inf), axis=0)

    row_best = row_sorted[:, 0][:, None]
    row_second = row_sorted[:, 1][:, None] if n > 1 else row_best
    col_best = col_sorted[0, :][None, :]
    col_second = col_sorted[1, :][None, :] if n > 1 else col_best

    row_regret = np.maximum(row_second - row_best, 0.0)
    col_regret = np.maximum(col_second - col_best, 0.0)

    row_regret = np.clip(row_regret / (scale + eps), 0.0, 20.0)
    col_regret = np.clip(col_regret / (scale + eps), 0.0, 20.0)

    regret_bonus = 1.0 + 0.20 * np.tanh(row_regret) + 0.20 * np.tanh(col_regret)

    # Bypass savings: prefer edges that are cheaper than a two-hop detour.
    left = clean[:, :, None]
    right = clean[None, :, :]
    two_hop = left + right
    np.fill_diagonal(two_hop[:, :, :], np.inf)

    bypass = np.min(two_hop, axis=1)
    savings = np.maximum(bypass - clean, 0.0)
    savings = np.where(valid, savings, 0.0)
    savings = np.clip(savings / (scale + eps), 0.0, 20.0)
    bypass_bonus = 1.0 + 0.25 * np.tanh(savings)

    # Initial mass estimates used to balance endpoint degrees.
    raw_mass = desirability * regret_bonus * bypass_bonus
    raw_mass[~valid] = 0.0

    row_mass = np.sum(raw_mass, axis=1)
    col_mass = np.sum(raw_mass, axis=0)

    row_target = np.mean(row_mass[row_mass > 0.0]) if np.any(row_mass > 0.0) else 1.0
    col_target = np.mean(col_mass[col_mass > 0.0]) if np.any(col_mass > 0.0) else 1.0

    row_balance = np.sqrt(
        np.clip(row_target / (row_mass + eps), 0.25, 4.0)
    )[:, None]
    col_balance = np.sqrt(
        np.clip(col_target / (col_mass + eps), 0.25, 4.0)
    )[None, :]

    # Mild concave compression preserves alternative edges and diversity.
    priors = np.power(np.maximum(raw_mass, 0.0), 0.75)
    priors *= row_balance * col_balance
    priors[~valid] = 0.0

    maximum = float(np.max(priors))
    if maximum > 0.0 and np.isfinite(maximum):
        priors /= maximum

    priors[~np.isfinite(priors)] = 0.0
    np.fill_diagonal(priors, 0.0)
    return priors
