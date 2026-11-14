import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return TSP edge priors from multi-temperature symmetric soft 2-matching."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = raw.copy()
    finite_positive = np.isfinite(d) & (d > 0.0)

    if np.any(finite_positive):
        positive = d[finite_positive]
        fallback = float(np.max(positive))
    else:
        return np.zeros((n, n), dtype=float)

    d[~np.isfinite(d)] = fallback
    d[d < 0.0] = fallback
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    positive_edges = d[off_diagonal & (d > 0.0)]

    if positive_edges.size:
        base_scale = float(np.median(positive_edges))
        upper_scale = float(np.quantile(positive_edges, 0.75))
    else:
        base_scale = 1.0
        upper_scale = 1.0

    eps = np.finfo(float).eps
    base_scale = max(base_scale, eps)
    upper_scale = max(upper_scale, base_scale)

    temperatures = np.array(
        [
            0.35 * base_scale,
            0.75 * base_scale,
            1.50 * base_scale,
            max(upper_scale, 2.50 * base_scale),
        ],
        dtype=float,
    )

    accumulated = np.zeros((n, n), dtype=float)
    stable_floor = np.full((n, n), np.inf, dtype=float)

    for temperature in temperatures:
        logits = -d / max(float(temperature), eps)
        logits = np.clip(logits, -60.0, 0.0)

        affinity = np.exp(logits)
        np.fill_diagonal(affinity, 0.0)
        affinity = np.maximum(affinity, eps)

        scaling = np.ones(n, dtype=float)

        for _ in range(400):
            row_mass = affinity @ scaling
            updated = 2.0 / np.maximum(row_mass, eps)

            if np.max(np.abs(updated - scaling)) < 1e-10:
                scaling = updated
                break

            scaling = 0.5 * scaling + 0.5 * updated

        soft_matching = scaling[:, None] * affinity * scaling[None, :]
        soft_matching = 0.5 * (soft_matching + soft_matching.T)
        np.fill_diagonal(soft_matching, 0.0)

        accumulated += soft_matching
        stable_floor = np.minimum(stable_floor, soft_matching)

    mean_mass = accumulated / float(len(temperatures))
    floor_mass = np.where(np.isfinite(stable_floor), stable_floor, 0.0)

    variability = np.zeros_like(mean_mass)
    variability = np.maximum(mean_mass - floor_mass, 0.0)

    score = 0.65 * floor_mass + 0.35 * np.maximum(mean_mass - variability, 0.0)
    score = 0.5 * (score + score.T)
    score[~off_diagonal] = 0.0

    maximum = float(np.max(score))
    minimum = float(np.min(score[off_diagonal]))

    if np.isfinite(maximum) and maximum > minimum:
        score = (score - minimum) / (maximum - minimum)
    elif np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    return np.clip(score, 0.0, 1.0).astype(float, copy=False)
