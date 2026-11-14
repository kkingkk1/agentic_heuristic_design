import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally normalized, diffusion-aware TSP edge priorities."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_values = distances[np.isfinite(distances)]
    positive_values = finite_values[finite_values > 0.0]

    fallback = (
        float(np.max(positive_values))
        if positive_values.size
        else 1.0
    )
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.nan_to_num(
        distances,
        nan=fallback,
        posinf=fallback,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    positive = d[d > 0.0]
    if positive.size:
        temperature = float(np.median(positive))
    else:
        temperature = fallback

    temperature = max(temperature, np.finfo(float).eps)

    affinity = np.exp(
        -np.clip(d / temperature, 0.0, 60.0)
    )
    np.fill_diagonal(affinity, 0.0)

    row_mass = affinity.sum(axis=1)
    row_mass = np.maximum(row_mass, np.finfo(float).eps)
    transition = affinity / row_mass[:, None]

    two_step = transition @ transition

    direct_strength = np.sqrt(
        np.maximum(transition * transition.T, 0.0)
    )
    indirect_strength = np.sqrt(
        np.maximum(two_step * two_step.T, 0.0)
    )

    direct_advantage = direct_strength / (
        direct_strength + indirect_strength
        + np.finfo(float).eps
    )

    degree = transition.sum(axis=0)
    degree_scale = np.maximum(1.0 + degree, 1.0)
    anti_hub = 1.0 / np.sqrt(
        degree_scale[:, None] * degree_scale[None, :]
    )
    anti_hub = anti_hub / np.max(anti_hub)

    if n > 2:
        probabilities = np.clip(
            transition,
            np.finfo(float).eps,
            1.0,
        )
        entropy = -np.sum(
            probabilities * np.log(probabilities),
            axis=1,
        )
        max_entropy = np.log(float(n - 1))
        concentration = 1.0 - entropy / max_entropy
        concentration = np.clip(concentration, 0.0, 1.0)
    else:
        concentration = np.ones(n, dtype=float)

    endpoint_confidence = 0.5 + 0.5 * np.sqrt(
        concentration[:, None] * concentration[None, :]
    )

    direct_affinity = np.sqrt(
        np.maximum(affinity * affinity.T, 0.0)
    )

    score = (
        direct_affinity
        * (0.20 + 0.80 * direct_advantage)
        * endpoint_confidence
        * (0.75 + 0.25 * anti_hub)
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
