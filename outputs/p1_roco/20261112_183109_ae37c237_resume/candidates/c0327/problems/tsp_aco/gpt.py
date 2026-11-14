import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return graph-competitive edge indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return scores

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        scale = float(np.median(positive_values))
    elif finite_values.size:
        scale = float(np.max(finite_values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    temperature = max(0.85 * scale, eps)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.clip(d[valid] / temperature, 0.0, 60.0)
    )
    np.fill_diagonal(affinity, 0.0)

    row_mass = affinity.sum(axis=1)
    safe_mass = np.maximum(row_mass, eps)

    transition = affinity / safe_mass[:, None]

    entropy = -np.sum(
        np.where(
            transition > 0.0,
            transition * np.log(np.maximum(transition, eps)),
            0.0,
        ),
        axis=1,
    )

    degree = np.maximum(np.sum(valid, axis=1).astype(float), 1.0)
    normalized_entropy = entropy / np.log(np.maximum(degree, 2.0))
    normalized_entropy = np.clip(
        np.nan_to_num(normalized_entropy, nan=1.0),
        0.0,
        1.0,
    )

    concentration = 1.0 - normalized_entropy
    endpoint_factor = 0.72 + 0.28 * np.sqrt(
        concentration[:, None] * concentration[None, :]
    )

    centrality = np.sqrt(
        np.maximum(row_mass, eps)
        * np.maximum(np.sum(affinity, axis=0), eps)
    )
    centrality_reference = float(np.median(centrality[centrality > 0.0]))
    if (
        not np.isfinite(centrality_reference)
        or centrality_reference <= 0.0
    ):
        centrality_reference = 1.0

    anti_hub = np.sqrt(
        centrality_reference
        / np.maximum(centrality, eps)
    )
    anti_hub = np.clip(anti_hub, 0.70, 1.35)
    anti_hub_factor = np.sqrt(
        anti_hub[:, None] * anti_hub[None, :]
    )

    two_step = transition @ transition
    two_step[~valid] = 0.0
    np.fill_diagonal(two_step, 0.0)

    two_step_reference = float(
        np.median(two_step[valid])
    ) if np.any(valid) else 0.0

    if (
        not np.isfinite(two_step_reference)
        or two_step_reference <= 0.0
    ):
        two_step_reference = 1.0

    directness = np.divide(
        transition,
        transition
        + 0.85 * two_step / two_step_reference,
        out=np.zeros_like(transition),
        where=valid,
    )
    directness = np.clip(directness, 0.0, 1.0)

    triangle_penalty = 1.0 - 0.16 * np.clip(
        two_step / np.maximum(
            two_step + transition,
            eps,
        ),
        0.0,
        1.0,
    )

    balance = np.sqrt(
        transition * transition.T
    )

    scores = (
        np.sqrt(affinity * affinity.T)
        * (
            0.46
            + 0.30 * balance
            + 0.24 * directness
        )
        * endpoint_factor
        * anti_hub_factor
        * triangle_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
