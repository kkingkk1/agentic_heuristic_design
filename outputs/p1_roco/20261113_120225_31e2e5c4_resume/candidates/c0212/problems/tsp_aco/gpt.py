import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a topology-aware TSP edge-priority matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(distances) & (distances >= 0.0)
    diagonal = np.eye(n, dtype=bool)
    finite &= ~diagonal

    if not np.any(finite):
        return result

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[finite] = distances[finite]
    symmetric = np.minimum(symmetric, symmetric.T)
    np.fill_diagonal(symmetric, np.inf)

    valid = np.isfinite(symmetric)
    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    positive = symmetric[valid & (symmetric > 0.0)]

    if positive.size:
        reference = float(np.median(positive))
        lower_scale = float(np.percentile(positive, 25.0))
        upper_scale = float(np.percentile(positive, 75.0))
    else:
        finite_values = symmetric[valid]
        reference = max(float(np.median(finite_values)), 1.0)
        lower_scale = reference
        upper_scale = reference

    reference = max(reference, eps)
    spread = max(upper_scale - lower_scale, eps)

    safe = np.where(valid, symmetric, np.inf)

    if n > 2:
        through = safe[:, :, None] + safe[None, :, :]
        through = np.transpose(through, (0, 2, 1))

        same_left = np.arange(n)[:, None, None] == np.arange(n)[None, None, :]
        same_right = np.arange(n)[None, :, None] == np.arange(n)[None, None, :]
        through[same_left | same_right] = np.inf

        two_hop = np.min(through, axis=2)
    else:
        two_hop = np.full((n, n), np.inf, dtype=float)

    direct = safe
    finite_detour = np.isfinite(two_hop)

    with np.errstate(divide="ignore", invalid="ignore"):
        detour_advantage = (two_hop - direct) / (
            two_hop + direct + reference
        )

    detour_advantage = np.nan_to_num(
        detour_advantage,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    detour_advantage = np.clip(detour_advantage, -1.0, 1.0)

    direct_scale = np.exp(
        -np.clip((direct - lower_scale) / (spread + reference), -4.0, 8.0)
    )

    if n > 2:
        triangle_gap = through - direct[:, :, None]
        triangle_gap = np.where(
            np.isfinite(triangle_gap),
            triangle_gap,
            np.inf,
        )

        supported = triangle_gap <= (0.35 * (reference + direct[:, :, None]))
        support_count = supported.sum(axis=2).astype(float)
        support_fraction = support_count / max(n - 2, 1)
    else:
        support_fraction = np.zeros((n, n), dtype=float)

    support_fraction = np.clip(
        np.nan_to_num(
            support_fraction,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )

    structural_score = (
        1.0
        + 0.85 * np.maximum(detour_advantage, 0.0)
        + 0.20 * support_fraction
        - 0.20 * np.maximum(-detour_advantage, 0.0)
    )

    scores = direct_scale * np.maximum(structural_score, 0.05)
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
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
