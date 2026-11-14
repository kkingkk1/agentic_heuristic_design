import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    clean = np.where(np.isfinite(raw), raw, 0.0)
    clean = np.maximum(clean, 0.0)
    clean = 0.5 * (clean + clean.T)

    observed = clean[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    maximum = float(np.max(observed)) if observed.size else scale
    invalid_penalty = maximum + 8.0 * scale

    distance = np.where(valid, clean, invalid_penalty)
    distance = 0.5 * (distance + distance.T)
    np.fill_diagonal(distance, invalid_penalty)

    finite_values = distance[~diagonal]
    finite_values = finite_values[np.isfinite(finite_values)]
    temperature = float(np.median(finite_values)) if finite_values.size else scale
    temperature = max(0.42 * temperature, np.finfo(float).eps)

    affinity = np.exp(-np.minimum(distance, invalid_penalty) / temperature)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    max_affinity = float(np.max(affinity))
    if not np.isfinite(max_affinity) or max_affinity <= 0.0:
        result = np.zeros((n, n), dtype=float)
        result[valid] = 1.0
        np.fill_diagonal(result, 0.0)
        return result

    base = affinity / max_affinity

    # Local rank quality: reward edges that are near-neighbors at both endpoints.
    local_scale = np.maximum(
        np.median(
            np.where(valid, distance, np.nan),
            axis=1,
        ),
        0.25 * scale,
    )
    local_scale = np.where(np.isfinite(local_scale), local_scale, scale)

    local_quality = np.exp(
        -distance / np.maximum(
            0.5 * (local_scale[:, None] + local_scale[None, :]),
            np.finfo(float).eps,
        )
    )
    local_quality[~valid] = 0.0
    np.fill_diagonal(local_quality, 0.0)

    # Endpoint competition: compare each edge with the best alternative at both ends.
    best_other = np.full(n, invalid_penalty, dtype=float)
    second_other = np.full(n, invalid_penalty, dtype=float)

    for i in range(n):
        values = distance[i].copy()
        values[i] = invalid_penalty
        values[~valid[i]] = invalid_penalty
        ordered = np.sort(values)
        best_other[i] = ordered[0]
        second_other[i] = ordered[1] if n > 2 else ordered[0]

    competition_scale = np.maximum(
        0.25 * scale,
        0.5 * (second_other - best_other),
    )
    competition_scale = np.maximum(
        competition_scale[:, None] + competition_scale[None, :],
        np.finfo(float).eps,
    )

    endpoint_score = np.exp(
        np.clip(
            (
                best_other[:, None]
                + best_other[None, :]
                - 2.0 * distance
            )
            / competition_scale,
            -20.0,
            20.0,
        )
    )
    endpoint_score[~valid] = 0.0
    np.fill_diagonal(endpoint_score, 0.0)
    endpoint_score = np.sqrt(endpoint_score * endpoint_score.T)
    endpoint_score /= max(float(np.max(endpoint_score)), np.finfo(float).eps)

    # Edge irreplaceability: quantify how much longer the best two-hop bypass is.
    bypass = np.full((n, n), invalid_penalty, dtype=float)
    for k in range(n):
        candidate = distance[:, k, None] + distance[None, k, :]
        bypass = np.minimum(bypass, candidate)

    np.fill_diagonal(bypass, invalid_penalty)
    advantage = np.maximum(bypass - distance, 0.0)
    bypass_scale = max(0.9 * scale, np.finfo(float).eps)
    irreplaceability = 1.0 - np.exp(-advantage / bypass_scale)
    irreplaceability[~valid] = 0.0
    np.fill_diagonal(irreplaceability, 0.0)

    # Shared-neighbor support: favor edges belonging to short local triangles.
    normalized_distance = distance / max(scale, np.finfo(float).eps)
    triangle_support = np.exp(
        -0.5 * (normalized_distance[:, :, None] + normalized_distance[None, :, :])
    )
    triangle_support[~valid] = 0.0
    triangle_support = np.sum(triangle_support, axis=2)
    triangle_support[~valid] = 0.0
    triangle_support -= np.min(triangle_support, axis=1, keepdims=True)
    triangle_denominator = np.max(triangle_support, axis=1, keepdims=True)
    triangle_support /= np.maximum(
        triangle_denominator,
        np.finfo(float).eps,
    )
    triangle_support = 0.5 * (
        triangle_support + triangle_support.T
    )
    triangle_support[~valid] = 0.0

    score = (
        0.48 * base
        + 0.22 * local_quality
        + 0.14 * endpoint_score
        + 0.10 * irreplaceability
        + 0.06 * triangle_support
    )
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    # Symmetric Sinkhorn-style balancing to reduce node-degree bias.
    balanced = score.copy()
    for _ in range(32):
        row_sum = np.maximum(
            balanced.sum(axis=1, keepdims=True),
            np.finfo(float).eps,
        )
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0
        np.fill_diagonal(balanced, 0.0)

    result = 0.72 * score + 0.28 * balanced
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)
    result = 0.5 * (result + result.T)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
