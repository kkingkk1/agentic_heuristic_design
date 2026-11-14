import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    positive = d[valid & (d > 0.0)]
    fallback = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    fallback = max(fallback, np.finfo(float).eps)

    work = np.where(valid, d, np.inf)

    # Robust outgoing and incoming local scales.
    k = min(4, n - 1)

    outgoing_sorted = np.sort(work, axis=1)
    incoming_sorted = np.sort(work, axis=0)

    outgoing_nearest = outgoing_sorted[:, :k]
    incoming_nearest = incoming_sorted[:k, :].T

    outgoing_finite = np.isfinite(outgoing_nearest)
    incoming_finite = np.isfinite(incoming_nearest)

    outgoing_scale = np.divide(
        np.where(outgoing_finite, outgoing_nearest, 0.0).sum(axis=1),
        outgoing_finite.sum(axis=1),
        out=np.full(n, fallback, dtype=float),
        where=outgoing_finite.sum(axis=1) > 0,
    )
    incoming_scale = np.divide(
        np.where(incoming_finite, incoming_nearest, 0.0).sum(axis=1),
        incoming_finite.sum(axis=1),
        out=np.full(n, fallback, dtype=float),
        where=incoming_finite.sum(axis=1) > 0,
    )

    outgoing_scale = np.maximum(outgoing_scale, np.finfo(float).eps)
    incoming_scale = np.maximum(incoming_scale, np.finfo(float).eps)

    geometric_scale = np.sqrt(
        outgoing_scale[:, None] * incoming_scale[None, :]
    )

    normalized_distance = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    clipped_distance = np.clip(normalized_distance, 0.0, 50.0)

    distance_quality = np.exp(-1.15 * clipped_distance)

    outgoing_ratio = np.divide(
        d,
        outgoing_scale[:, None],
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    incoming_ratio = np.divide(
        d,
        incoming_scale[None, :],
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    outgoing_quality = np.exp(-np.clip(outgoing_ratio, 0.0, 50.0))
    incoming_quality = np.exp(-np.clip(incoming_ratio, 0.0, 50.0))

    # Edge-specific agreement from both endpoint perspectives.
    mutual_support = np.sqrt(outgoing_quality * incoming_quality)

    scale_disagreement = np.abs(
        np.log(
            np.divide(
                outgoing_scale[:, None],
                incoming_scale[None, :],
                out=np.ones_like(d),
                where=(
                    (outgoing_scale[:, None] > 0.0)
                    & (incoming_scale[None, :] > 0.0)
                ),
            )
        )
    )
    scale_disagreement = np.clip(scale_disagreement, 0.0, 4.0)

    # Use more mutual evidence only where one-sided normalization is fragile.
    support_weight = 0.14 + 0.18 * (
        scale_disagreement / (1.0 + scale_disagreement)
    )

    mixed_quality = (
        (1.0 - support_weight) * distance_quality
        + support_weight * mutual_support
    )

    # Mild soft bridge preference around the local geometric scale.
    # It peaks near normalized distance 1 and decays for very long edges.
    bridge_shape = (
        2.0
        * clipped_distance
        * np.exp(-clipped_distance)
    )
    bridge_bonus = 1.0 + 0.035 * bridge_shape

    score = mixed_quality * bridge_bonus
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    degree = valid.sum(axis=1, keepdims=True)

    # Reserve a small, explicitly normalized residual probability floor.
    floor = 0.025
    floor_mass = np.minimum(floor * degree, 0.20)
    residual = np.maximum(1.0 - floor_mass, 0.0)

    normalized_score = np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )

    floor_per_edge = np.divide(
        floor_mass,
        degree,
        out=np.zeros_like(floor_mass),
        where=degree > 0,
    )

    result = floor_per_edge * valid + residual * normalized_score
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    return result
