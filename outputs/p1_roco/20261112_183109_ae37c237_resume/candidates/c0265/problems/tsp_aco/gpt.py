import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge priorities based on global two-hop replacement cost."""
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

    with np.errstate(all="ignore"):
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

    normalized = np.divide(
        d,
        scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    # Global distance preference with a bounded rational decay.
    distance_term = np.divide(
        1.0,
        1.0 + np.maximum(normalized, 0.0),
        out=np.zeros((n, n), dtype=float),
        where=valid,
    )

    # Endpoint accessibility is computed from the whole valid graph.
    reciprocal = np.divide(
        1.0,
        1.0 + np.maximum(normalized, 0.0),
        out=np.zeros((n, n), dtype=float),
        where=valid,
    )
    accessibility = reciprocal.sum(axis=1)
    positive_accessibility = accessibility[accessibility > 0.0]

    if positive_accessibility.size:
        accessibility_reference = float(
            np.median(positive_accessibility)
        )
    else:
        accessibility_reference = 1.0

    accessibility_reference = max(
        accessibility_reference,
        eps,
    )

    endpoint_balance = np.sqrt(
        np.divide(
            accessibility[:, None],
            accessibility_reference,
            out=np.zeros((n, n), dtype=float),
        )
        * np.divide(
            accessibility[None, :],
            accessibility_reference,
            out=np.zeros((n, n), dtype=float),
        )
    )
    endpoint_balance = np.clip(endpoint_balance, 0.70, 1.30)

    # Degree balance favors edges that help low-connectivity endpoints.
    degree = valid.sum(axis=1).astype(float)
    degree_reference = max(
        float(np.median(degree[degree > 0.0]))
        if np.any(degree > 0.0)
        else 1.0,
        1.0,
    )

    degree_balance = np.sqrt(
        np.divide(
            degree_reference,
            np.maximum(degree[:, None], 1.0),
        )
        * np.divide(
            degree_reference,
            np.maximum(degree[None, :], 1.0),
        )
    )
    degree_balance = np.clip(degree_balance, 0.75, 1.35)

    # Compute the best feasible two-hop replacement for each edge.
    first_leg = d[:, :, None]
    second_leg = d[None, :, :]

    two_hop = first_leg + second_leg
    intermediary_valid = (
        valid[:, :, None]
        & valid[None, :, :]
    )

    indices = np.arange(n)
    not_endpoint = (
        (indices[None, None, :] != indices[:, None, :])
        & (indices[None, None, :] != indices[None, :, :])
    )

    intermediary_valid &= not_endpoint
    two_hop[~intermediary_valid] = np.inf

    best_two_hop = np.min(two_hop, axis=2)
    common_intermediaries = intermediary_valid.sum(axis=2)

    replacement_excess = np.divide(
        best_two_hop - d,
        scale,
        out=np.zeros((n, n), dtype=float),
        where=valid & np.isfinite(best_two_hop),
    )
    replacement_excess = np.maximum(replacement_excess, 0.0)

    replacement_term = 1.0 - np.exp(
        -np.clip(0.85 * replacement_excess, 0.0, 20.0)
    )
    replacement_term = np.where(
        common_intermediaries > 0,
        replacement_term,
        0.0,
    )

    # Average direct-edge savings over all feasible intermediaries.
    detour_saving = (
        d[:, :, None] * 0.0
        + d[:, None, :]
        + d[None, :, :]
        - d[:, :, None]
    )
    detour_saving = np.maximum(detour_saving, 0.0)
    detour_saving[~intermediary_valid] = 0.0

    saving_sum = detour_saving.sum(axis=2)
    saving_mean = np.divide(
        saving_sum,
        np.maximum(common_intermediaries, 1),
        out=np.zeros((n, n), dtype=float),
    )

    saving_term = 1.0 - np.exp(
        -np.clip(0.42 * saving_mean / scale, 0.0, 20.0)
    )

    # Edges with low global support are mildly downweighted unless
    # their two-hop replacement cost indicates structural necessity.
    edge_support = reciprocal
    support_reference_values = edge_support[valid]
    support_reference = (
        float(np.median(support_reference_values))
        if support_reference_values.size
        else 1.0
    )
    support_reference = max(support_reference, eps)

    support_term = np.sqrt(
        np.clip(
            edge_support / support_reference,
            0.55,
            1.45,
        )
    )

    score = (
        0.34 * distance_term
        + 0.24 * replacement_term
        + 0.18 * saving_term
        + 0.12 * np.clip(
            0.5 + 0.5 * endpoint_balance,
            0.0,
            1.0,
        )
        + 0.12 * np.clip(
            0.5 + 0.5 * degree_balance,
            0.0,
            1.0,
        )
    )

    score *= 0.92 + 0.08 * support_term
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
