import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    directed_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(directed_valid):
        return scores

    raw = np.where(directed_valid, distances, np.inf)
    reverse = raw.T

    forward_valid = np.isfinite(raw)
    backward_valid = np.isfinite(reverse)
    both_valid = forward_valid & backward_valid

    safe = np.full((n, n), np.inf, dtype=float)
    safe[both_valid] = 0.5 * (
        raw[both_valid] + reverse[both_valid]
    )

    only_forward = forward_valid & ~backward_valid
    only_backward = backward_valid & ~forward_valid

    safe[only_forward] = raw[only_forward]
    safe[only_backward] = reverse[only_backward]

    np.fill_diagonal(safe, np.inf)
    valid = np.isfinite(safe)

    if not np.any(valid):
        return scores

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = float(np.median(finite_values))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    scale = max(scale, eps)

    # Robust edge-length percentile score.
    sorted_values = np.sort(
        np.where(valid, safe, np.inf),
        axis=None,
    )
    finite_sorted = sorted_values[np.isfinite(sorted_values)]

    if finite_sorted.size:
        lower = float(np.percentile(finite_sorted, 10.0))
        upper = float(np.percentile(finite_sorted, 90.0))
    else:
        lower = 0.0
        upper = scale

    spread = max(upper - lower, 0.20 * scale, eps)

    length_position = np.clip(
        (safe - lower) / spread,
        0.0,
        8.0,
    )
    length_score = np.exp(-0.85 * length_position)
    length_score[~valid] = 0.0

    # Row-wise soft preferences estimate how concentrated each endpoint is.
    normalized_rows = np.divide(
        safe,
        scale,
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    normalized_rows = np.clip(normalized_rows, 0.0, 50.0)

    row_logits = -1.15 * normalized_rows
    row_logits[~valid] = -np.inf

    row_max = np.max(
        np.where(valid, row_logits, -np.inf),
        axis=1,
        keepdims=True,
    )

    stabilized = np.where(
        valid,
        np.exp(np.clip(row_logits - row_max, -60.0, 0.0)),
        0.0,
    )

    row_totals = stabilized.sum(axis=1, keepdims=True)
    preferences = np.divide(
        stabilized,
        row_totals,
        out=np.zeros_like(stabilized),
        where=row_totals > 0.0,
    )

    # Entropy measures whether a node has many plausible alternatives.
    candidate_counts = valid.sum(axis=1).astype(float)
    entropy = -np.sum(
        np.where(
            preferences > 0.0,
            preferences * np.log(np.maximum(preferences, eps)),
            0.0,
        ),
        axis=1,
    )

    entropy_reference = np.log(np.maximum(candidate_counts, 2.0))
    normalized_entropy = np.divide(
        entropy,
        np.maximum(entropy_reference, eps),
    )
    normalized_entropy = np.clip(normalized_entropy, 0.0, 1.0)

    endpoint_flexibility = np.sqrt(
        normalized_entropy[:, None]
        * normalized_entropy[None, :]
    )

    # Compute the best finite two-hop substitute for every edge.
    two_hop = np.full((n, n), np.inf, dtype=float)

    for middle in range(n):
        through_middle = (
            safe[:, middle, None]
            + safe[None, middle, :]
        )
        two_hop = np.minimum(two_hop, through_middle)

    np.fill_diagonal(two_hop, np.inf)

    direct_advantage = np.divide(
        two_hop - safe,
        scale,
        out=np.zeros((n, n), dtype=float),
        where=valid & np.isfinite(two_hop),
    )

    direct_advantage = np.clip(direct_advantage, -6.0, 6.0)

    # Smoothly reward edges that are difficult to replace by two-hop routes.
    shortcut_score = 0.5 + 0.5 * np.tanh(0.72 * direct_advantage)
    shortcut_score[~valid] = 0.0

    # Joint endpoint preference, with a mild anti-concentration correction.
    joint_preference = np.sqrt(
        preferences * preferences.T
    )

    support_score = 0.35 + 0.65 * np.maximum(
        preferences,
        preferences.T,
    )

    balance_score = 0.78 + 0.22 * endpoint_flexibility

    # Prefer edges that are jointly plausible but not solely supported by
    # one endpoint. The square root keeps this correction bounded.
    reciprocal_support = np.sqrt(
        np.maximum(joint_preference, 0.0)
        * np.maximum(support_score, 0.0)
    )

    scores = (
        length_score
        * (0.42 + 0.58 * shortcut_score)
        * (0.30 + 0.70 * reciprocal_support)
        * balance_score
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
