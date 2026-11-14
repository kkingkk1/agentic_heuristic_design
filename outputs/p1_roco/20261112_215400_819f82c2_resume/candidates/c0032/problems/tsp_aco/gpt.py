import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return nonnegative symmetric edge priors for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square two-dimensional array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_off = np.isfinite(d) & ~diagonal

    if not np.any(finite_off):
        return np.zeros((n, n), dtype=float)

    finite_values = d[finite_off]
    positive_values = finite_values[finite_values > 0]
    fallback = float(np.max(positive_values)) if positive_values.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)

    cleaned = np.where(np.isfinite(d), d, fallback)
    cleaned = np.maximum(cleaned, 0.0)
    cleaned = np.minimum(cleaned, cleaned.T)
    np.fill_diagonal(cleaned, np.inf)

    valid = np.isfinite(cleaned)
    valid &= ~diagonal

    # Row-wise percentile ranks in [0, 1]. Lower values indicate shorter edges.
    ranks = np.ones((n, n), dtype=float)
    for i in range(n):
        order = np.argsort(cleaned[i], kind="mergesort")
        usable = order[valid[i, order]]
        if usable.size:
            ranks[i, usable] = np.arange(usable.size, dtype=float) / max(
                usable.size - 1, 1
            )

    ranks[~valid] = 1.0

    # Adaptive Gibbs transitions based only on ordinal local preference.
    row_spread = np.median(
        np.abs(ranks - np.median(ranks, axis=1, keepdims=True)),
        axis=1,
    )
    temperature = np.maximum(0.08 + 0.8 * row_spread, 0.08)

    logits = -ranks / temperature[:, None]
    logits[~valid] = -np.inf
    row_max = np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits - row_max)
    weights[~np.isfinite(weights)] = 0.0

    row_sum = weights.sum(axis=1, keepdims=True)
    transitions = np.divide(
        weights,
        row_sum,
        out=np.zeros_like(weights),
        where=row_sum > 0,
    )

    # Two-step support: edges that participate in plausible local continuations.
    two_step = transitions @ transitions
    two_step = np.clip(two_step, 0.0, None)
    two_step_scale = np.maximum(
        np.max(two_step, axis=1, keepdims=True),
        np.finfo(float).eps,
    )
    continuation = np.sqrt(
        (two_step / two_step_scale)
        * (two_step.T / np.maximum(two_step_scale.T, np.finfo(float).eps))
    )

    # Endpoint distribution dissimilarity favors inter-region connector edges.
    row_norm = np.linalg.norm(transitions, axis=1, keepdims=True)
    cosine = (transitions @ transitions.T) / (
        row_norm @ row_norm.T + np.finfo(float).eps
    )
    separation = np.clip(1.0 - cosine, 0.0, 1.0)

    # Compare direct rank against the strongest two-hop alternative.
    direct = np.minimum(ranks, ranks.T)
    via_score = np.minimum(
        np.min(ranks[:, :, None] + ranks[None, :, :], axis=1),
        np.min(ranks[:, :, None] + ranks[None, :, :], axis=2),
    )
    detour_margin = np.clip(via_score - direct, 0.0, 1.0)

    # Higher entropy indicates an endpoint is not trapped in one local choice.
    entropy = -np.sum(
        transitions * np.log(np.maximum(transitions, np.finfo(float).tiny)),
        axis=1,
    )
    entropy /= np.maximum(np.log(max(n - 1, 2)), np.finfo(float).eps)
    entropy_factor = 0.7 + 0.3 * np.sqrt(
        np.clip(entropy[:, None] * entropy[None, :], 0.0, 1.0)
    )

    base = np.exp(-2.5 * direct)
    score = (
        base
        * (0.55 + 0.45 * continuation)
        * (0.65 + 0.35 * separation)
        * (0.75 + 0.25 * detour_margin)
        * entropy_factor
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float)
