import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diffusion-based edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & (raw >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    observed = raw[valid]
    positive = observed[observed > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = float(np.median(observed))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    q90 = float(np.quantile(observed, 0.90))
    penalty = max(q90, scale) + 10.0 * scale
    eps = max(np.finfo(float).eps * scale, 1e-12)

    distances = np.full((n, n), penalty, dtype=float)
    distances[valid] = raw[valid]
    np.fill_diagonal(distances, penalty)

    temperature = max(
        0.55 * scale
        + 0.20 * float(np.std(observed)),
        eps,
    )

    logits = -np.minimum(distances / temperature, 60.0)
    logits[~valid] = -60.0
    logits -= np.max(logits, axis=1, keepdims=True)

    transition = np.exp(logits)
    transition[~valid] = 0.0

    row_total = transition.sum(axis=1, keepdims=True)
    transition /= np.maximum(row_total, eps)

    incoming_total = transition.sum(axis=0, keepdims=True)
    incoming_share = transition / np.maximum(incoming_total, eps)

    outgoing_share = transition.copy()

    outgoing_entropy = -np.sum(
        outgoing_share
        * np.log(np.maximum(outgoing_share, eps)),
        axis=1,
        keepdims=True,
    )
    incoming_entropy = -np.sum(
        incoming_share
        * np.log(np.maximum(incoming_share, eps)),
        axis=0,
        keepdims=True,
    )

    outgoing_concentration = 1.0 / (
        1.0 + outgoing_entropy
    )
    incoming_concentration = 1.0 / (
        1.0 + incoming_entropy
    )

    endpoint_scarcity = np.sqrt(
        np.maximum(
            outgoing_share * incoming_share,
            0.0,
        )
    )

    second_order = transition @ transition
    second_order /= np.maximum(
        second_order.max(axis=1, keepdims=True),
        eps,
    )

    direct_strength = transition / np.maximum(
        transition.max(axis=1, keepdims=True),
        eps,
    )

    diffusion_advantage = np.maximum(
        direct_strength - second_order,
        0.0,
    )

    ranks = np.argsort(
        np.argsort(
            distances,
            axis=1,
            kind="stable",
        ),
        axis=1,
    ).astype(float)

    rank_decay = max(
        1.0,
        0.65 * np.sqrt(float(n)),
    )
    rank_preference = np.exp(
        -ranks / rank_decay
    )

    row_best = np.min(
        distances,
        axis=1,
        keepdims=True,
    )
    col_best = np.min(
        distances,
        axis=0,
        keepdims=True,
    )

    row_regret = np.maximum(
        distances - row_best,
        0.0,
    )
    col_regret = np.maximum(
        distances - col_best,
        0.0,
    )

    regret_scale = max(
        0.75 * scale,
        eps,
    )

    insertion_urgency = np.exp(
        -(
            row_regret
            + col_regret
        )
        / regret_scale
    )

    local_affinity = np.exp(
        -np.minimum(
            distances / max(temperature, eps),
            60.0,
        )
    )

    local_affinity[~valid] = 0.0
    local_affinity /= max(
        float(local_affinity.max()),
        eps,
    )

    entropy_weight = np.sqrt(
        np.maximum(
            outgoing_concentration
            * incoming_concentration,
            0.0,
        )
    )

    score = (
        0.30 * local_affinity
        + 0.22 * endpoint_scarcity
        + 0.18 * rank_preference
        + 0.16 * diffusion_advantage
        + 0.14 * insertion_urgency
    )

    score *= (
        0.70
        + 0.30 * entropy_weight
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    balanced = score.copy()

    for _ in range(20):
        row_norm = balanced.sum(
            axis=1,
            keepdims=True,
        )
        balanced /= np.maximum(
            row_norm,
            eps,
        )

        col_norm = balanced.sum(
            axis=0,
            keepdims=True,
        )
        balanced /= np.maximum(
            col_norm,
            eps,
        )

    result = (
        0.72 * score
        + 0.28 * balanced
    )

    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum
    else:
        result.fill(0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
