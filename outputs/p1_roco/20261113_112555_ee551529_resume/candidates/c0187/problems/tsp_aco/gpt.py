import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diverse, long-term-aware TSP edge priors."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    positive = d[finite & (d > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
        q25, q75 = np.percentile(positive, [25.0, 75.0])
    else:
        global_scale = 1.0
        q25, q75 = 1.0, 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)
    robust_width = max(float(q75 - q25), 0.25 * global_scale, eps)

    reverse_finite = finite.T
    both_valid = finite & reverse_finite
    forward_only = finite & ~reverse_finite
    backward_only = reverse_finite & ~finite

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[both_valid] = 0.5 * (d[both_valid] + d.T[both_valid])
    symmetric[forward_only] = d[forward_only]
    symmetric[backward_only] = d.T[backward_only]
    np.fill_diagonal(symmetric, np.inf)

    valid = np.isfinite(symmetric)
    if not np.any(valid):
        return result

    local_scale = np.full(n, global_scale, dtype=float)
    nearest = np.full(n, global_scale, dtype=float)
    second = np.full(n, global_scale, dtype=float)
    continuation = np.full(n, global_scale, dtype=float)

    k = min(max(4, int(np.ceil(np.sqrt(max(n, 2))))), 6, n - 1)

    for i in range(n):
        values = symmetric[i, valid[i]]
        if values.size == 0:
            continue

        values = np.sort(values)
        local_values = values[:k]
        local_scale[i] = max(float(np.median(local_values)), eps)
        nearest[i] = max(float(values[0]), eps)
        second[i] = max(float(values[1] if values.size > 1 else values[0]), eps)

        if values.size > 1:
            continuation[i] = max(
                float(np.mean(values[1:min(values.size, k + 1)])),
                eps,
            )
        else:
            continuation[i] = nearest[i]

    local_scale = np.maximum(local_scale, eps)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        local_relative = symmetric / np.sqrt(
            np.outer(local_scale, local_scale)
        )
        global_relative = symmetric / global_scale

        local_shortness = np.exp(-np.clip(local_relative, 0.0, 30.0))
        global_shortness = np.exp(
            -np.clip((symmetric - q25) / robust_width, -8.0, 30.0)
        )

        # A soft, bounded rank signal over the 4--6 nearest valid neighbors.
        local_rank = np.zeros((n, n), dtype=float)
        for i in range(n):
            row = symmetric[i].copy()
            row[~valid[i]] = np.inf
            order = np.argsort(row, kind="stable")
            for rank, j in enumerate(order[:k]):
                if np.isfinite(row[j]):
                    local_rank[i, j] = np.exp(-rank / max(k - 1, 1))

        rank_consensus = np.sqrt(local_rank * local_rank.T)

        # Weak reciprocal-top-neighborhood reinforcement.
        reciprocal_mass = np.sqrt(
            np.maximum(local_rank * local_rank.T, 0.0)
        )

        # Reward edges whose endpoints retain good alternatives after selection.
        edge_cost = np.maximum(symmetric, eps)
        endpoint_options = np.sqrt(
            np.outer(continuation, continuation)
        )
        option_ratio = endpoint_options / edge_cost
        future_potential = 0.65 + 0.35 * (
            1.0 - np.exp(-np.clip(option_ratio, 0.0, 8.0))
        )

        # Mild diversity term: do not eliminate globally useful, slightly longer
        # edges when they connect locally compatible endpoints.
        diversity = 0.75 + 0.25 * np.exp(
            -np.clip(np.abs(global_relative - local_relative), 0.0, 20.0)
        )

        scores = (
            (0.62 * local_shortness + 0.38 * global_shortness)
            * (0.70 + 0.30 * rank_consensus)
            * (1.0 + 0.16 * reciprocal_mass)
            * future_potential
            * diversity
        )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
