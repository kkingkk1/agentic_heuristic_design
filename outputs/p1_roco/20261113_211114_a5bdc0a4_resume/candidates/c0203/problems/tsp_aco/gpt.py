import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diffusion-consistent edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    offdiag = ~np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & (raw >= 0.0) & offdiag

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    observed = raw[valid]
    positive = observed[observed > 0.0]

    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    largest = float(np.max(observed)) if observed.size else scale
    penalty = largest + 16.0 * scale

    distances = np.full((n, n), penalty, dtype=float)
    distances[valid] = raw[valid]
    np.fill_diagonal(distances, penalty)

    valid_values = distances[valid]
    temperature = float(np.median(valid_values)) if valid_values.size else scale
    temperature = max(0.35 * temperature, np.finfo(float).eps)

    affinity = np.exp(
        -np.minimum(distances / temperature, 60.0)
    )
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    row_sum = affinity.sum(axis=1, keepdims=True)
    missing_rows = row_sum[:, 0] <= np.finfo(float).eps

    if np.any(missing_rows):
        for i in np.flatnonzero(missing_rows):
            allowed = valid[i]
            if np.any(allowed):
                affinity[i, allowed] = 1.0
        row_sum = affinity.sum(axis=1, keepdims=True)

    transition = affinity / np.maximum(
        row_sum,
        np.finfo(float).eps,
    )

    # Approximate a balanced transport matrix while preserving zero support.
    for _ in range(28):
        column_sum = transition.sum(axis=0, keepdims=True)
        transition /= np.maximum(
            column_sum,
            np.finfo(float).eps,
        )

        row_sum = transition.sum(axis=1, keepdims=True)
        transition /= np.maximum(
            row_sum,
            np.finfo(float).eps,
        )

    transition[~valid] = 0.0
    np.fill_diagonal(transition, 0.0)

    reverse_flow = transition * transition.T

    two_step = transition @ transition
    three_step = two_step @ transition

    # A directed edge is favored when low-cost diffusion can return to
    # its source after two or three additional transitions.
    three_cycle = transition * two_step.T
    four_cycle = transition * three_step.T

    cycle_max = max(
        float(np.max(three_cycle)),
        float(np.max(four_cycle)),
        np.finfo(float).eps,
    )
    three_cycle /= cycle_max
    four_cycle /= cycle_max

    reverse_max = max(
        float(np.max(reverse_flow)),
        np.finfo(float).eps,
    )
    reverse_flow /= reverse_max

    # Penalize edges that dominate neither their outgoing nor incoming flow.
    outgoing_peak = np.maximum(
        np.max(transition, axis=1, keepdims=True),
        np.finfo(float).eps,
    )
    incoming_peak = np.maximum(
        np.max(transition, axis=0, keepdims=True),
        np.finfo(float).eps,
    )

    relative_flow = np.sqrt(
        transition / outgoing_peak
        * transition / incoming_peak
    )
    relative_flow[~valid] = 0.0

    score = (
        0.42 * three_cycle
        + 0.28 * four_cycle
        + 0.18 * reverse_flow
        + 0.12 * relative_flow
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score = valid.astype(float)

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
