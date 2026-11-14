import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded, direction-aware edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(raw)
    offdiag = ~np.eye(n, dtype=bool)
    valid = finite & (raw >= 0.0) & offdiag

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    observed = raw[valid]
    positive = observed[observed > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    q25, q75 = np.quantile(observed, [0.25, 0.75])
    robust_width = float(q75 - q25)
    if not np.isfinite(robust_width) or robust_width <= 0.0:
        robust_width = scale

    maximum = float(np.max(observed))
    if not np.isfinite(maximum):
        maximum = scale

    penalty = maximum + 16.0 * max(scale, robust_width)
    if not np.isfinite(penalty) or penalty <= 0.0:
        penalty = 16.0 * scale

    distances = np.full((n, n), penalty, dtype=float)
    distances[valid] = raw[valid]
    np.fill_diagonal(distances, penalty)

    eps = np.finfo(float).eps
    temperature = max(0.55 * robust_width, 0.20 * scale, eps)

    affinity = np.exp(-np.minimum(distances / temperature, 50.0))
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    affinity_max = float(np.max(affinity))
    if np.isfinite(affinity_max) and affinity_max > eps:
        affinity /= affinity_max

    def directional_ranks(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
        ranks = np.zeros_like(matrix, dtype=float)
        for row in range(n):
            indices = np.flatnonzero(mask[row])
            if indices.size:
                order = indices[np.argsort(matrix[row, indices], kind="stable")]
                ranks[row, order] = np.arange(order.size, dtype=float)
                ranks[row, order] /= max(float(order.size - 1), 1.0)
        return ranks

    outgoing_rank = directional_ranks(distances, valid)

    incoming_rank = directional_ranks(
        distances.T,
        valid.T,
    ).T

    rank_signal = 1.0 / np.sqrt(
        (1.0 + 3.0 * outgoing_rank)
        * (1.0 + 3.0 * incoming_rank)
    )
    rank_signal[~valid] = 0.0

    def alternative_gap(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = np.zeros_like(matrix, dtype=float)

        for row in range(n):
            indices = np.flatnonzero(mask[row])
            if indices.size == 0:
                continue

            values = matrix[row, indices]
            best = float(np.min(values))

            if indices.size >= 2:
                smallest = np.partition(values, 1)[:2]
                second = float(np.max(smallest))
            else:
                second = best + robust_width

            local_scale = max(second - best, 0.20 * robust_width, eps)
            gap = np.maximum(second - values, 0.0)
            signal = np.clip(gap / local_scale, 0.0, 4.0) / 4.0
            result[row, indices] = signal

        return result

    outgoing_pressure = alternative_gap(distances, valid)
    incoming_pressure = alternative_gap(distances.T, valid.T).T

    competition = np.sqrt(outgoing_pressure * incoming_pressure)
    competition[~valid] = 0.0

    # Approximate directed two-step lookahead using a small set of outgoing
    # nearest intermediates. This preserves useful bridge evidence without
    # constructing a full cubic all-intermediate tensor.
    two_step = np.full((n, n), np.inf, dtype=float)
    candidate_count = min(6, max(1, n - 1))

    for source in range(n):
        neighbors = np.flatnonzero(valid[source])
        if neighbors.size == 0:
            continue

        ordered = neighbors[
            np.argsort(distances[source, neighbors], kind="stable")
        ]
        intermediates = ordered[:candidate_count]

        for middle in intermediates:
            reachable = valid[middle]
            candidate = distances[source, middle] + distances[middle]
            update = reachable & (candidate < two_step[source])
            two_step[source, update] = candidate[update]

    bridge_advantage = np.zeros((n, n), dtype=float)
    available_two_step = valid & np.isfinite(two_step)

    if np.any(available_two_step):
        advantage = np.maximum(
            two_step[available_two_step] - distances[available_two_step],
            0.0,
        )
        bridge_advantage[available_two_step] = 1.0 - np.exp(
            -advantage / max(0.85 * robust_width, eps)
        )

    bridge_advantage = np.clip(bridge_advantage, 0.0, 1.0)

    # Additive fusion keeps the main distance signal dominant while allowing
    # local competition and bridge evidence to contribute conservatively.
    score = (
        0.50 * affinity
        + 0.24 * rank_signal
        + 0.16 * competition
        + 0.10 * bridge_advantage
    )
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    # Direction-preserving row normalization.
    row_sum = score.sum(axis=1, keepdims=True)
    normalized = np.divide(
        score,
        np.maximum(row_sum, eps),
        out=np.zeros_like(score),
        where=row_sum > eps,
    )

    # A small valid-edge floor prevents deterministic collapse onto one
    # nearest-neighbor pattern while leaving invalid edges at exactly zero.
    degree = valid.sum(axis=1, keepdims=True).astype(float)
    floor_mass = 0.035
    exploration = np.divide(
        valid.astype(float),
        np.maximum(degree, 1.0),
    )

    result = (1.0 - floor_mass) * normalized + floor_mass * exploration
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > eps:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
