import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, normalized heuristic matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    diagonal = np.eye(n, dtype=bool)
    finite_forward = np.isfinite(distances)
    valid_forward = finite_forward & (distances >= 0.0) & ~diagonal

    raw = np.where(valid_forward, distances, np.inf)
    reverse = raw.T

    forward_exists = np.isfinite(raw)
    reverse_exists = np.isfinite(reverse)
    both_exists = forward_exists & reverse_exists

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both_exists] = (
        0.5 * (raw[both_exists] + reverse[both_exists])
    )

    only_forward = forward_exists & ~reverse_exists
    only_reverse = reverse_exists & ~forward_exists

    distance[only_forward] = raw[only_forward]
    distance[only_reverse] = reverse[only_reverse]

    np.fill_diagonal(distance, np.inf)
    valid = np.isfinite(distance)

    if not np.any(valid):
        return result

    values = distance[valid]
    positive_values = values[values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    valid_counts = valid.sum(axis=1).astype(float)
    max_neighbors = max(1, n - 1)
    neighborhood_size = min(
        max_neighbors,
        max(2, int(np.ceil(1.5 * np.sqrt(n)))),
    )

    order = np.argsort(distance, axis=1, kind="stable")
    sorted_distance = np.take_along_axis(distance, order, axis=1)
    sorted_valid = np.take_along_axis(valid, order, axis=1)

    local_values = np.where(
        sorted_valid[:, :neighborhood_size],
        sorted_distance[:, :neighborhood_size],
        global_scale,
    )

    local_scale = np.median(local_values, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale >= 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, 0.01 * global_scale)

    relative_local = np.divide(
        distance,
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )
    relative_global = np.divide(
        distance,
        global_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    relative_local = np.clip(relative_local, 0.0, 60.0)
    relative_global = np.clip(relative_global, 0.0, 60.0)

    local_affinity = np.exp(-0.82 * relative_local)
    global_affinity = np.exp(-0.24 * relative_global)

    local_affinity[~valid] = 0.0
    global_affinity[~valid] = 0.0

    logits = np.log(
        np.maximum(local_affinity, eps)
    ) + 0.30 * np.log(
        np.maximum(global_affinity, eps)
    )
    logits[~valid] = -np.inf

    row_max = np.max(
        np.where(valid, logits, -np.inf),
        axis=1,
        keepdims=True,
    )
    stabilized = np.where(valid, logits - row_max, -np.inf)

    sharp = np.where(valid, np.exp(stabilized), 0.0)
    sharp_sum = sharp.sum(axis=1, keepdims=True)
    sharp = np.divide(
        sharp,
        np.maximum(sharp_sum, eps),
        out=np.zeros_like(sharp),
        where=sharp_sum > 0.0,
    )

    entropy = -np.sum(
        np.where(
            sharp > 0.0,
            sharp * np.log(np.maximum(sharp, eps)),
            0.0,
        ),
        axis=1,
    )

    max_entropy = np.log(np.maximum(valid_counts, 2.0))
    entropy_ratio = np.divide(
        entropy,
        np.maximum(max_entropy, eps),
        out=np.zeros_like(entropy),
        where=max_entropy > 0.0,
    )
    entropy_ratio = np.clip(entropy_ratio, 0.0, 1.0)

    # Sparse nodes receive a lower exponent and therefore a flatter distribution.
    temperature_exponent = 0.62 + 0.68 * entropy_ratio
    calibrated = np.zeros_like(sharp)

    for index in range(n):
        if valid_counts[index] <= 0.0:
            continue

        row = np.where(valid[index], sharp[index], 0.0)
        powered = np.power(np.maximum(row, 0.0), temperature_exponent[index])
        total = float(np.sum(powered))

        if total > 0.0 and np.isfinite(total):
            calibrated[index] = powered / total
        else:
            calibrated[index] = (
                valid[index].astype(float)
                / max(valid_counts[index], 1.0)
            )

    np.fill_diagonal(calibrated, 0.0)

    # Opportunity measures how much useful outgoing choice exists at a node.
    outgoing_quality = np.max(calibrated, axis=1)
    opportunity = calibrated.sum(axis=0)
    opportunity = 0.5 * opportunity + 0.5 * outgoing_quality

    opportunity_scale = np.max(opportunity)
    if opportunity_scale > 0.0:
        opportunity = opportunity / opportunity_scale

    # Two-step reachability estimates whether an edge leads into a useful region.
    two_step = calibrated @ calibrated
    three_step = two_step @ calibrated

    two_step_max = np.max(two_step, axis=1, keepdims=True)
    two_step = np.divide(
        two_step,
        np.maximum(two_step_max, eps),
        out=np.zeros_like(two_step),
        where=two_step_max > 0.0,
    )

    three_step_max = np.max(three_step, axis=1, keepdims=True)
    three_step = np.divide(
        three_step,
        np.maximum(three_step_max, eps),
        out=np.zeros_like(three_step),
        where=three_step_max > 0.0,
    )

    long_term_reachability = 0.65 * two_step + 0.35 * three_step

    endpoint_future = np.sqrt(
        np.maximum(opportunity[:, None], 0.0)
        * np.maximum(opportunity[None, :], 0.0)
    )

    mutual_future = np.sqrt(
        np.maximum(long_term_reachability, 0.0)
        * np.maximum(long_term_reachability.T, 0.0)
    )

    # Cosine dissimilarity rewards globally useful bridge edges.
    row_norms = np.sqrt(np.sum(calibrated * calibrated, axis=1))
    row_norms = np.maximum(row_norms, eps)

    cosine = (calibrated @ calibrated.T) / (
        row_norms[:, None] * row_norms[None, :]
    )
    cosine = np.clip(cosine, 0.0, 1.0)

    bridge_diversity = 1.0 - cosine
    bridge_diversity = 0.55 + 0.45 * bridge_diversity

    # Mutual transition strength preserves local competitiveness.
    mutual_transition = np.sqrt(
        np.maximum(calibrated, 0.0)
        * np.maximum(calibrated.T, 0.0)
    )

    local_signal = np.exp(
        -0.68 * relative_local - 0.14 * relative_global
    )
    local_signal[~valid] = 0.0

    global_signal = np.exp(-0.20 * relative_global)
    global_signal[~valid] = 0.0

    long_term_signal = (
        0.58 * mutual_future
        + 0.24 * endpoint_future
        + 0.18 * mutual_transition
    )
    long_term_signal = np.clip(long_term_signal, 0.0, 1.0)

    scores = (
        0.46 * local_signal
        + 0.17 * global_signal
        + 0.21 * long_term_signal
        + 0.16 * bridge_diversity * mutual_transition
    )

    # A small global-diversity floor applies only to valid edges.
    diversity_floor = 0.018 + 0.052 * (
        1.0 - np.minimum(entropy_ratio[:, None], entropy_ratio[None, :])
    )
    scores += diversity_floor * bridge_diversity * global_signal

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
