import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic directed edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(raw)
    valid = finite & (raw >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    observed = raw[valid]
    positive = observed[observed > 0.0]

    q25, q50, q75 = np.quantile(observed, [0.25, 0.50, 0.75])
    spread = float(q75 - q25)

    if not np.isfinite(spread) or spread <= 0.0:
        spread = float(q50)
    if not np.isfinite(spread) or spread <= 0.0:
        spread = float(np.mean(positive)) if positive.size else 1.0
    if not np.isfinite(spread) or spread <= 0.0:
        spread = 1.0

    scale = max(spread, np.finfo(float).eps)
    upper = float(np.quantile(observed, 0.98))
    if not np.isfinite(upper):
        upper = float(np.max(observed))

    replacement = max(upper, float(np.max(observed))) + 14.0 * scale
    clean = np.where(valid, raw, replacement).astype(float, copy=True)
    np.fill_diagonal(clean, replacement)

    eps = np.finfo(float).eps
    positive_scale = max(scale, eps)

    row_values = np.where(valid, clean, replacement)
    row_best = np.min(row_values, axis=1)
    row_second = np.partition(row_values, 1, axis=1)[:, 1]

    row_gap = np.maximum(row_second - row_best, 0.0)
    row_gap_scale = np.maximum(
        row_gap,
        0.15 * positive_scale,
    )

    local_preference = np.exp(
        -np.minimum(clean / positive_scale, 50.0)
    )
    local_preference[~valid] = 0.0

    temperature = float(
        np.median(
            np.log1p(observed / positive_scale)
        )
    )
    temperature = float(
        np.clip(0.65 + 0.55 * temperature, 0.65, 2.50)
    )

    logits = -clean / (positive_scale * temperature)
    logits = np.clip(logits, -60.0, 0.0)
    logits[~valid] = -np.inf

    transition = np.zeros((n, n), dtype=float)

    for i in range(n):
        allowed = valid[i]
        if not np.any(allowed):
            continue

        values = np.exp(logits[i, allowed])
        total = float(np.sum(values))

        if not np.isfinite(total) or total <= 0.0:
            transition[i, allowed] = 1.0 / float(np.sum(allowed))
        else:
            values /= total
            floor = min(0.08, 2.0 / max(float(np.sum(allowed)), 1.0))
            transition[i, allowed] = (
                (1.0 - floor) * values
                + floor / float(np.sum(allowed))
            )

    visitation = np.full(n, 1.0 / n, dtype=float)

    for _ in range(80):
        updated = visitation @ transition
        if not np.all(np.isfinite(updated)) or float(np.sum(updated)) <= 0.0:
            break

        updated /= float(np.sum(updated))

        if np.max(np.abs(updated - visitation)) < 1e-13:
            visitation = updated
            break

        visitation = updated

    edge_flow = visitation[:, None] * transition

    continuation = np.zeros(n, dtype=float)
    for j in range(n):
        allowed = valid[j]
        if not np.any(allowed):
            continuation[j] = replacement
            continue

        weights = transition[j].copy()
        weights[j] = 0.0
        weights_sum = float(np.sum(weights))

        if weights_sum <= eps:
            continuation[j] = float(np.mean(clean[j, allowed]))
        else:
            continuation[j] = float(
                np.sum(weights * clean[j]) / weights_sum
            )

    edge_risk = np.zeros((n, n), dtype=float)

    for i in range(n):
        targets = np.flatnonzero(valid[i])
        if targets.size == 0:
            continue

        immediate = clean[i, targets]
        future = continuation[targets]

        return_penalty = np.zeros(targets.size, dtype=float)
        reverse_valid = valid[targets, i]
        return_penalty[reverse_valid] = (
            transition[targets[reverse_valid], i]
            * clean[targets[reverse_valid], i]
        )

        total_cost = immediate + 0.72 * future + 0.18 * return_penalty
        normalized = total_cost / positive_scale

        edge_risk[i, targets] = np.exp(
            -np.minimum(normalized, 60.0)
        )

    row_risk_max = np.max(edge_risk, axis=1, keepdims=True)
    edge_risk /= np.maximum(row_risk_max, eps)

    destination_support = np.sum(transition, axis=0)
    destination_support /= max(
        float(np.max(destination_support)),
        eps,
    )

    reverse_flow = transition.T
    reciprocity_penalty = 1.0 - np.minimum(
        transition * reverse_flow,
        1.0,
    )

    flow_score = np.sqrt(
        np.maximum(edge_flow, 0.0)
        * np.maximum(destination_support[None, :], 0.0)
    )
    flow_max = float(np.max(flow_score))
    if flow_max > 0.0 and np.isfinite(flow_max):
        flow_score /= flow_max

    result = (
        0.48 * edge_risk
        + 0.34 * flow_score
        + 0.12 * local_preference
        + 0.06 * reciprocity_penalty
    )

    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
