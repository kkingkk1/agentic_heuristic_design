import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized directional edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    finite_values = distances[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        scale = float(np.median(positive_values))
    elif finite_values.size:
        scale = float(np.median(finite_values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    work = np.where(valid, distances, np.inf)

    normalized = np.divide(
        work,
        scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    distance_term = np.exp(
        -np.clip(normalized / 0.78, 0.0, 60.0)
    )
    distance_term[~valid] = 0.0

    if n > 2:
        indices = np.arange(n)

        # For edge i -> j, evaluate continuation j -> k while
        # excluding the starting node i.
        outgoing_values = work[None, :, :]
        outgoing_mask = valid[None, :, :]
        exclude_start = indices[:, None, None] != indices[None, None, :]
        outgoing_mask = outgoing_mask & exclude_start

        outgoing_values = np.where(
            outgoing_mask,
            outgoing_values,
            np.inf,
        )
        continuation_out = np.min(outgoing_values, axis=2)

        # For edge i -> j, evaluate predecessor k -> i while
        # excluding the destination node j.
        incoming_values = work.T[:, None, :]
        incoming_mask = valid.T[:, None, :]
        exclude_destination = indices[None, :, None] != indices[None, None, :]
        incoming_mask = incoming_mask & exclude_destination

        incoming_values = np.where(
            incoming_mask,
            incoming_values,
            np.inf,
        )
        continuation_in = np.min(incoming_values, axis=2)

        continuation_cost = (
            continuation_out + continuation_in
        )

        continuation_cost = np.where(
            np.isfinite(continuation_cost),
            continuation_cost,
            2.0 * scale,
        )

        continuation_term = np.exp(
            -np.clip(0.34 * continuation_cost / scale, 0.0, 40.0)
        )

        # Cheap alternative two-step returns indicate a tendency to
        # close a short local cycle, so mildly suppress such edges.
        cycle_values = (
            work[None, :, :] + work.T[:, None, :]
        )

        cycle_mask = (
            valid[None, :, :]
            & valid.T[:, None, :]
            & exclude_start
            & exclude_destination
        )

        cycle_values = np.where(
            cycle_mask,
            cycle_values,
            np.inf,
        )
        alternate_cycle = np.min(cycle_values, axis=2)
        alternate_cycle = np.where(
            np.isfinite(alternate_cycle),
            alternate_cycle,
            4.0 * scale,
        )

        closure_strength = np.exp(
            -np.clip(alternate_cycle / scale, 0.0, 40.0)
        )
        closure_term = 1.0 - 0.16 * closure_strength
    else:
        continuation_term = np.ones((n, n), dtype=float)
        closure_term = np.ones((n, n), dtype=float)

    incoming_affinity = distance_term.sum(axis=0)
    positive_affinity = incoming_affinity[incoming_affinity > 0.0]

    if positive_affinity.size:
        affinity_reference = float(np.median(positive_affinity))
    else:
        affinity_reference = 1.0

    if (
        not np.isfinite(affinity_reference)
        or affinity_reference <= 0.0
    ):
        affinity_reference = 1.0

    destination_balance = np.sqrt(
        affinity_reference
        / np.maximum(incoming_affinity, eps)
    )
    destination_balance = np.clip(
        destination_balance,
        0.82,
        1.18,
    )

    balance_term = destination_balance[None, :]

    scores = (
        distance_term
        * (0.72 + 0.28 * continuation_term)
        * closure_term
        * balance_term
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    row_totals = scores.sum(axis=1, keepdims=True)

    return np.divide(
        scores,
        row_totals,
        out=result,
        where=row_totals > 0.0,
    )
