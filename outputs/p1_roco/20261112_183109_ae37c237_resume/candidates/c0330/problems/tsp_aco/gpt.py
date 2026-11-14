import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized TSP edge priorities from two-hop detour savings."""
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

    work = np.where(valid, distances, np.inf)
    np.fill_diagonal(work, np.inf)

    finite_values = work[valid]
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

    if n > 2:
        via = work[:, :, None] + work[None, :, :]
        middle_allowed = np.ones((n, n), dtype=bool)
        np.fill_diagonal(middle_allowed, False)
        via[:, ~middle_allowed] = np.inf
        two_hop = np.min(via, axis=1)
    else:
        two_hop = np.full((n, n), np.inf, dtype=float)

    finite_two_hop = np.isfinite(two_hop)

    detour_saving = np.zeros_like(work, dtype=float)
    saving_mask = valid & finite_two_hop
    detour_saving[saving_mask] = (
        two_hop[saving_mask] - work[saving_mask]
    ) / (
        two_hop[saving_mask] + work[saving_mask] + scale
    )
    detour_saving = np.clip(detour_saving, -1.0, 1.0)

    direct_attraction = 1.0 / (1.0 + work / scale)
    direct_attraction[~valid] = 0.0

    saving_bonus = 1.0 + 0.85 * np.maximum(detour_saving, 0.0)
    penalty = 1.0 - 0.25 * np.maximum(-detour_saving, 0.0)

    outgoing_mass = np.sum(
        np.where(valid, 1.0 / (1.0 + work / scale), 0.0),
        axis=1,
    )
    incoming_mass = np.sum(
        np.where(valid, 1.0 / (1.0 + work / scale), 0.0),
        axis=0,
    )

    outgoing_pressure = np.sqrt(
        np.maximum(np.median(outgoing_mass[outgoing_mass > 0.0])
                   if np.any(outgoing_mass > 0.0) else 1.0, eps)
        / np.maximum(outgoing_mass, eps)
    )
    incoming_pressure = np.sqrt(
        np.maximum(np.median(incoming_mass[incoming_mass > 0.0])
                   if np.any(incoming_mass > 0.0) else 1.0, eps)
        / np.maximum(incoming_mass, eps)
    )

    pressure = np.sqrt(
        outgoing_pressure[:, None] * incoming_pressure[None, :]
    )
    pressure = np.clip(pressure, 0.88, 1.16)

    reverse_valid = valid.T
    reverse_distance = distances.T
    reverse_scale = 1.0 / (1.0 + np.where(
        reverse_valid,
        reverse_distance / scale,
        np.inf,
    ))

    directional_agreement = np.ones_like(work, dtype=float)
    agreement_mask = valid & reverse_valid
    directional_agreement[agreement_mask] = np.sqrt(
        direct_attraction[agreement_mask]
        * reverse_scale[agreement_mask]
    )

    score = (
        direct_attraction
        * saving_bonus
        * penalty
        * pressure
        * directional_agreement
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = np.sum(score, axis=1, keepdims=True)
    result = np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
