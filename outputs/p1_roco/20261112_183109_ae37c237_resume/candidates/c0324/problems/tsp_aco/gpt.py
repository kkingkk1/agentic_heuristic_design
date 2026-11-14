import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized continuation-aware edge priorities for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros_like(distances, dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    positive = distances[valid & (distances > 0.0)]
    if positive.size:
        scale = float(np.median(positive))
    else:
        finite = distances[valid]
        scale = float(np.max(finite)) if finite.size else 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    safe = np.where(valid, distances, np.nan)

    outgoing_counts = valid.sum(axis=1).astype(float)
    incoming_counts = valid.sum(axis=0).astype(float)

    outgoing_sum = np.nansum(np.where(valid, distances, 0.0), axis=1)
    incoming_sum = np.nansum(np.where(valid, distances, 0.0), axis=0)

    outgoing_mean = np.divide(
        outgoing_sum,
        np.maximum(outgoing_counts, 1.0),
    )
    incoming_mean = np.divide(
        incoming_sum,
        np.maximum(incoming_counts, 1.0),
    )

    outgoing_mean = np.where(
        outgoing_counts > 0.0,
        outgoing_mean,
        scale,
    )
    incoming_mean = np.where(
        incoming_counts > 0.0,
        incoming_mean,
        scale,
    )

    accessibility = np.sqrt(
        np.maximum(outgoing_mean, np.finfo(float).eps)
        * np.maximum(incoming_mean, np.finfo(float).eps)
    )

    edge_scale = np.sqrt(
        accessibility[:, None] * accessibility[None, :]
    )

    relative = np.divide(
        np.where(valid, distances, 0.0),
        np.maximum(edge_scale, np.finfo(float).eps),
    )

    immediate = -np.log1p(relative)

    continuation = -0.5 * np.log1p(
        np.maximum(outgoing_mean[None, :], 0.0) / scale
    )

    origin_support = -0.5 * np.log1p(
        np.maximum(incoming_mean[:, None], 0.0) / scale
    )

    incoming_pressure = np.sum(
        np.exp(
            -np.divide(
                np.where(valid, distances, 0.0),
                scale,
            )
        )
        * valid,
        axis=0,
    )

    pressure_reference = np.median(
        incoming_pressure[incoming_pressure > 0.0]
    ) if np.any(incoming_pressure > 0.0) else 1.0

    if not np.isfinite(pressure_reference) or pressure_reference <= 0.0:
        pressure_reference = 1.0

    congestion = -0.18 * np.log1p(
        incoming_pressure[None, :] / pressure_reference
    )

    utility = (
        2.20 * immediate
        + 0.55 * continuation
        + 0.35 * origin_support
        + congestion
    )

    utility[~valid] = -np.inf

    finite_utility = np.where(valid, utility, np.nan)
    row_max = np.nanmax(finite_utility, axis=1)

    row_min = np.nanmin(finite_utility, axis=1)
    row_spread = np.maximum(row_max - row_min, 0.0)

    temperature = 0.55 + 0.45 / (1.0 + row_spread)
    temperature = np.maximum(temperature, 0.25)

    logits = np.divide(
        utility - row_max[:, None],
        temperature[:, None],
    )
    logits[~valid] = -np.inf

    weights = np.zeros((n, n), dtype=float)
    finite_logits = np.isfinite(logits)

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        weights[finite_logits] = np.exp(logits[finite_logits])

    totals = weights.sum(axis=1, keepdims=True)

    nonempty = valid.any(axis=1)
    zero_rows = nonempty & (totals[:, 0] <= 0.0)

    if np.any(zero_rows):
        weights[zero_rows] = valid[zero_rows].astype(float)
        totals = weights.sum(axis=1, keepdims=True)

    result = np.divide(
        weights,
        totals,
        out=np.zeros_like(weights),
        where=totals > 0.0,
    )

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
