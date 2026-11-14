import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed row-normalized one-step look-ahead edge priorities."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    priorities = np.zeros((n, n), dtype=float)

    if n < 2:
        return priorities

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return priorities

    eps = np.finfo(float).eps
    work = np.where(valid, distances, np.inf)

    outgoing_min = np.min(work, axis=1)
    incoming_min = np.min(work, axis=0)

    finite_values = distances[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif finite_values.size:
        global_scale = float(np.median(finite_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    safe_outgoing = np.where(
        np.isfinite(outgoing_min),
        outgoing_min,
        global_scale,
    )
    safe_incoming = np.where(
        np.isfinite(incoming_min),
        incoming_min,
        global_scale,
    )

    finite_max = float(np.max(finite_values))
    finite_max = max(finite_max, global_scale)

    continuation_scale = np.maximum(
        0.5 * (safe_outgoing[:, None] + safe_incoming[None, :]),
        eps,
    )

    lookahead_cost = (
        distances
        + 0.85 * safe_outgoing[None, :]
        + 0.85 * safe_incoming[:, None]
    )

    reference_cost = (
        global_scale
        + 0.85 * global_scale
        + 0.85 * global_scale
    )

    normalized_cost = np.divide(
        lookahead_cost,
        reference_cost + continuation_scale,
        out=np.full_like(distances, np.inf, dtype=float),
        where=valid,
    )

    normalized_cost = np.maximum(normalized_cost, 0.0)

    # A bounded reciprocal kernel keeps moderately long edges available
    # while strongly preferring edges with low current and continuation cost.
    quality = 1.0 / np.square(1.0 + normalized_cost)

    # Penalize edges whose endpoint continuation is unusually expensive.
    continuation_ratio = np.divide(
        continuation_scale,
        global_scale,
        out=np.full_like(distances, np.inf, dtype=float),
        where=valid,
    )
    continuation_penalty = 1.0 / (
        1.0 + 0.35 * np.maximum(continuation_ratio - 1.0, 0.0)
    )

    # Add a small uniform valid-edge floor for stochastic exploration.
    floor = 0.035
    priorities = floor + (1.0 - floor) * quality * continuation_penalty

    priorities[~valid] = 0.0
    np.fill_diagonal(priorities, 0.0)

    priorities = np.nan_to_num(
        priorities,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    row_totals = priorities.sum(axis=1, keepdims=True)

    return np.divide(
        priorities,
        row_totals,
        out=np.zeros_like(priorities),
        where=row_totals > 0.0,
    ).astype(float, copy=False)
