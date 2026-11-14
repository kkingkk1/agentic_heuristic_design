import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally scaled, balanced edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    finite_values = distances[valid]
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

    normalized = np.divide(
        distances,
        scale,
        out=np.full_like(distances, np.inf, dtype=float),
        where=valid,
    )

    kernel = np.exp(
        -np.clip(normalized * normalized, 0.0, 60.0)
    )
    kernel[~valid] = 0.0
    np.fill_diagonal(kernel, 0.0)

    outgoing_mass = kernel.sum(axis=1)
    incoming_mass = kernel.sum(axis=0)

    safe_outgoing = np.maximum(outgoing_mass, eps)
    safe_incoming = np.maximum(incoming_mass, eps)

    continuation = np.sqrt(
        safe_outgoing[None, :] * safe_incoming[:, None]
    )
    continuation /= np.maximum(
        np.median(continuation[valid]),
        eps,
    )

    scores = kernel * np.sqrt(np.maximum(continuation, 0.0))
    scores[~valid] = 0.0

    active_rows = valid.any(axis=1)
    active_cols = valid.any(axis=0)

    for _ in range(8):
        row_totals = scores.sum(axis=1, keepdims=True)
        row_scale = np.divide(
            1.0,
            row_totals,
            out=np.zeros_like(row_totals),
            where=row_totals > 0.0,
        )
        scores *= row_scale
        scores[~valid] = 0.0

        col_totals = scores.sum(axis=0, keepdims=True)
        col_scale = np.divide(
            1.0,
            col_totals,
            out=np.zeros_like(col_totals),
            where=col_totals > 0.0,
        )
        scores *= col_scale
        scores[~valid] = 0.0

    row_totals = scores.sum(axis=1, keepdims=True)
    scores = np.divide(
        scores,
        row_totals,
        out=np.zeros_like(scores),
        where=row_totals > 0.0,
    )

    scores[~valid] = 0.0
    scores[~active_rows, :] = 0.0
    scores[:, ~active_cols] = 0.0

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
