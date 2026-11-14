import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(directed_valid):
        return scores

    forward = np.where(directed_valid, distances, np.inf)
    backward = forward.T

    valid = np.isfinite(forward) | np.isfinite(backward)
    safe = np.minimum(forward, backward)
    safe[~valid] = np.inf
    np.fill_diagonal(safe, np.inf)

    valid = np.isfinite(safe)
    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = float(np.median(finite_values))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    conductance = np.zeros((n, n), dtype=float)
    conductance[valid] = 1.0 / (safe[valid] + scale)

    degree = conductance.sum(axis=1)
    degree_reference = float(np.median(degree[degree > 0.0]))
    if not np.isfinite(degree_reference) or degree_reference <= 0.0:
        degree_reference = 1.0

    normalized_degree = degree / degree_reference
    degree_balance = 1.0 / np.sqrt(
        (1.0 + np.abs(normalized_degree[:, None] - 1.0))
        * (1.0 + np.abs(normalized_degree[None, :] - 1.0))
    )

    laplacian = -conductance.copy()
    np.fill_diagonal(laplacian, degree)

    try:
        laplacian_pinv = np.linalg.pinv(
            laplacian,
            rcond=1e-10,
        )
        resistance = (
            np.diag(laplacian_pinv)[:, None]
            + np.diag(laplacian_pinv)[None, :]
            - 2.0 * laplacian_pinv
        )
        resistance = np.maximum(resistance, 0.0)
    except np.linalg.LinAlgError:
        resistance = np.zeros((n, n), dtype=float)

    resistance[~valid] = 0.0

    positive_resistance = resistance[valid & (resistance > eps)]
    if positive_resistance.size:
        resistance_scale = float(np.median(positive_resistance))
    else:
        resistance_scale = 1.0
    resistance_scale = max(resistance_scale, eps)

    bridge_score = np.sqrt(
        np.clip(resistance / resistance_scale, 0.0, 8.0)
    )
    bridge_score[~valid] = 0.0

    incident_strength = conductance.sum(axis=1)
    edge_share = np.zeros((n, n), dtype=float)

    edge_share[valid] = conductance[valid] / np.sqrt(
        np.maximum(
            incident_strength[:, None][valid]
            * incident_strength[None, :][valid],
            eps,
        )
    )

    local_competition = np.zeros((n, n), dtype=float)
    for i in range(n):
        row = safe[i]
        finite_row = row[np.isfinite(row)]

        if finite_row.size == 0:
            continue

        reference = float(np.median(finite_row))
        reference = max(reference, eps)

        local_competition[i, valid[i]] = 1.0 / (
            1.0
            + np.maximum(
                safe[i, valid[i]] / reference - 1.0,
                0.0,
            )
        )

    local_competition = np.sqrt(
        local_competition * local_competition.T
    )

    conductance_score = np.zeros((n, n), dtype=float)
    conductance_score[valid] = conductance[valid] / max(
        float(np.max(conductance[valid])),
        eps,
    )

    bridge_score = bridge_score / max(
        float(np.max(bridge_score)),
        eps,
    )

    edge_share = np.clip(edge_share, 0.0, 1.0)

    scores = (
        (0.28 + 0.72 * conductance_score)
        * (0.48 + 0.52 * bridge_score)
        * (0.62 + 0.38 * degree_balance)
        * (0.58 + 0.42 * local_competition)
        * (0.70 + 0.30 * edge_share)
    )

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
