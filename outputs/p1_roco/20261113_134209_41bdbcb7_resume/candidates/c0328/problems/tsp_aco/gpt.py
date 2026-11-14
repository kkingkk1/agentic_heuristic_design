import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(finite_nonnegative):
        return result

    forward = np.where(finite_nonnegative, distances, np.inf)
    backward = forward.T

    # Build one symmetric edge length from whichever directed entries exist.
    both = np.isfinite(forward) & np.isfinite(backward)
    only_forward = np.isfinite(forward) & ~np.isfinite(backward)
    only_backward = np.isfinite(backward) & ~np.isfinite(forward)

    edge_distance = np.full((n, n), np.inf, dtype=float)
    edge_distance[both] = 0.5 * (
        forward[both] + backward[both]
    )
    edge_distance[only_forward] = forward[only_forward]
    edge_distance[only_backward] = backward[only_backward]
    np.fill_diagonal(edge_distance, np.inf)

    valid = np.isfinite(edge_distance) & ~diagonal
    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    finite_values = edge_distance[valid]

    positive_values = finite_values[finite_values > 0.0]
    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = max(float(np.median(finite_values)), 1.0)

    scale = max(scale, eps)

    # A bounded conductance transform preserves zero-cost edges without
    # allowing a single tiny distance to dominate all other edges.
    transformed = np.zeros((n, n), dtype=float)
    normalized = edge_distance / scale
    transformed[valid] = 1.0 / (
        1.0
        + normalized[valid]
        + 0.35 * normalized[valid] ** 2
    )

    transformed = 0.5 * (transformed + transformed.T)
    transformed[~valid] = 0.0
    np.fill_diagonal(transformed, 0.0)

    # Symmetric Sinkhorn-style balancing. The scaling vector equalizes the
    # total incident conductance while preserving edge symmetry.
    scaling = np.ones(n, dtype=float)
    active = np.sum(transformed, axis=1) > 0.0

    for _ in range(96):
        incident = transformed @ scaling
        updated = np.ones(n, dtype=float)

        usable = active & np.isfinite(incident) & (incident > eps)
        updated[usable] = 1.0 / incident[usable]

        # Normalize the scale to avoid numerical drift.
        positive_scale = updated[active & (updated > 0.0)]
        if positive_scale.size:
            geometric_center = np.exp(
                np.mean(np.log(np.maximum(positive_scale, eps)))
            )
            updated[active] /= max(geometric_center, eps)

        if np.max(np.abs(updated - scaling)) < 1e-10:
            scaling = updated
            break

        scaling = updated

    balanced = transformed * scaling[:, None] * scaling[None, :]
    balanced[~valid] = 0.0
    balanced = np.maximum(balanced, 0.0)

    # Blend the balanced network with the original conductance. This keeps
    # distance information visible while retaining degree correction.
    scores = np.sqrt(
        np.maximum(transformed, 0.0)
        * np.maximum(balanced, 0.0)
    )

    incident_transformed = np.sum(transformed, axis=1)
    incident_balanced = np.sum(balanced, axis=1)

    load = np.ones(n, dtype=float)
    load[active] = np.sqrt(
        np.maximum(incident_transformed[active], eps)
        * np.maximum(incident_balanced[active], eps)
    )

    load_scale = float(np.median(load[active])) if np.any(active) else 1.0
    load_scale = max(load_scale, eps)

    endpoint_correction = np.sqrt(
        load_scale / np.maximum(load[:, None], eps)
        * load_scale / np.maximum(load[None, :], eps)
    )
    endpoint_correction = np.clip(endpoint_correction, 0.65, 1.35)

    scores *= endpoint_correction
    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)

    np.fill_diagonal(scores, 0.0)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
