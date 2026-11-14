import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return graph-structural edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    nonnegative = raw >= 0.0
    valid = finite & nonnegative & ~diagonal

    if not np.any(valid | valid.T):
        return np.zeros((n, n), dtype=float)

    directed_values = np.where(finite & nonnegative, raw, 0.0)
    symmetric = 0.5 * (directed_values + directed_values.T)
    symmetric_valid = (valid | valid.T) & ~diagonal

    symmetric[symmetric_valid & ~(valid & valid.T)] = np.where(
        valid[symmetric_valid & ~(valid & valid.T)],
        raw[symmetric_valid & ~(valid & valid.T)],
        raw.T[symmetric_valid & ~(valid & valid.T)],
    )
    symmetric[~symmetric_valid] = np.inf
    np.fill_diagonal(symmetric, np.inf)

    finite_edges = symmetric_valid & np.isfinite(symmetric)
    if not np.any(finite_edges):
        return np.zeros((n, n), dtype=float)

    positive = symmetric[finite_edges & (symmetric > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    # Rank-based kernel is less sensitive to the numerical scale of distances.
    ranks = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_valid = finite_edges[i]
        count = int(np.sum(row_valid))
        if count == 0:
            continue
        order = np.argsort(
            np.where(row_valid, symmetric[i], np.inf),
            kind="stable",
        )
        rank = 0
        for j in order:
            if row_valid[j]:
                ranks[i, j] = rank / max(count - 1, 1)
                rank += 1

    rank_distance = 0.5 * (ranks + ranks.T)
    temperature = 0.85
    conductance = np.exp(-rank_distance / temperature)
    conductance[~symmetric_valid] = 0.0
    np.fill_diagonal(conductance, 0.0)

    degree = conductance.sum(axis=1)
    if not np.any(degree > 0.0):
        return np.zeros((n, n), dtype=float)

    laplacian = -conductance.copy()
    np.fill_diagonal(laplacian, degree)

    # Effective resistance gives the inclusion probability of an edge in a
    # weighted random spanning tree, up to multiplication by its conductance.
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    cutoff = max(float(np.max(eigenvalues)), 1.0) * 1e-11
    inverse_eigenvalues = np.where(
        eigenvalues > cutoff,
        1.0 / eigenvalues,
        0.0,
    )
    laplacian_pinv = (
        eigenvectors
        @ (inverse_eigenvalues[:, None] * eigenvectors.T)
    )

    diagonal_pinv = np.diag(laplacian_pinv)
    resistance = (
        diagonal_pinv[:, None]
        + diagonal_pinv[None, :]
        - 2.0 * laplacian_pinv
    )
    resistance = np.maximum(resistance, 0.0)

    tree_inclusion = conductance * resistance
    tree_inclusion = 0.5 * (tree_inclusion + tree_inclusion.T)

    conductance_max = float(np.max(conductance))
    if conductance_max > 0.0:
        local_signal = conductance / conductance_max
    else:
        local_signal = np.zeros_like(conductance)

    inclusion_max = float(np.max(tree_inclusion))
    if inclusion_max > 0.0:
        structural_signal = tree_inclusion / inclusion_max
    else:
        structural_signal = np.zeros_like(tree_inclusion)

    score = 0.78 * structural_signal + 0.22 * local_signal
    score[symmetric_valid] += 0.015
    score[~symmetric_valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
