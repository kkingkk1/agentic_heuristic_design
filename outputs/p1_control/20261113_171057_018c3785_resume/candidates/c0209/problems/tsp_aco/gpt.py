import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return geometry- and bypass-aware stochastic priorities for TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    feasible = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(feasible, False)

    # Convert the input into an undirected working geometry.
    directed = np.where(feasible, d, np.inf)
    work = np.minimum(directed, directed.T)
    valid = np.isfinite(work)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    finite = work[valid]
    positive = finite[finite > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe = np.where(valid, work, 0.0)

    # Classical multidimensional scaling from the distance geometry.
    squared = np.where(valid, work * work, 0.0)
    centering = np.eye(n) - np.ones((n, n), dtype=float) / float(n)
    gram = -0.5 * centering @ squared @ centering
    gram = 0.5 * (gram + gram.T)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        positive_eigenvalues = np.maximum(eigenvalues[:2], 0.0)
        embedding = eigenvectors[:, :2] * np.sqrt(positive_eigenvalues)[None, :]
    except np.linalg.LinAlgError:
        embedding = np.zeros((n, 2), dtype=float)

    if embedding.shape[1] == 1:
        embedding = np.pad(embedding, ((0, 0), (0, 1)))

    centered_embedding = embedding - embedding.mean(axis=0, keepdims=True)
    radii = np.linalg.norm(centered_embedding, axis=1)

    if np.max(radii) > np.finfo(float).eps:
        angles = np.arctan2(
            centered_embedding[:, 1],
            centered_embedding[:, 0],
        )
        cyclic_order = np.argsort(angles, kind="stable")
        position = np.empty(n, dtype=float)
        position[cyclic_order] = np.arange(n, dtype=float)

        circular_gap = np.abs(position[:, None] - position[None, :])
        circular_gap = np.minimum(circular_gap, n - circular_gap)

        gap_scale = max(1.0, np.sqrt(float(n)))
        cyclic_score = np.exp(-circular_gap / gap_scale)
    else:
        cyclic_score = np.zeros((n, n), dtype=float)

    # Edge indispensability: direct edges with expensive third-node bypasses
    # are useful connectors in a Hamiltonian cycle.
    bypass = safe[:, :, None] + safe[None, :, :]
    edge_lengths = safe[:, :, None]
    detour_margin = bypass - edge_lengths.transpose(0, 2, 1)

    # Recompute with the intended (i, k, j) indexing.
    detour_margin = (
        safe[:, None, :] + safe[None, :, :]
        - safe[:, :, None].transpose(0, 2, 1)
    )
    third_valid = valid[:, None, :] & valid[None, :, :]
    np.fill_diagonal(third_valid.reshape(n * n, n), False)

    normalized_margin = np.maximum(detour_margin, 0.0) / scale
    normalized_margin[~third_valid] = 0.0

    bypass_strength = np.mean(
        1.0 - np.exp(-np.minimum(normalized_margin, 30.0)),
        axis=2,
    )

    # Preserve a direct geometric preference without using rank or Gibbs
    # normalization. This term is deliberately broad.
    direct_preference = 1.0 / (1.0 + work / scale)
    direct_preference[~valid] = 0.0

    # A second embedding-axis consistency term rewards edges that are short
    # in the recovered global geometry, even when their raw distance scale is
    # atypical.
    embedding_difference = (
        centered_embedding[:, None, :] - centered_embedding[None, :, :]
    )
    embedded_length = np.linalg.norm(embedding_difference, axis=2)
    embedded_scale = embedded_length[valid]
    embedded_scale = (
        float(np.median(embedded_scale))
        if embedded_scale.size
        else 1.0
    )
    embedded_scale = max(embedded_scale, np.finfo(float).eps)
    embedding_consistency = 1.0 / (
        1.0 + embedded_length / embedded_scale
    )
    embedding_consistency[~valid] = 0.0

    score = (
        0.22 * direct_preference
        + 0.30 * cyclic_score
        + 0.33 * bypass_strength
        + 0.15 * embedding_consistency
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
