import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized directed edge-priority indicators for TSP."""
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

    eps = np.finfo(float).eps
    work = np.where(valid, distances, np.inf)

    def local_statistics(matrix: np.ndarray):
        ordered = np.sort(matrix, axis=1)
        k = min(3, n - 1)
        neighbors = ordered[:, :k]
        finite_neighbors = np.isfinite(neighbors)

        fallback = np.ones(n, dtype=float)
        sums = np.sum(
            np.where(finite_neighbors, neighbors, 0.0),
            axis=1,
        )
        counts = finite_neighbors.sum(axis=1)

        scale = np.divide(
            sums,
            np.maximum(counts, 1),
            out=fallback.copy(),
            where=counts > 0,
        )
        scale = np.maximum(scale, eps)

        best = ordered[:, 0]
        second = ordered[:, 1] if n > 2 else best.copy()

        safe_best = np.where(np.isfinite(best), best, scale)
        safe_second = np.where(
            np.isfinite(second),
            second,
            2.0 * np.maximum(safe_best, eps),
        )

        regret = np.divide(
            safe_second - safe_best,
            np.maximum(safe_second, eps),
            out=np.zeros(n, dtype=float),
            where=np.isfinite(safe_second),
        )
        regret = np.clip(regret, 0.0, 1.0)

        competition = np.sum(
            valid & (matrix <= safe_second[:, None]),
            axis=1,
            dtype=float,
        )

        return scale, regret, competition

    out_scale, out_regret, out_competition = local_statistics(work)
    in_scale, in_regret, in_competition = local_statistics(work.T)

    pair_scale = np.sqrt(
        np.maximum(out_scale[:, None], eps)
        * np.maximum(in_scale[None, :], eps)
    )

    relative_distance = np.divide(
        distances,
        pair_scale,
        out=np.full_like(distances, np.inf),
        where=valid,
    )

    distance_quality = np.exp(
        -np.clip(0.95 * relative_distance, 0.0, 60.0)
    )

    out_ratio = np.divide(
        distances,
        np.maximum(out_scale[:, None], eps),
        out=np.full_like(distances, np.inf),
        where=valid,
    )
    in_ratio = np.divide(
        distances,
        np.maximum(in_scale[None, :], eps),
        out=np.full_like(distances, np.inf),
        where=valid,
    )

    endpoint_quality = np.sqrt(
        np.exp(-np.clip(out_ratio, 0.0, 60.0))
        * np.exp(-np.clip(in_ratio, 0.0, 60.0))
    )

    row_order = np.argsort(work, axis=1, kind="stable")
    row_rank = np.full((n, n), np.inf, dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)[None, :]

    col_order = np.argsort(work, axis=0, kind="stable")
    col_rank = np.full((n, n), np.inf, dtype=float)
    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    rank_scale = max(1.0, 0.9 * np.sqrt(float(n)))
    rank_quality = np.exp(
        -0.5 * (row_rank + col_rank) / rank_scale
    )
    rank_quality = np.nan_to_num(
        rank_quality,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    scarcity = 1.0 + 0.12 / np.sqrt(
        np.maximum(out_competition[:, None], 1.0)
        * np.maximum(in_competition[None, :], 1.0)
    )

    urgency = 1.0 + 0.16 * (
        out_regret[:, None] + in_regret[None, :]
    )

    long_edge_penalty = np.exp(
        -0.42 * np.maximum(relative_distance - 1.0, 0.0)
    )

    preference = (
        0.48
        + 0.27 * endpoint_quality
        + 0.25 * rank_quality
    )

    scores = (
        distance_quality
        * preference
        * scarcity
        * urgency
        * long_edge_penalty
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    row_totals = scores.sum(axis=1, keepdims=True)

    return np.divide(
        scores,
        row_totals,
        out=np.zeros_like(scores),
        where=row_totals > 0.0,
    )
