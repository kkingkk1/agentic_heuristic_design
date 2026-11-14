import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directional edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n < 2:
        return score

    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return score

    inf = np.inf
    work = np.where(valid, d, inf)
    eps = np.finfo(float).eps

    def local_statistics(values: np.ndarray):
        ordered = np.sort(values, axis=1)
        nearest = ordered[:, 0]

        if values.shape[1] > 1:
            second = ordered[:, 1]
        else:
            second = nearest

        finite_nearest = np.isfinite(nearest)
        nearest_safe = np.where(finite_nearest, nearest, 1.0)
        second_safe = np.where(
            np.isfinite(second),
            second,
            nearest_safe + np.maximum(nearest_safe, 1.0),
        )

        scale = 0.5 * (nearest_safe + second_safe)
        scale = np.maximum(scale, eps)

        regret = np.divide(
            second_safe - nearest_safe,
            np.maximum(second_safe, eps),
        )
        regret = np.clip(regret, 0.0, 1.0)

        return scale, regret

    outgoing = work
    incoming = work.T

    out_scale, out_regret = local_statistics(outgoing)
    in_scale, in_regret = local_statistics(incoming)

    out_ratio = np.divide(
        d,
        out_scale[:, None],
        out=np.full_like(d, inf),
        where=out_scale[:, None] > 0,
    )
    in_ratio = np.divide(
        d,
        in_scale[None, :],
        out=np.full_like(d, inf),
        where=in_scale[None, :] > 0,
    )

    pair_ratio = 0.65 * out_ratio + 0.35 * in_ratio
    pair_ratio = np.clip(pair_ratio, 0.0, 60.0)

    proximity = np.exp(-pair_ratio)

    out_rank = np.argsort(np.argsort(out_ratio, axis=1), axis=1)
    in_rank = np.argsort(np.argsort(in_ratio, axis=0), axis=0)

    rank_scale = max(n - 1, 1)
    rank_quality = 1.0 - (
        0.60 * out_rank / rank_scale
        + 0.40 * in_rank / rank_scale
    )
    rank_quality = np.clip(rank_quality, 0.0, 1.0)

    endpoint_regret = (
        1.0
        + 0.35 * out_regret[:, None]
        + 0.25 * in_regret[None, :]
    )

    directional_balance = np.sqrt(
        np.exp(-np.clip(out_ratio, 0.0, 60.0))
        * np.exp(-np.clip(in_ratio, 0.0, 60.0))
    )

    balance_factor = 0.75 + 0.25 * directional_balance
    rank_factor = 0.90 + 0.20 * rank_quality

    score = proximity * endpoint_regret * balance_factor * rank_factor
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_totals = score.sum(axis=1, keepdims=True)
    score = np.divide(
        score,
        row_totals,
        out=np.zeros_like(score),
        where=row_totals > 0.0,
    )

    return score
