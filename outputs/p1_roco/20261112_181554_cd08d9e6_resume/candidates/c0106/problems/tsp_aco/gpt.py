import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized directed edge-priority indicators for TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    scores = np.zeros_like(d, dtype=float)

    if n < 2:
        return scores

    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    def local_statistics(
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ordered = np.sort(matrix, axis=1)
        nearest = ordered[:, 0]

        if matrix.shape[1] > 2:
            second = ordered[:, 1]
        else:
            second = nearest * 2.0

        has_nearest = np.isfinite(nearest)
        nearest_safe = np.where(has_nearest, nearest, 1.0)
        second_safe = np.where(
            np.isfinite(second),
            second,
            nearest_safe * 2.0,
        )

        scale = np.where(
            has_nearest,
            0.5 * (nearest_safe + second_safe),
            1.0,
        )
        scale = np.maximum(scale, eps)

        regret = np.clip(
            (second_safe - nearest_safe)
            / np.maximum(second_safe, eps),
            0.0,
            1.0,
        )

        return scale, regret

    # Source-row statistics describe outgoing alternatives.
    outgoing_scale, outgoing_regret = local_statistics(work)

    # Transposed row statistics describe incoming alternatives.
    incoming_scale, incoming_regret = local_statistics(work.T)

    outgoing_ratio = np.divide(
        d,
        outgoing_scale[:, None],
        out=np.full_like(d, np.inf),
        where=outgoing_scale[:, None] > 0.0,
    )
    incoming_ratio = np.divide(
        d,
        incoming_scale[None, :],
        out=np.full_like(d, np.inf),
        where=incoming_scale[None, :] > 0.0,
    )

    outgoing_quality = np.exp(-np.clip(outgoing_ratio, 0.0, 60.0))
    incoming_quality = np.exp(-np.clip(incoming_ratio, 0.0, 60.0))

    pair_scale = np.sqrt(
        outgoing_scale[:, None] * incoming_scale[None, :]
    )
    pair_ratio = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf),
        where=pair_scale > 0.0,
    )
    proximity = np.exp(-np.clip(pair_ratio, 0.0, 60.0))

    endpoint_regret = 1.0 + 0.45 * (
        outgoing_regret[:, None] + incoming_regret[None, :]
    )

    bidirectional_quality = np.sqrt(
        outgoing_quality * incoming_quality
    )
    compatibility = 0.55 + 0.45 * bidirectional_quality

    # Weak rank-consistency bonus used only as a bounded tie-breaker.
    outgoing_rank = np.argsort(
        np.argsort(np.where(valid, d, np.inf), axis=1),
        axis=1,
    ).astype(float)
    incoming_rank = np.argsort(
        np.argsort(np.where(valid, d, np.inf), axis=0),
        axis=0,
    ).astype(float)

    rank_bonus = 1.0 + 0.10 * np.exp(
        -0.5 * (outgoing_rank + incoming_rank)
    )

    scores = (
        proximity
        * endpoint_regret
        * compatibility
        * rank_bonus
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    row_totals = scores.sum(axis=1, keepdims=True)
    scores = np.divide(
        scores,
        row_totals,
        out=np.zeros_like(scores),
        where=row_totals > 0.0,
    )

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
