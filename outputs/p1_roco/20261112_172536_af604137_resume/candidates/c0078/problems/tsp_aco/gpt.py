import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed edge desirability indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    eps = np.finfo(float).eps
    valid = np.isfinite(distances) & (distances > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    positive = distances[valid]
    global_scale = float(np.median(positive))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    safe_distances = np.where(valid, distances, np.nan)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out_scale = np.nanmedian(safe_distances, axis=1)
        in_scale = np.nanmedian(safe_distances, axis=0)

        out_scale = np.where(
            np.isfinite(out_scale) & (out_scale > 0.0),
            out_scale,
            global_scale,
        )
        in_scale = np.where(
            np.isfinite(in_scale) & (in_scale > 0.0),
            in_scale,
            global_scale,
        )

        relative_out = distances / np.maximum(out_scale[:, None], eps)
        relative_in = distances / np.maximum(in_scale[None, :], eps)

        base = np.where(
            valid,
            1.0 / np.sqrt(
                np.maximum(relative_out, eps)
                * np.maximum(relative_in, eps)
            ),
            0.0,
        )

        safe_for_min = np.where(valid, distances, np.inf)

        out_best = np.min(safe_for_min, axis=1)
        in_best = np.min(safe_for_min, axis=0)

        out_second = np.partition(safe_for_min, 1, axis=1)[:, 1]
        in_second = np.partition(safe_for_min, 1, axis=0)[1, :]

        out_count = np.sum(valid, axis=1)
        in_count = np.sum(valid, axis=0)

        out_gap = np.where(
            (out_count >= 2) & np.isfinite(out_second),
            np.clip(
                (out_second - out_best)
                / np.maximum(out_second, eps),
                0.0,
                1.0,
            ),
            0.0,
        )
        in_gap = np.where(
            (in_count >= 2) & np.isfinite(in_second),
            np.clip(
                (in_second - in_best)
                / np.maximum(in_second, eps),
                0.0,
                1.0,
            ),
            0.0,
        )

        out_regret = np.where(
            valid & (out_count[:, None] >= 2),
            np.clip(
                (out_second[:, None] - distances)
                / np.maximum(out_second[:, None], eps),
                0.0,
                1.0,
            ),
            0.0,
        )
        in_regret = np.where(
            valid & (in_count[None, :] >= 2),
            np.clip(
                (in_second[None, :] - distances)
                / np.maximum(in_second[None, :], eps),
                0.0,
                1.0,
            ),
            0.0,
        )

        gap_factor = (
            1.0
            + 0.04
            * np.sqrt(out_gap[:, None] * in_gap[None, :])
            * np.sqrt(out_regret * in_regret)
        )

        score = np.where(valid, base * gap_factor, 0.0)

        row_mass = np.sum(score, axis=1, keepdims=True)
        row_prob = np.divide(
            score,
            np.maximum(row_mass, eps),
            out=np.zeros_like(score),
            where=row_mass > 0.0,
        )
        target_load = np.sum(row_prob, axis=0)

        positive_load = target_load[target_load > 0.0]
        load_scale = (
            max(float(np.median(positive_load)), eps)
            if positive_load.size
            else 1.0
        )
        load_ratio = np.clip(target_load / load_scale, 0.5, 2.0)
        diversity_factor = np.ones(n, dtype=float)
        occupied = target_load > 0.0
        diversity_factor[occupied] = np.exp(
            -0.08 * np.clip(np.log(load_ratio[occupied]), -0.7, 0.7)
        )

        continuation = np.ones(n, dtype=float)
        for node in range(n):
            choices = score[node, valid[node]]
            if choices.size == 0:
                continuation[node] = 0.8
            elif choices.size == 1:
                continuation[node] = 0.94 * max(float(choices[0]), eps)
            else:
                top_two = np.partition(choices, -2)[-2:]
                continuation[node] = np.sqrt(
                    max(float(top_two[0] * top_two[1]), eps)
                )

        continuation_positive = continuation[continuation > 0.0]
        continuation_scale = max(
            float(np.median(continuation_positive)),
            eps,
        )
        continuation_ratio = np.clip(
            continuation / continuation_scale,
            0.7,
            1.5,
        )
        future_factor = np.exp(
            0.08 * np.clip(np.log(continuation_ratio), -0.4, 0.4)
        )

        score *= diversity_factor[None, :]
        score *= future_factor[None, :]

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    row_max = np.max(score, axis=1, keepdims=True)
    row_max = np.where(row_max > 0.0, row_max, 1.0)
    scores = score / row_max

    np.fill_diagonal(scores, 0.0)
    return scores.astype(float, copy=False)
