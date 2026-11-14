import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust directed edge desirability indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    valid = np.isfinite(distances) & (distances > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    positive = distances[valid]
    global_scale = float(np.median(positive))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    safe = np.where(valid, distances, np.nan)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out_scale = np.nanmedian(safe, axis=1)
        in_scale = np.nanmedian(safe, axis=0)

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

        relative_out = np.where(valid, np.maximum(relative_out, eps), np.inf)
        relative_in = np.where(valid, np.maximum(relative_in, eps), np.inf)

        base = 1.0 / np.sqrt(relative_out * relative_in)

        safe_distances = np.where(valid, distances, np.inf)

        out_count = np.sum(valid, axis=1)
        in_count = np.sum(valid, axis=0)

        out_best = np.min(safe_distances, axis=1)
        in_best = np.min(safe_distances, axis=0)

        if n >= 2:
            out_second = np.partition(safe_distances, 1, axis=1)[:, 1]
            in_second = np.partition(safe_distances, 1, axis=0)[1, :]
        else:
            out_second = np.full(n, np.inf, dtype=float)
            in_second = np.full(n, np.inf, dtype=float)

        out_advantage = np.zeros(n, dtype=float)
        in_advantage = np.zeros(n, dtype=float)

        out_mask = (out_count >= 2) & np.isfinite(out_second)
        in_mask = (in_count >= 2) & np.isfinite(in_second)

        out_advantage[out_mask] = np.clip(
            (out_second[out_mask] - out_best[out_mask])
            / np.maximum(out_second[out_mask], eps),
            0.0,
            1.0,
        )
        in_advantage[in_mask] = np.clip(
            (in_second[in_mask] - in_best[in_mask])
            / np.maximum(in_second[in_mask], eps),
            0.0,
            1.0,
        )

        out_edge_advantage = np.zeros_like(distances)
        in_edge_advantage = np.zeros_like(distances)

        out_edge_advantage[valid] = np.clip(
            (
                out_second[:, None][valid] - distances[valid]
            )
            / np.maximum(out_second[:, None][valid], eps),
            0.0,
            1.0,
        )
        in_edge_advantage[valid] = np.clip(
            (
                in_second[None, :][valid] - distances[valid]
            )
            / np.maximum(in_second[None, :][valid], eps),
            0.0,
            1.0,
        )

        confidence = (
            np.sqrt(out_advantage[:, None] * in_advantage[None, :])
            * np.sqrt(out_edge_advantage * in_edge_advantage)
        )

        continuation = np.ones(n, dtype=float)
        continuation_mask = out_mask

        continuation[continuation_mask] = np.sqrt(
            np.maximum(
                1.0 / np.maximum(out_best[continuation_mask], eps),
                eps,
            )
            * np.maximum(
                1.0 / np.maximum(out_second[continuation_mask], eps),
                eps,
            )
        )

        positive_continuation = continuation[continuation_mask]
        if positive_continuation.size:
            continuation_scale = max(
                float(np.median(positive_continuation)),
                eps,
            )
            continuation_ratio = np.clip(
                continuation / continuation_scale,
                0.7,
                1.3,
            )
            continuation_factor = np.where(
                continuation_mask,
                np.exp(
                    0.04
                    * np.clip(
                        np.log(continuation_ratio),
                        -0.35,
                        0.35,
                    )
                ),
                1.0,
            )
        else:
            continuation_factor = np.ones(n, dtype=float)

        score = np.where(
            valid,
            base
            * continuation_factor[None, :]
            * (1.0 + 0.025 * confidence),
            0.0,
        )

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    row_max = np.where(row_max > 0.0, row_max, 1.0)
    score /= row_max

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
