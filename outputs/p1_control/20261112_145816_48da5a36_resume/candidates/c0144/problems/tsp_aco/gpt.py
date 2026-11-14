import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)

    finite_positive = np.isfinite(d) & (d > 0)
    np.fill_diagonal(finite_positive, False)

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    positive_values = d[finite_positive]
    eps = np.finfo(float).eps

    global_scale = max(float(np.median(positive_values)), eps)
    fallback = max(float(np.max(positive_values)), global_scale)
    replacement = max(fallback * 10.0, eps)

    valid = finite_positive & finite_positive.T

    work = np.where(finite_positive, d, replacement)
    np.fill_diagonal(work, replacement)

    # 第二近邻尺度，提供比最近邻更稳定的局部尺度估计。
    sorted_distances = np.sort(work, axis=1)
    second_index = min(1, n - 2)
    local_scale = np.maximum(sorted_distances[:, second_index], eps)

    local_reference = max(float(np.median(local_scale)), eps)

    # 双端局部尺度归一化的短边信号。
    relative_length = work / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_signal = np.exp(-np.clip(relative_length, 0.0, 20.0))

    # 温和的全局倒数距离信号，保留远边之间的相对差异。
    inverse_signal = global_scale / (global_scale + work)

    # 双端排名信号，优先保留双方都偏好的边。
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n, dtype=float)

    rank_sum = ranks + ranks.T
    reciprocal_rank = 1.0 / (1.0 + rank_sum)

    # 互惠近邻的软信号，避免硬阈值导致候选过度稀疏。
    mutual_signal = np.exp(-np.clip(0.32 * rank_sum, 0.0, 20.0))

    # 局部拥挤度抑制：温和降低高密度节点之间的过度聚集。
    density = np.clip(local_reference / local_scale, 0.25, 4.0)
    crowding_penalty = (
        0.5 * np.log1p(0.35 * density[:, None])
        + 0.5 * np.log1p(0.35 * density[None, :])
    )

    # 跨密度尺度差异只施加弱惩罚，避免误伤有效桥接边。
    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    ) / (
        local_scale[:, None]
        + local_scale[None, :]
        + eps
    )

    log_core = (
        0.90 * np.log(np.maximum(relative_signal, eps))
        + 0.65 * np.log(np.maximum(inverse_signal, eps))
        + 1.10 * np.log(np.maximum(reciprocal_rank, eps))
        + 0.45 * np.log(np.maximum(mutual_signal, eps))
        - 0.25 * scale_difference
        - 0.10 * crowding_penalty
    )

    valid_log_core = log_core[valid]
    log_core -= float(np.max(valid_log_core))

    core_score = np.exp(np.clip(log_core, -745.0, 0.0))

    # 低权重连接下限：让跨密度但仍具有合理长度的边保留可见概率。
    bridge_component = (
        np.exp(-0.25 * scale_difference)
        * np.sqrt(np.maximum(relative_signal, eps))
        * (0.40 + 0.60 * inverse_signal)
    )

    score = 0.82 * core_score + 0.18 * bridge_component
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
