import math
from typing import List

def non_dominated_sort(scores: List[List[float]], maximize: List[bool]) -> List[List[int]]:
    """
    Returns indices of solutions grouped by non-domination level (fronts).
    The first front is the set of non-dominated solutions.
    """
    fronts: List[List[int]] = [[]]
    domination_rank = [0] * len(scores)
    dominated_scores = [[] for _ in scores]

    for i, score_i in enumerate(scores):
        for j, score_j in enumerate(scores):
            if i == j:
                continue
            if dominates(score_i, score_j, maximize):
                dominated_scores[i].append(j)
            elif dominates(score_j, score_i, maximize):
                domination_rank[i] += 1

        # If no one dominates i, it's in the first front
        if domination_rank[i] == 0:
            fronts[0].append(i)

    # Build subsequent fronts
    front_rank = 0
    while len(fronts[front_rank]) > 0:
        next_front = []
        for i in fronts[front_rank]:
            # For each solution that i dominates:
            for d_j in dominated_scores[i]:
                domination_rank[d_j] -= 1
                if domination_rank[d_j] == 0:
                    next_front.append(d_j)
        front_rank += 1
        fronts.append(next_front)

    # The last front is always empty, so exclude it
    return fronts[:-1]

def dominates(x, y, maximize) -> bool:
    """
    Returns True if x dominates y under the given maximize flags.
    Each dimension's comparison flips based on maximize[m].
    """
    assert len(x) == len(y) == len(maximize), "Mismatch between length of scores and 'maximize' array."

    not_worst = all(
        (x_i >= y_i if m_i else x_i <= y_i)
        for x_i, y_i, m_i in zip(x, y, maximize)
    )
    better = any(
        (x_i > y_i if m_i else x_i < y_i)
        for x_i, y_i, m_i in zip(x, y, maximize)
    )
    return not_worst and better

def crowding_distance_with_maximize(values: List[List[float]], maximize: List[bool]) -> List[float]:
    """
    Computes the crowding distance for each solution in `values`, 
    respecting the 'maximize' array by flipping minimized dimensions first.
    Returns a list of distances in the same index order.
    """
    # Step 1) Scale everything so that bigger=better on [0..1].
    scaled_vals = feature_scaling_with_maximize(values, maximize)

    n = len(scaled_vals)
    if n == 0:
        return []
    if n == 1:
        return [float('inf')]

    # Initialize distances
    distances = [0.0 for _ in range(n)]
    dim = len(scaled_vals[0])

    for m in range(dim):
        # Sort solutions by dimension m
        sorted_idx = sorted(
            range(n), 
            key=lambda i: scaled_vals[i][m] if scaled_vals[i][m] != -math.inf else float('-inf')
        )

        # Mark boundary points as infinite distance
        distances[sorted_idx[0]] = float('inf')
        distances[sorted_idx[-1]] = float('inf')

        # Accumulate distance for interior points
        for i in range(1, n - 1):
            idx = sorted_idx[i]
            if (distances[idx] != float('inf')):
                prev_ = scaled_vals[sorted_idx[i-1]][m]
                next_ = scaled_vals[sorted_idx[i+1]][m]

                if prev_ == -math.inf or next_ == -math.inf:
                    # If either neighbor is -inf, skip
                    continue

                distances[idx] += (next_ - prev_)

    return distances

def crowding_distance(
    scores: List[List[float]], front: List[int], maximize: List[bool], index: int
) -> float:
    scaled_scores = feature_scaling(scores)
    crowding_distances: List[float] = [0 for _ in scores]
    for m in range(len(maximize)):
        front = sorted(front, key=lambda x: scores[x][m])
        crowding_distances[front[0]] = math.inf
        crowding_distances[front[-1]] = math.inf
        m_values = [scaled_scores[i][m] for i in front]
        scale: float = max(m_values) - min(m_values)
        if scale == 0:
            scale = 1
        for i in range(1, len(front) - 1):
            crowding_distances[i] += (
                scaled_scores[front[i + 1]][m] - scaled_scores[front[i - 1]][m]
            ) / scale
    return crowding_distances[index]
    
def feature_scaling(solutions_scores: List[List[float]]) -> List[List[float]]:
    total_metrics = len(solutions_scores[0])
    scaled_scores = [list() for _ in solutions_scores]

    metric_selector = 0
    while metric_selector < total_metrics:
        # All scores per solution
        # sol1: [1, 2]
        # sol2: [3, 4]
        # m_score[0] -> [1, 3]
        # m_score[1] -> [3, 4]
        m_scores = [score[metric_selector] for score in solutions_scores]
        if len(m_scores) == 1:
            for scaled in scaled_scores:
                scaled.append(1)
            metric_selector += 1
            continue

        filtered_m_scores = [v for v in m_scores if v != -math.inf]
        if len(filtered_m_scores) == 0:
            for scaled in scaled_scores:
                scaled.append(-math.inf)
            metric_selector += 1
            continue

        max_value = max(filtered_m_scores)
        min_value = min(filtered_m_scores)
        diff = max_value - min_value

        # When there is just one valid solution (everyone else is minus infinity)
        if diff == 0:
            index = m_scores.index(max_value)
            for i, scaled in enumerate(scaled_scores):
                if i == index or m_scores[i] != -math.inf:
                    scaled.append(1)
                else:
                    scaled.append(-math.inf)
            metric_selector += 1
            continue

        for i, scaled in enumerate(scaled_scores):
            scaled_value = (
                m_scores[i] - min_value
            ) / diff  # if m_scores[i] != -math.inf else 0
            scaled.append(scaled_value)
        metric_selector += 1

    return scaled_scores

def feature_scaling_with_maximize(solutions_scores: List[List[float]], maximize: List[bool]) -> List[List[float]]:
    """
    Perform min-max scaling on each dimension, flipping dimensions that should be minimized.
    That way, in the scaled space, "larger is always better".
    """
    if not solutions_scores:
        return []

    n = len(solutions_scores)
    d = len(solutions_scores[0])

    # 1) Possibly invert dimensions that are to be minimized
    #    so that bigger→better for them too.
    adjusted_scores = []
    for s in solutions_scores:
        s_adj = []
        for val, m in zip(s, maximize):
            if m:
                s_adj.append(val)      # maximize => keep as is
            else:
                s_adj.append(-val)     # minimize => flip sign
        adjusted_scores.append(s_adj)

    # 2) Now do min–max scaling dimension by dimension
    scaled_scores = [[0.0] * d for _ in range(n)]

    for m in range(d):
        col = [row[m] for row in adjusted_scores]
        # Filter out -inf from the dimension
        valid = [v for v in col if v != -math.inf]
        if not valid:
            # If all are -inf, all remain -inf or 0.0?
            for i in range(n):
                scaled_scores[i][m] = -math.inf
            continue

        mx = max(valid)
        mn = min(valid)
        diff = mx - mn
        if diff == 0:
            # All valid solutions are the same => scale to 1 if not -inf
            for i in range(n):
                scaled_scores[i][m] = 1.0 if col[i] != -math.inf else -math.inf
        else:
            for i in range(n):
                if col[i] == -math.inf:
                    scaled_scores[i][m] = -math.inf
                else:
                    scaled_scores[i][m] = (col[i] - mn) / diff

    return scaled_scores