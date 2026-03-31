from __future__ import annotations

from typing import Dict

import numpy as np



def c_index(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> float:
    comparable = 0
    concordant = 0
    ties = 0
    n = len(times)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if times[i] < times[j] and events[i] == 0:
                comparable += 1
                if risks[i] > risks[j]:
                    concordant += 1
                elif risks[i] == risks[j]:
                    ties += 1
    if comparable == 0:
        return 0.0
    return float((concordant + 0.5 * ties) / comparable)



def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    total = 0.0
    count = 0
    for p in pos:
        for n in neg:
            count += 1
            if p > n:
                total += 1.0
            elif p == n:
                total += 0.5
    return total / count if count > 0 else np.nan



def approximate_time_dependent_auc(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> float:
    unique_times = np.unique(times[events == 0])
    aucs = []
    for t in unique_times:
        positives = (times <= t) & (events == 0)
        negatives = times > t
        mask = positives | negatives
        labels = positives[mask].astype(np.int64)
        auc = binary_auc(risks[mask], labels)
        if not np.isnan(auc):
            aucs.append(auc)
    if not aucs:
        return 0.0
    return float(np.mean(aucs))



def summarize_metrics(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> Dict[str, float]:
    return {
        'c_index': c_index(times, events, risks),
        't_auc': approximate_time_dependent_auc(times, events, risks),
    }
