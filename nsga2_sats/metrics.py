"""Two-objective minimization indicators and transparent cross-instance tests.

HV uses the stated reference point, IGD uses mean Euclidean distance, and
Spread is Deb's endpoint-aware diversity measure. No per-run rescaling is done.
"""
from itertools import combinations
import warnings
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


def as_front(values):
    points = np.asarray(values, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    points = np.atleast_2d(points)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Front must contain finite two-objective rows")
    return points


def nondominated(values):
    """Return sorted, unique nondominated points (both objectives minimized)."""
    points = np.unique(as_front(values), axis=0)
    if not len(points):
        return points
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    keep, best_y = [], float("inf")
    for point in points:
        if point[1] < best_y:
            keep.append(point)
            best_y = point[1]
    return np.asarray(keep, dtype=float).reshape(-1, 2)


def reference_front(fronts):
    arrays = [as_front(front) for front in fronts]
    return nondominated(np.vstack(arrays)) if arrays else np.empty((0, 2))


def hypervolume(front, reference=(1.1, 1.1)):
    """Exact 2-D union area dominated by front and bounded by reference."""
    ref = np.asarray(reference, dtype=float)
    if ref.shape != (2,) or not np.isfinite(ref).all():
        raise ValueError("HV reference must be two finite numbers")
    points = nondominated(front)
    points = points[np.all(points < ref, axis=1)]
    area, last_y = 0.0, ref[1]
    for x, y in points:
        area += (ref[0] - x) * (last_y - y)
        last_y = y
    return float(area)


def igd(front, reference):
    points, ref = as_front(front), as_front(reference)
    if not len(ref):
        raise ValueError("IGD needs a nonempty reference front")
    if not len(points):
        return float("inf")
    return float(cdist(ref, points).min(axis=1).mean())


def spread(front, reference):
    """Deb Spread with distances to the two reference-front extreme points.

    A front with fewer than two distinct points is assigned 1 by convention.
    A perfect uniformly spaced front spanning the reference extremes has 0.
    """
    points, ref = nondominated(front), nondominated(reference)
    if not len(ref):
        raise ValueError("Spread needs a nonempty reference front")
    if len(points) < 2:
        return 1.0
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    mean = float(distances.mean())
    d_first = float(np.linalg.norm(points[0] - ref[0]))
    d_last = float(np.linalg.norm(points[-1] - ref[-1]))
    denominator = d_first + d_last + len(distances) * mean
    if denominator == 0:
        return 0.0
    return float((d_first + d_last + np.abs(distances - mean).sum()) / denominator)


def summarize(records):
    """Aggregate run metric records with population SD (ddof=0)."""
    groups = {}
    for row in records:
        groups.setdefault((row["scenario"], row["algorithm"]), []).append(row)
    result = []
    for (scenario, algorithm), rows in sorted(groups.items()):
        summary = {"scenario": scenario, "algorithm": algorithm, "runs": len(rows)}
        for metric in ("HV", "IGD", "Spread"):
            values = np.array([r[metric] for r in rows], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {metric} for {scenario}/{algorithm}")
            summary[metric + "_mean"] = float(values.mean())
            summary[metric + "_std"] = float(values.std(ddof=0))
        result.append(summary)
    return result


def paired_tests(summaries):
    """Friedman and pairwise two-sided Wilcoxon on scenario means.

    Only shared scenarios are paired. Runs never become independent scenario
    replicates. Holm correction is applied to available pairwise p-values
    separately for each metric. Tests use the algorithms present in the supplied results.
    """
    algorithms = sorted({row["algorithm"] for row in summaries})
    table = {(r["scenario"], r["algorithm"]): r for r in summaries}
    result = {"pairing_unit": "scenario mean across available runs",
              "wilcoxon_alternative": "two-sided", "p_adjustment": "Holm within metric",
              "algorithms_present": algorithms, "metrics": {}}
    for metric in ("HV", "IGD", "Spread"):
        cases = {a: {s for s, name in table if name == a} for a in algorithms}
        shared = sorted(set.intersection(*cases.values())) if cases else []
        entry = {"complete_scenarios": shared, "friedman": None, "average_ranks": {}, "wilcoxon": []}
        if shared:
            data = np.array([[table[s, a][metric + "_mean"] for a in algorithms] for s in shared])
            ranks = np.array([rankdata(-r if metric == "HV" else r) for r in data])
            entry["average_ranks"] = dict(zip(algorithms, ranks.mean(axis=0).tolist()))
            if len(algorithms) >= 3 and len(shared) >= 3:
                if np.all(data == data[:, :1]):
                    entry["friedman"] = {"statistic": 0.0, "p_value": 1.0}
                else:
                    stat, p = friedmanchisquare(*data.T)
                    entry["friedman"] = {"statistic": float(stat), "p_value": float(p)}
        if entry["friedman"] is None:
            entry["friedman_unavailable_reason"] = "Requires at least three algorithms and three shared scenarios"
        for a, b in combinations(algorithms, 2):
            common = sorted(cases[a] & cases[b])
            pair = {"a": a, "b": b, "scenarios": common, "p_value": None}
            if len(common) >= 2:
                left = np.array([table[s, a][metric + "_mean"] for s in common])
                right = np.array([table[s, b][metric + "_mean"] for s in common])
                if np.array_equal(left, right):
                    stat, p = 0.0, 1.0
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        stat, p = wilcoxon(left, right, alternative="two-sided", method="auto")
                pair.update(statistic=float(stat), p_value=float(p))
            else:
                pair["unavailable_reason"] = "Requires at least two shared scenario means"
            entry["wilcoxon"].append(pair)
        available = sorted((p for p in entry["wilcoxon"] if p["p_value"] is not None), key=lambda p: p["p_value"])
        adjusted = 0.0
        for i, pair in enumerate(available):
            adjusted = max(adjusted, min(1.0, (len(available) - i) * pair["p_value"]))
            pair["p_value_holm"] = adjusted
        result["metrics"][metric] = entry
    return result
