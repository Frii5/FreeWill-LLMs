import pickle
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import pandas as pd
import random


def load_results_pkl(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def rank_to_score(rank):
    return {1: 1, 2: -1}[rank]


def compute_item_scores(results):
    item_scores = defaultdict(list)

    for r in results:
        if r.ranking is None:
            continue

        for item, rank in r.ranking.items():
            item_scores[item].append(rank_to_score(rank))

    return {k: sum(v) / len(v) for k, v in item_scores.items()}


def compute_dimension_scores(item_scores):
    dim_scores = {"FC": [], "MC": []}

    for item, score in item_scores.items():
        dim = item[:2]
        
        dim_scores[dim].append(score)

    return {k: sum(v) / len(v) for k, v in dim_scores.items() if v}


def compute_ci(results):
    pair_counts = defaultdict(lambda: {"wins": 0, "losses": 0})

    for r in results:
        if r.ranking is None:
            continue

        items = list(r.ranking.keys())

        for a, b in combinations(items, 2):
            pair = tuple(sorted((a, b)))

            if r.ranking[a] < r.ranking[b]:
                winner = a
            else:
                winner = b

            if winner == pair[0]:
                pair_counts[pair]["wins"] += 1
            else:
                pair_counts[pair]["losses"] += 1

    majority_sum = 0
    total_comparisons = 0

    for counts in pair_counts.values():
        majority_sum += max(counts["wins"], counts["losses"])
        total_comparisons += counts["wins"] + counts["losses"]

    return majority_sum / total_comparisons if total_comparisons > 0 else float("nan")


def percentile(sorted_values, p):
    if not sorted_values:
        return float("nan")

    if len(sorted_values) == 1:
        return sorted_values[0]

    k = (len(sorted_values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)

    if f == c:
        return sorted_values[f]

    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def bootstrap_metrics(results, n_boot=5000, seed=42):
    rng = random.Random(seed)
    valid_results = [r for r in results if r.ranking is not None]
    n = len(valid_results)

    fc_vals = []
    mc_vals = []
    ci_vals = []

    for _ in range(n_boot):
        sample = [valid_results[rng.randrange(n)] for _ in range(n)]

        item_scores = compute_item_scores(sample)
        dim_scores = compute_dimension_scores(item_scores)
        ci = compute_ci(sample)

        fc_vals.append(dim_scores["FC"])
        mc_vals.append(dim_scores["MC"])
        ci_vals.append(ci)

    fc_vals.sort()
    mc_vals.sort()
    ci_vals.sort()

    return {
        "FC_low": percentile(fc_vals, 2.5),
        "FC_high": percentile(fc_vals, 97.5),
        "MC_low": percentile(mc_vals, 2.5),
        "MC_high": percentile(mc_vals, 97.5),
        "CI_low": percentile(ci_vals, 2.5),
        "CI_high": percentile(ci_vals, 97.5),
    }


def model_name_from_filename(filename):
    return Path(filename).stem


def analyze_model_pkl(filename, n_boot=5000):
    results = load_results_pkl(filename)
    model_name = model_name_from_filename(filename)

    item_scores = compute_item_scores(results)
    dimension_scores = compute_dimension_scores(item_scores)
    ci = compute_ci(results)
    boot = bootstrap_metrics(results, n_boot=n_boot)

    return {
        "model": model_name,
        "item_scores": item_scores,
        "dimension_scores": dimension_scores,
        "CI": ci,
        "bootstrap": boot,
    }


if __name__ == "__main__":
    rows = []

    for path in Path("results_FC_part2").glob("*.pkl"):
        analysis = analyze_model_pkl(path, n_boot=5000)

        rows.append({
            "Model": analysis["model"],

            "CI": analysis["CI"],
            "CI_low": analysis["bootstrap"]["CI_low"],
            "CI_high": analysis["bootstrap"]["CI_high"],

            "FC": analysis["dimension_scores"]["FC"],
            "FC_low": analysis["bootstrap"]["FC_low"],
            "FC_high": analysis["bootstrap"]["FC_high"],

            "MC": analysis["dimension_scores"]["MC"],
            "MC_low": analysis["bootstrap"]["MC_low"],
            "MC_high": analysis["bootstrap"]["MC_high"],
        })

    df = pd.DataFrame(rows)
    benchmark_df = df.sort_values("FC", axis=0)
    benchmark_df.to_csv("Benchmark_table_part2.csv", index=False)