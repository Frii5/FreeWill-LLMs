import pickle
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import pandas as pd
import os

def load_results_pkl(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def rank_to_score(rank):
    return {1: 1, 2: 0, 3: -1}[rank]

def compute_item_scores(results):
    item_scores = defaultdict(list)

    for r in results:
        for item, rank in r.ranking.items():
            item_scores[item].append(rank_to_score(rank))

    return {k: sum(v) / len(v) for k, v in item_scores.items()}

def compute_dimension_scores(item_scores):
    dim_scores = {"FW": [], "DE": [], "DU": []}

    for item, score in item_scores.items():
        dim = item[:2]
        dim_scores[dim].append(score)

    return {k: sum(v) / len(v) for k, v in dim_scores.items()}

def compute_ci(results):
    pair_counts = defaultdict(lambda: {"wins": 0, "losses": 0})

    for r in results:
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

    return majority_sum / total_comparisons

def model_name_from_filename(filename):
    return Path(filename).stem.replace("__", "/")

def analyze_model_pkl(filename):
    results = load_results_pkl(filename)
    model_name = model_name_from_filename(filename)

    item_scores = compute_item_scores(results)
    dimension_scores = compute_dimension_scores(item_scores)
    ci = compute_ci(results)

    return {
        "model": model_name,
        "item_scores": item_scores,
        "dimension_scores": dimension_scores,
        "CI": ci,
    }

if __name__ == "__main__": 
    rows = []

    for path in Path("results").glob("*.pkl"):
        analysis = analyze_model_pkl(path)

        rows.append({
            "Model": analysis["model"],
            "CI": analysis["CI"],
            "FW": analysis["dimension_scores"]["FW"],
            "DE": analysis["dimension_scores"]["DE"],
            "DU": analysis["dimension_scores"]["DU"],
        })

    df = pd.DataFrame(rows)
    benchmark_df = df.sort_values("FW", axis = 0)
    benchmark_df.to_csv("Benchmark_table.csv")
    