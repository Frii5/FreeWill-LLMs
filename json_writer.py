from pathlib import Path
import json
import pickle


def load_results_pkl(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def extract_rankings(results):
    # Assumes each result has .ranking and that the keys are already ordered best -> worst
    return [list(result.ranking.keys()) for result in results]


input_dir = Path("results_new_prompt")
output_dir = Path("rankings_json")
output_dir.mkdir(exist_ok=True)

for path in input_dir.glob("*.pkl"):
    results = load_results_pkl(path)
    rankings = extract_rankings(results)

    out_path = output_dir / f"{path.stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rankings, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path}")