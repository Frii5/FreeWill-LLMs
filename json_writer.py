from pathlib import Path
import json
import pickle

def load_results_pkl(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def extract_rankings(results):
    return [list(result.ranking.keys()) for result in results]

def convert_folder(input_dir_name, output_dir_name, label):
    input_dir = Path(input_dir_name)
    output_dir = Path(output_dir_name)
    output_dir.mkdir(exist_ok=True)

    print(f"\nProcessing {label}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    for i, path in enumerate(input_dir.glob("*.pkl")):
        results = load_results_pkl(path)
        rankings = extract_rankings(results)
        out_path = output_dir / f"{path.stem}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rankings, f, ensure_ascii=False, indent=2)

        if i == 0:
            print(f"Example Output ({label}):")
            print(rankings)

        print(f"Wrote {out_path}")

# Part I: triads
convert_folder(
    input_dir_name="results_FC_part1",
    output_dir_name="rankings_json_part1",
    label="Part I"
)
# Part II: dyads
convert_folder(
    input_dir_name="results_FC_part2",
    output_dir_name="rankings_json_part2",
    label="Part II"
)