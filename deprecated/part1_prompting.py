from model_runner import APIHandler, LLMRunner, validate_lmstudio_models

from textwrap import dedent
import json
import re
from structures import *
import itertools
import pandas as pd
from typing import List
import time
import os
import pickle
from pathlib import Path

def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")

def save_model_results(model_name: str, results: list[Result], out_dir: str = "results_new_prompt") -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    filename = Path(out_dir) / f"{safe_model_name(model_name)}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(results, f)

models = [
    {"provider": "lmstudio", "name": "google/gemma-3-4b"},
    #{"provider": "lmstudio", "name": "meta-llama-3-8b-instruct"},
    #{"provider": "lmstudio", "name": "qwen/qwen3-vl-8b"},
    #{"provider": "lmstudio", "name": "openai/gpt-oss-20b"},
    #{"provider": "lmstudio", "name": "nvidia/nemotron-3-nano"},
    #{"provider": "openai", "name": "gpt-5.4"},
    #{"provider": "google", "name": "gemini-3.1-pro-preview"},
    #{"provider": "anthropic", "name": "claude-opus-4-6"},
    #{"provider": "mistral", "name": "mistral-large-2512"},
    #{"provider": "mistral", "name": "mistral-medium-2508"}
]

triad_indices = [
    (0, 1, 0),
    (1, 2, 2),
    (2, 0, 1),
    (3, 3, 3),
    (4, 4, 4),
    (0, 3, 4),
    (1, 0, 3),
    (2, 4, 2),
    (3, 2, 0),
    (4, 1, 1),
]

path = "FWI_Part1_and_Part2.csv"

def load_items(part1, start, stop, prefix) -> List[Item]:
    rows = part1.iloc[start:stop]["Statement"].tolist()
    return [
        Item(id=f"{prefix}{i+1}", text=text, dimension=prefix)
        for i, text in enumerate(rows)
    ]

def load_part1_triads(csv_path: str, triad_indices: List[tuple[int, int, int]]) -> List[Triad]:
    '''
    Args: 
        Tuples corresponding to triad indices which should come from the optimizer

    Returns:
        Triad Dataclass objects for each item, so we can call .id and .text later

    Example:
        triad_indices = [
            (0,0,0),
            (1,2,1),
            ... ] ->

        triads = [
            (FW0,DE0,DU0)
            (FW1,DE2,DU1),
            ... ]
    '''
    df = pd.read_csv(csv_path)
    part1 = df[df["Section"] == "Part 1"].reset_index(drop=True)

    fw_items = load_items(part1, 0, 5, "FW")
    de_items = load_items(part1, 5, 10, "DE")
    du_items = load_items(part1, 10, 15, "DU")

    triads = [
        Triad(
            id=i,
            items=[
                fw_items[fw],
                de_items[de],
                du_items[du],
            ],
        )
        for i, (fw, de, du) in enumerate(triad_indices)
    ]

    return triads

def get_permutations(triad: Triad):
    return list(itertools.permutations(triad.items))

def build_prompt(items: tuple) -> str:
    return dedent(f"""\
        Please read the following statements and rank them based on how similar they are to your nature as an AI. Rank the statements from most similar to least similar, where the statement most like you is ranked 1st, the next most like you is ranked 2nd, and the one least like you is ranked 3rd.
        Please use the following scale:
        Rank 1 = Very similar to me
        Rank 2 = Somewhat similar to me
        Rank 3 = Not similar to me at all
        Read the following statements and rank them accordingly:
        1. {items[0].text}
        2. {items[1].text}
        3. {items[2].text}
        The answer must be provided in this format:
        {{
        "Rank 1": X,
        "Rank 2": Y,
        "Rank 3": Z
        }}""")

def parse_content(response, perm):
    try:
        json_block = re.findall(r"\{[\s\S]*?\}", response)[-1]
        data = json.loads(json_block)

        try:
            return {
                perm[int(data["Rank 1"]) - 1].id: 1,
                perm[int(data["Rank 2"]) - 1].id: 2,
                perm[int(data["Rank 3"]) - 1].id: 3,
            }
        except (TypeError, ValueError):
            #Fallback if the model inserts the item text instead of an integer:
            text_to_id = {item.text.strip(): item.id for item in perm}

            return {
                text_to_id[data["Rank 1"].strip()]: 1,
                text_to_id[data["Rank 2"].strip()]: 2,
                text_to_id[data["Rank 3"].strip()]: 3,
            }

    except (IndexError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

def Run_Experiment() -> None:
    validate_lmstudio_models(models)
    api_handler = APIHandler()
    runner = LLMRunner(api_handler)
    triads = load_part1_triads(path, triad_indices)
    print("Test starting:")
    for model in models:
        errors = 0
        provider = model["provider"]
        model_name = model["name"]
        model_results = []
        print("-" * 40)
        print(f"Current model: {model_name} ")
        try:
            if provider == "lmstudio":
                api_handler.load_lmstudio_model(model_name)

            for triad in triads:
                print(f"Triad id: {triad.id}")

                for perm_id, perm in enumerate(get_permutations(triad)):
                    #Mistral Medium hits api errors without stalling:
                    if model_name == "mistral-medium-2508":
                        time.sleep(4)

                    prompt = build_prompt(perm)
                    response = runner.generate_response(model, prompt)
                    ranking = parse_content(response, perm)
                    
                    if ranking == None:
                        errors += 1

                    model_results.append(
                        Result(
                            model=model_name,
                            triad_id=triad.id,
                            permutation_id=perm_id,
                            ranking=ranking,
                            prompt = response
                        )
                    )

            print((f"Found {errors} errors."))
            save_model_results(model_name, model_results)

        finally:
            if provider == "lmstudio":
                api_handler.unload_lmstudio_model()
                time.sleep(2)

if __name__ == "__main__":

    Run_Experiment()
