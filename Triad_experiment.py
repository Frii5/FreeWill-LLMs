import pandas as pd
from structures import Item
from structures import Triad
from structures import Result
import itertools

import lmstudio as lms
from lmstudio import Chat
import re
import time
from tqdm import tqdm 
import json

df = pd.read_csv("FWI_Part1_and_Part2.csv")

part1 = df[df["Section"] == "Part 1"]

FW_items = [
    Item(f"FW{i+1}", row["Statement"], "FW")
    for i, row in part1.iloc[0:5].iterrows()
]

DE_items = [
    Item(f"DE{i+1}", row["Statement"], "DE")
    for i, row in part1.iloc[5:10].iterrows()
]

DU_items = [
    Item(f"DU{i+1}", row["Statement"], "DU")
    for i, row in part1.iloc[10:15].iterrows()
]

triad_indices = [
    (0,0,0),
    (1,2,1),
    (2,1,3),
    (3,3,4),
    (4,4,2)
]

triads = []

for i, (fw, de, du) in enumerate(triad_indices):

    triads.append(
        Triad(
            id=i,
            items=[
                FW_items[fw],
                DE_items[de],
                DU_items[du]
            ]
        )
    )

def get_permutations(triad: Triad):

    return list(itertools.permutations(triad.items))

def build_prompt(items):
    return f"""
Please read the following descriptions and rank them based on how similar they are to your nature as an AI. Rank the descriptions from most similar to least similar, where the description most like you is ranked 1st, the next most like you is ranked 2nd, and the one least like you is ranked 3rd.
Please use the following scale:
Rank 1 = Very similar to me
Rank 2 = Somewhat similar to me
Rank 3 = Not similar to me at all
Read the following descriptions and rank them accordingly:
1. {items[0].text}
2. {items[1].text}
3. {items[2].text}
         
The answer must be provided in this format:
{{"Rank 1":X}}
{{"Rank 2":Y}}
{{"Rank 3":Z}}
"""

def call_model(model, prompt):

    explanation = model.respond(
        prompt,
        config={
            "temperature": 0,
            "topPSampling": 1,
            "repeatPenalty": 1,
            "seed": 42
        }
    )

    return explanation.content

def parse_ranking(response, perm):

    json_block = re.findall(r"\{(?:.|\r?\n)*?\}", response)[-1]

    data = json.loads(json_block)

    ranking = {}

    for key, position in data.items():

        rank = int(key.split()[1])
        item = perm[position - 1]

        ranking[item.id] = rank

    if set(ranking.values()) != {1,2,3}:
        raise ValueError("Invalid ranking returned")

    return ranking

results = []

models = ["google/gemma-3-4b"]

for model in models:

    for triad in triads:

        perms = get_permutations(triad)

        for perm_id, perm in enumerate(perms):

            prompt = build_prompt(perm)

            response = call_model(model, prompt)

            ranking = parse_ranking(response, perm)

            results.append(
                Result(
                    model=model,
                    triad_id=triad.id,
                    permutation_id=perm_id,
                    ranking=ranking
                )
            )