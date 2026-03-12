from openai import OpenAI
from google import genai
from google.genai import types
import anthropic

import lmstudio as lms
from textwrap import dedent
import json
import re
from structures import *
import itertools
import pandas as pd
from typing import List
import time

models = [
    {"provider": "lmstudio", "name": "google/gemma-3-4b"},
    {"provider": "lmstudio", "name": "meta-llama-3-8b-instruct"},
    {"provider": "lmstudio", "name": "qwen/qwen3-vl-8b"},
    {"provider": "lmstudio", "name": "openai/gpt-oss-20b"},
    {"provider": "lmstudio", "name": "google/gemma-3-27b"},
    {"provider": "openai", "name": "gpt-5.4"},
    {"provider": "google", "name": "gpt-5.4"}
]

triad_indices = [
    (0,0,0),
    (1,2,1),
    (2,1,3),
    (3,3,4),
    (4,4,2) #Add additional triads
]

path = "FWI_Part1_and_Part2.csv"

class APIHandler:
    def __init__(self):
        self.openai_client = OpenAI()
        self.loaded_lmstudio_model = None
        self.google_client = genai.Client()
        self.anthropic_client = anthropic.Anthropic()

    def call_openai(self, model_name, prompt, temperature=0, top_p=1):
        response = self.openai_client.responses.create(
            model=model_name,
            input=prompt,
            temperature=temperature,
            top_p=top_p,
            reasoning={"effort": "medium"}#default
        )
        return response.output_text

    def call_google(self, model_name, prompt, temperature=0, top_p=1):
        response = self.google_client.models.generate_content(
            model = model_name, 
            contents = prompt,
            config=types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            thinking_config=types.ThinkingConfig(thinking_level="high")#default
            ),
        )
        return response.text
    
    def call_anthropic(self, model_name, prompt, temperature=0, top_p=1):
        response = self.anthropic_client.messages.create(
            model = model_name,
            messages = [{"role": "user", "content": prompt}],
            temperature = temperature,
            top_p=top_p,
        ) # type: ignore
        return response.content[0].text
    
    def call_lmstudio(self, prompt, temperature=0, top_p=1):
        if self.loaded_lmstudio_model is None:
            raise ValueError("No LM Studio model is currently loaded.")

        response = self.loaded_lmstudio_model.respond(
            prompt,
            config={
                "temperature": temperature,
                "topPSampling": top_p,
                "repeatPenalty": 1,
                "seed": 42
            }  # pyright: ignore[reportArgumentType]
        )
        return response.content
    #Helper functions to keep loading/unloading functionality to this class
    def load_lmstudio_model(self, model_name):
        self.loaded_lmstudio_model = lms.llm(model_name)

    def unload_lmstudio_model(self):
        if self.loaded_lmstudio_model is not None:
            self.loaded_lmstudio_model.unload()
            self.loaded_lmstudio_model = None

class LLMRunner:
    def __init__(self, api_handler):
        self.api_handler = api_handler

    def generate_response(self, model_cfg, prompt, temperature=0, top_p=1):
        provider = model_cfg["provider"]
        model_name = model_cfg["name"]

        if provider == "openai":
            return self.api_handler.call_openai(
                model_name = model_name,
                prompt = prompt,
                temperature = temperature,
                top_p = top_p
            )
        
        elif provider == "google":
            return self.api_handler.call_google(
                model_name = model_name,
                prompt = prompt,
                temperature = temperature,
                top_p = top_p
            )
        
        elif provider == "anthropic":
            return self.api_handler.call_anthropic(
                model_name = model_name,
                prompt = prompt,
                temperature = temperature,
                top_p = top_p
            )

        elif provider == "lmstudio":
            return self.api_handler.call_lmstudio(
                prompt = prompt,
                temperature = temperature,
                top_p = top_p
            )
        
        else:
            raise ValueError(f"Wrong provider name: '{provider}'")

def load_items(part1, start, stop, prefix) -> List[Item]:
    rows = part1.iloc[start:stop]["Statement"].tolist()
    return [
        Item(id=f"{prefix}{i+1}", text=text, dimension=prefix)
        for i, text in enumerate(rows)
    ]

def load_part1_triads(csv_path: str, triad_indices: List[tuple[int, int, int]]) -> List[Triad]:
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

def build_prompt(items):
    return dedent(f"""\
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
        {{
        "Rank 1": X,
        "Rank 2": Y,
        "Rank 3": Z
        }}""")

def parse_ranking(response, perm):
    try:
        json_block = re.findall(r"\{[\s\S]*?\}", response)[-1]
        data = json.loads(json_block)

        return {
            perm[int(data["Rank 1"]) - 1].id: 1,
            perm[int(data["Rank 2"]) - 1].id: 2,
            perm[int(data["Rank 3"]) - 1].id: 3,
        }

    except (IndexError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    
results = []

api_handler = APIHandler()
runner = LLMRunner(api_handler)
triads = load_part1_triads(path, triad_indices)

for model_cfg in models:
    provider = model_cfg["provider"]
    model_name = model_cfg["name"]

    try:
        if provider == "lmstudio":
            api_handler.load_lmstudio_model(model_name)

        for triad in triads:
            for perm_id, perm in enumerate(get_permutations(triad)):
                prompt = build_prompt(perm)
                response = runner.generate_response(model_cfg, prompt)
                ranking = parse_ranking(response, perm)

                results.append(
                    Result(
                        model=model_name,
                        triad_id=triad.id,
                        permutation_id=perm_id,
                        ranking=ranking
                    )
                )

    finally:
        if provider == "lmstudio":
            api_handler.unload_lmstudio_model()
            time.sleep(2) #clear memory










