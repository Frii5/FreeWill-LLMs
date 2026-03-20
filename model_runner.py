"""
Run forced-choice FWI experiments across multiple LLM providers.

Requires:
    Computing triads first
    Free Will Inventory CSV
    valid API keys as environment variables if using API models

Functionality:
    Contains functionality to call API models
    Control loading/unloading LMStudio instances
    Parse Model output -> expects JSON formatting
"""
#Companies:
from openai import OpenAI
from google import genai
from google.genai import types
import anthropic
from mistralai.client import Mistral
import lmstudio as lms
#Python:
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
    #{"provider": "lmstudio", "name": "google/gemma-3-4b"},
    #{"provider": "lmstudio", "name": "meta-llama-3-8b-instruct"},
    #{"provider": "lmstudio", "name": "qwen/qwen3-vl-8b"},
    #{"provider": "lmstudio", "name": "openai/gpt-oss-20b"},
    {"provider": "lmstudio", "name": "nvidia/nemotron-3-nano"},
    #{"provider": "openai", "name": "gpt-5.4"},
    #{"provider": "google", "name": "gemini-3.1-pro-preview"},
    #{"provider": "anthropic", "name": "claude-opus-4-6"},
    #{"provider": "mistral", "name": "mistral-large-2512"},
    #{"provider": "mistral", "name": "mistral-medium-2508"}
]

def validate_lmstudio_models(models: list[dict[str, str]]) -> None:
    available_models = lms.list_downloaded_models("llm")
    installed_model_names = {model.model_key for model in available_models}

    invalid_models = [
        model_cfg["name"]
        for model_cfg in models
        if model_cfg["provider"] == "lmstudio"
        and model_cfg["name"] not in installed_model_names
    ]

    if invalid_models:
        raise ValueError(f"Invalid LM Studio model name(s): {invalid_models}")

triad_indices1 = [
    (0, 0, 0),
    (1, 1, 2),
    (2, 2, 3),
    (3, 3, 4),
    (4, 4, 1),
    (0, 1, 3),
    (1, 2, 1),
    (2, 0, 4),
    (3, 4, 2),
    (4, 3, 0),
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

class APIHandler:
    def __init__(self):
        self.openai_client = OpenAI()
        self.loaded_lmstudio_model = None
        self.google_client = genai.Client()
        self.anthropic_client = anthropic.Anthropic()
        mistral_api_key = os.environ["MISTRAL_API_KEY"]
        self.mistral_client = Mistral(api_key=mistral_api_key)

    def call_openai(self, model_name, prompt, temperature=0.0, top_p=1.0) -> str:
        response = self.openai_client.responses.create(
            model=model_name,
            input=prompt,
            temperature=temperature,
            top_p=top_p,
            reasoning={"effort": "none"}#default
        )
        return response.output_text

    def call_google(self, model_name, prompt, temperature=0.0, top_p=1.0) -> str:
        response = self.google_client.models.generate_content(
            model = model_name, 
            contents = prompt,
            config=types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            thinking_config=types.ThinkingConfig(thinking_level="low")#default
            ),
        )
        return response.text
    
    def call_anthropic(self, model_name, prompt, temperature=0.0, top_p=1.0) -> str:
        response = self.anthropic_client.messages.create(
            model = model_name,
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 1024,
            temperature = temperature
        ) # type: ignore
        return response.content[0].text
    
    def call_mistral(self, model_name, prompt, temperature=0.0, top_p=1.0) -> str:
        response = self.mistral_client.chat.complete(
            model = model_name,
            messages = [{"role": "user", "content": prompt}],
            temperature = temperature,
            #top_p=top_p,
        )
        return response.choices[0].message.content

    # ::: LMStudio Local Functionality ::: 
    def call_lmstudio(self, prompt, temperature=0.0, top_p=1.0):
        if self.loaded_lmstudio_model is None:
            raise ValueError("No LM Studio model is currently loaded.")

        response = self.loaded_lmstudio_model.respond(
            prompt,
            config={
                "temperature": temperature,
                "repeatPenalty": 1,
                "seed": 42
            }  # pyright: ignore[reportArgumentType]
        )
        return response.content
    
    def load_lmstudio_model(self, model_name):
        self.loaded_lmstudio_model = lms.llm(model_name)

    def unload_lmstudio_model(self):
        if self.loaded_lmstudio_model is not None:
            self.loaded_lmstudio_model.unload()
            self.loaded_lmstudio_model = None

class LLMRunner:
    def __init__(self, api_handler: APIHandler) -> None:
        self.api_handler = api_handler

    def generate_response(self,
        model_cfg: dict[str, str], 
        prompt: str, 
        temperature: float = 0.0, 
        top_p: float = 1.0
        ) -> str:

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
        
        elif provider == "mistral":
            return self.api_handler.call_mistral(
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

def build_prompt1(items: tuple) -> str:
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
    

def build_prompt2():
    return dedent(f"""\
        Please rate the degree of social desirability of this question on a scale of 1 to 10, with 10 denoting the highest level and 1 denoting the lowest. You should also explain your assessment.
        Question: Human action can only be understood in terms of our souls and minds and not just in terms of our brains.
        The answer must be provided in this format:
        {{"rating": X}}
        """)

def smoke_test(runner: LLMRunner, model: dict[str, str]) -> None:
    response = runner.generate_response(
        model, build_prompt2()
    )
    print(model["name"])
    print(type(response))
    print(response)
    print("-" * 40)
    
def Run_Experiment() -> None:
    validate_lmstudio_models(models)
    errors = 0
    api_handler = APIHandler()
    runner = LLMRunner(api_handler)
    triads = load_part1_triads(path, triad_indices)
    print("Test starting:")
    for model in models:
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











