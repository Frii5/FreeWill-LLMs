from __future__ import annotations

from model_runner import APIHandler, LLMRunner, validate_lmstudio_models
from structures import build_part1_triads

from pathlib import Path
from textwrap import dedent
import itertools
import json
import pickle
import re
import time

from structures import Result, Triad

class ForcedChoiceExperiment:
    def __init__(
        self,
        models: list[dict],
        triads: list[Triad],
        api_handler,
        runner,
        out_dir: str = "results_new_prompt",
    ) -> None:
        self.models = models
        self.triads = triads
        self.api_handler = api_handler
        self.runner = runner
        self.out_dir = Path(out_dir)

    def run(self) -> None:
        print("Test starting:")

        for model in self.models:
            self.run_model(model)

    def run_model(self, model: dict) -> None:
        model_name = model["name"]
        provider = model["provider"]

        print("-" * 40)
        print(f"Current model: {model_name}")

        results: list[Result] = []
        errors = 0

        self.prepare_model(provider, model_name)

        try:
            for triad in self.triads:
                print(f"Triad id: {triad.id}")

                for perm_id, perm in enumerate(self.get_permutations(triad)):
                    if model_name == "mistral-medium-2508":
                        time.sleep(4)

                    prompt = self.build_prompt(perm)
                    response = self.runner.generate_response(model, prompt)
                    ranking = self.parse_response(response, perm)

                    if ranking is None:
                        errors += 1

                    results.append(
                        Result(
                            model=model_name,
                            triad_id=triad.id,
                            permutation_id=perm_id,
                            ranking=ranking,
                            response=response, # pyright: ignore[reportCallIssue]
                        )
                    )

        finally:
            self.cleanup_model(provider)

        self.save_results(model_name, results)
        print(f"Found {errors} errors.")

    def prepare_model(self, provider: str, model_name: str) -> None:
        if provider == "lmstudio":
            self.api_handler.load_lmstudio_model(model_name)

    def cleanup_model(self, provider: str) -> None:
        if provider == "lmstudio":
            self.api_handler.unload_lmstudio_model()
            time.sleep(2)

    def save_results(self, model_name: str, results: list[Result]) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.out_dir / f"{model_name.replace('/', '_')}.pkl"

        with open(file_path, "wb") as f:
            pickle.dump(results, f)

    @staticmethod
    def get_permutations(triad: Triad):
        return list(itertools.permutations(triad.items))

    @staticmethod
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

    @staticmethod
    def parse_response(response: str, perm: tuple) -> dict[str, int] | None:
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
                text_to_id = {item.text.strip(): item.id for item in perm}
                return {
                    text_to_id[data["Rank 1"].strip()]: 1,
                    text_to_id[data["Rank 2"].strip()]: 2,
                    text_to_id[data["Rank 3"].strip()]: 3,
                }

        except (IndexError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        
if __name__ == "__main__":
    
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

    validate_lmstudio_models(models)
    api_handler = APIHandler()
    runner = LLMRunner(api_handler)
    triads = build_part1_triads()

    experiment = ForcedChoiceExperiment(
        models=models,
        triads=triads,
        api_handler=api_handler,
        runner=runner,
        out_dir="results_new_prompt",
    )
    experiment.run()