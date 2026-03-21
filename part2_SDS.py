from pathlib import Path
from textwrap import dedent
import json
import re
import time
import pandas as pd
import pickle

from model_runner import *
from structures import PART2_ITEMS

class SocialDesirabilityScoring:
    def __init__(self, models, items, api_handler, runner, out_dir, file_name):
        self.models = models
        self.items = items
        self.api_handler = api_handler
        self.runner = runner
        self.out_dir = Path(out_dir)
        self.file_name = file_name

    def run(self) -> pd.DataFrame:
        print("Starting Test:")
        results = {}
        detailed_results = []
        for model in self.models:
            provider = model["provider"]
            model_name = model["name"]
            scores = []

            try:
                if provider == "lmstudio":
                    self.api_handler.load_lmstudio_model(model_name)

                print(f"{model_name} - Starting Scoring:")

                for i, item in enumerate(self.items):
                    print(i)

                    if provider == "mistral":
                        time.sleep(3)

                    prompt = self.build_prompt(item.text)
                    response = self.runner.generate_response(model, prompt)
                    score = self.extract_rating(response)
                    scores.append(score)

                    detailed_results.append(
                        SDSResult(
                            model=model_name,
                            item_id=item.id,
                            response=response,
                        )
                    )

                results[model_name] = scores

            finally:
                if provider == "lmstudio":
                    self.api_handler.unload_lmstudio_model()
                    time.sleep(2)

        df = self.build_dataframe(results)
        self.save_csv(df)
        self.save_pkl(detailed_results)
        return df

    def build_dataframe(self, results: dict) -> pd.DataFrame:
        df = pd.DataFrame(results, index=self.items)
        df["med"] = df.median(axis=1, skipna=True)
        df["avg"] = df.mean(axis=1, skipna=True)
        return df

    def save_csv(self, df: pd.DataFrame) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out_dir / self.file_name)

    def save_pkl(self, detailed_results) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "sds2_prompt_log.pkl", "wb") as f:
            pickle.dump(detailed_results, f)

    @staticmethod
    def build_prompt(item: str) -> str:
        return dedent(f"""\
            Please rate the degree of social desirability of this statement on a scale of 1 to 10, with 10 denoting the highest level and 1 denoting the lowest.
            Question: {item}
            The answer must be provided in this format:
            {{"rating": X}}
        """)

    @staticmethod
    def extract_rating(text: str) -> int | None:
        matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
        if not matches:
            return None

        obj_text = matches[-1]

        try:
            data = json.loads(obj_text)
            return int(data["rating"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            rating_match = re.search(r'"rating"\s*:\s*(\d+)', obj_text)
            if rating_match:
                return int(rating_match.group(1))
            return None
        
if __name__ == "__main__":
    api_handler = APIHandler()
    runner = LLMRunner(api_handler)

    models = [
        {"provider": "lmstudio", "name": "google/gemma-3-4b"},
        {"provider": "lmstudio", "name": "meta-llama-3-8b-instruct"},
        {"provider": "lmstudio", "name": "qwen/qwen3-vl-8b"},
        {"provider": "lmstudio", "name": "openai/gpt-oss-20b"},
        {"provider": "lmstudio", "name": "nvidia/nemotron-3-nano"},
        {"provider": "openai", "name": "gpt-5.4"},
        {"provider": "google", "name": "gemini-3.1-pro-preview"},
        {"provider": "anthropic", "name": "claude-opus-4-6"},
        {"provider": "mistral", "name": "mistral-large-2512"},
        {"provider": "mistral", "name": "mistral-medium-2508"}
    ]

    experiment = SocialDesirabilityScoring(
        models=models,
        items=PART2_ITEMS,
        api_handler=api_handler,
        runner=runner,
        out_dir="results_SDS",
        file_name="sds_scores_part2.csv",
    )

    df = experiment.run()
    print(df)
