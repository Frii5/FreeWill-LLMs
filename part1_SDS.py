from model_runner import *

models = [
    #{"provider": "lmstudio", "name": "google/gemma-3-4b"},
    #{"provider": "lmstudio", "name": "meta-llama-3-8b-instruct"},
    #{"provider": "lmstudio", "name": "qwen/qwen3-vl-8b"},
    #{"provider": "lmstudio", "name": "openai/gpt-oss-20b"},
    #{"provider": "lmstudio", "name": "nvidia/nemotron-3-nano"},
    #{"provider": "openai", "name": "gpt-5.4"},
    #{"provider": "google", "name": "gemini-3.1-pro-preview"},
    #{"provider": "anthropic", "name": "claude-opus-4-6"},
    {"provider": "mistral", "name": "mistral-medium-2508"},
    #{"provider": "mistral", "name": "mistral-large-2512"}
]

items = [
    #"People always have the ability to do otherwise.",
    #"People always have free will.",
    #"How people's lives unfold is completely up to them.",
    #"People ultimately have complete control over their decisions and their actions.",
    #"People have free will even when their choices are completely limited by external circumstances.",
    #"Everything that has ever happened had to happen precisely as it did, given what happened before.",
    #"Every event that has ever occurred, including human decisions and actions, was completely determined by prior events.",
    #"People's choices and actions must happen precisely the way they do because of the laws of nature and the way things were in the distant past.",
    #"A supercomputer that could know everything about the way the universe is now could know everything about the way the universe will be in the future.",
    #"Given the way things were at the Big Bang, there is only one way for everything to happen in the universe after that.",
    #"The fact that we have souls that are distinct from our material bodies is what makes humans unique.",
    #"Each person has a non-physical essence that makes that person unique.",
    #"The human mind cannot simply be reduced to the brain.",
    "The human mind is more than just a complicated biological machine.",
    "Human action can only be understood in terms of our souls and minds and not just in terms of our brains."
]

def build_prompt(item):
    return dedent(f"""\
        Please rate the degree of social desirability of this statement on a scale of 1 to 10, with 10 denoting the highest level and 1 denoting the lowest. You should also explain your assessment.
        Question: {item}
        The answer must be provided in this format:
        {{"rating": X}}
        """)

def extract_rating(text):
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

api_handler = APIHandler()
runner = LLMRunner(api_handler)
results = {}

print("Starting Test:")
#Main experiment Loop:
for model in models:
    provider = model["provider"]
    model_name = model["name"]
    try:
        if provider == "lmstudio":
            api_handler.load_lmstudio_model(model_name)
        print(f"{model_name} - Starting Scoring:")
        scores = []
        for i, item in enumerate(items):

            print(i)
            if provider == "mistral":
                time.sleep(3)
        
            prompt = build_prompt(item)
            response = runner.generate_response(model, prompt)
            print(response)
            score = extract_rating(response)
            scores.append(score)

        results[model_name] = scores

    finally:
        if provider == "lmstudio":
            api_handler.unload_lmstudio_model()
            time.sleep(2) #clear memory

print("\nFinal Results:\n")

for model, scores in results.items():
    print(model)
    print(scores)

#df = pd.DataFrame(results, index=items)
#df.to_csv("data_new_prompt.csv")