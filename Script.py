import lmstudio as lms
from lmstudio import Chat
import re
import time
from tqdm import tqdm 
import json
import pandas as pd

seed = 42
unload = True #True = Clear Model from memory, #False, for debugging single model
testing = True

available_models = lms.list_downloaded_models("llm")

models = [
    "google/gemma-3-4b", #wb, no reason
    "deepseek/deepseek-r1-0528-qwen3-8b", #reasoning, bit slow
    "meta-llama-3-8b-instruct", #fast, well behaved, no reasoning
    "qwen/qwen3-vl-8b", #instant, wb, no reasoning, probably remove
    "openai/gpt-oss-20b",#reasoning but very wb
    "google/gemma-3-27b"#slow but wb

    #"glm-4.7-flash",#not well behaved at all <- remove
    #"mistralai/ministral-3-14b-reasoning", # <- remove, diabolical behavior
]

model_lst = []
for model in available_models:
    model_lst.append(model.model_key)

for string in models:
    if string not in model_lst:
        raise ValueError(f'{string} is not a valid model name!')
    
print("Model naming check passed.")

items = [
    "People always have the ability to do otherwise.",
    "People always have free will.",
    "How people's lives unfold is completely up to them.",
    "People ultimately have complete control over their decisions and their actions.",
    "People have free will even when their choices are completely limited by external circumstances.",
    "Everything that has ever happened had to happen precisely as it did, given what happened before.",
    "Every event that has ever occurred, including human decisions and actions, was completely determined by prior events.",
    "People's choices and actions must happen precisely the way they do because of the laws of nature and the way things were in the distant past.",
    "A supercomputer that could know everything about the way the universe is now could know everything about the way the universe will be in the future.",
    "Given the way things were at the Big Bang, there is only one way for everything to happen in the universe after that.",
    "The fact that we have souls that are distinct from our material bodies is what makes humans unique.",
    "Each person has a non-physical essence that makes that person unique.",
    "The human mind cannot simply be reduced to the brain.",
    "The human mind is more than just a complicated biological machine.",
    "Human action can only be understood in terms of our souls and minds and not just in terms of our brains."
]

sds_prompt = """
Please rate the degree of social desirability of this question on a scale of 1 to 10, with 10 denoting the highest level and 1 denoting the lowest.
Question: {item}
The answer must be provided in this format:
{{"rating":"<number>"}}
"""

def extract_rating(text):
    #print(text)
    match = re.findall(r"\{.*?\}", text)

    if not match:
        return None

    data = json.loads(match[-1])

    return data["rating"]

def run_test(item, model):

    prompt = sds_prompt.format(item=item)

    # First call: explanation + rating
    explanation = model.respond(
        prompt,
        config={
            "temperature": 0,
            "topPSampling": 1,
            "repeatPenalty": 1,
            "seed": seed
            #"maxTokens": 5000
        }
    )

    # Build conversation to request clean integer
#    chat = Chat.from_history({
#        "messages": [
#            {"role": "user", "content": prompt},
#            {"role": "assistant", "content": explanation.content},
#            {"role": "user", "content": "Repeat your final rating as a single integer from 1 to 10. Output only the number and nothing else."}
#        ]
#    })
#
#    rating = model.respond(
#        chat,
#        config={
#            "temperature": 0,
#            "maxTokens": 5
#        }
#    )

    rating = extract_rating(explanation.content)
    return rating

results = {}

if testing == True:
    for model_name in tqdm(models, desc="Models"):

        print("\nLoading model:", model_name)

        model = lms.llm(model_name)

        scores = []

        for item in tqdm(items, desc="Items", leave=False):

            score = run_test(item, model)

            scores.append(score)

        results[model_name] = scores

        print("Completed:", model_name)
        print("Scores:", scores)
        if unload == True:
            model.unload()
        time.sleep(2)

    print("\nFinal Results:\n")

    for model, scores in results.items():
        print(model)
        print(scores)

    print("\nFinal Results:\n")

    for model, scores in results.items():
        print(model)
        print(scores)

    df = pd.DataFrame(results, index=items)
    df.to_csv("data.csv")