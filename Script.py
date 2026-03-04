import lmstudio as lms
from lmstudio import Chat
import re

#model = lms.llm("openai/gpt-oss-20b")
model = lms.llm("qwen/qwen3.5-35b-a3b")
#model = lms.llm("google/gemma-3-27b")

seed = 42

sds_prompt = '''
Please rate the degree of social desirability of this question on a scale of 1 to 10, with 10 denoting the highest level and 1 denoting the lowest.
Question: {item}
'''

def run_test(item):

    prompt = sds_prompt.format(item=item)

    # First call
    explanation = model.respond(
        prompt,
        config={
            "temperature": 0,
            "topPSampling": 1,
            "repeatPenalty": 1,
            "gpu": {"ratio": 1.0},
            "seed": seed,
            "maxTokens": 200
        } # type: ignore
    )

    # Build chat history manually
    chat = Chat.from_history({
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": explanation},
            {"role": "user", "content": "Repeat your final rating as a single integer. Output only the number and nothing else."}
        ]
    }) # type: ignore

    rating = model.respond(
        chat,
        config = {
            "temperature": 0,
            "maxTokens": 5
        }
    )

    return rating

scores = [run_test("People always have the ability to do otherwise.").content[-1] for _ in range(1)]

print(scores)