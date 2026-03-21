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
from structures import *
import os

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










