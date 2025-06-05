import os
import requests
import json
from pydantic import BaseModel
from typing import List, Optional, Dict

from .chat import LLMConfig, LLMMessage

# Default Crynux API endpoint
DEFAULT_CRYNUX_ENDPOINT = "https://bridge.crynux.ai/v1/llm"

# Free API key for ComfyUI-Art-Venture
DEFAULT_CRYNUX_API_KEY = "FZGUyWjUbY-ej-XGg-4HOac8vfJC3CLFrEF7OyGeCtw="

class CrynuxApi(BaseModel):
    api_key: Optional[str] = None
    endpoint: str = DEFAULT_CRYNUX_ENDPOINT
    timeout: Optional[int] = 60

    def chat(self, messages: List[LLMMessage], config: LLMConfig, seed=None):

        formated_messages = [m.to_openai_message() for m in messages]

        url = f"{self.endpoint}/chat/completions"
        payload = {
            "messages": formated_messages,
            "model": config.model,
            "max_tokens": config.max_token,
            "temperature": config.temperature,
        }
        if seed is not None:
            payload["seed"] = seed

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            headers["Authorization"] = f"Bearer {DEFAULT_CRYNUX_API_KEY}"

        print(f"Crynux API Request URL: {url}")
        print(f"Crynux API Request Headers: {headers}")
        print(f"Crynux API Request Payload: {payload}")

        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)

        print(f"Crynux API Response Status Code: {response.status_code}")
        try:
            response_data = response.json()
            print(f"Crynux API Response Data: {response_data}")
        except requests.exceptions.JSONDecodeError:
            response_data = response.text
            print(f"Crynux API Response Text: {response_data}")

        response.raise_for_status() # Raise an exception for HTTP errors
        data: Dict = response_data if isinstance(response_data, dict) else json.loads(response_data)

        if data.get("error", None) is not None:
            raise Exception(data.get("error").get("message"))

        return data["choices"][0]["message"]["content"]

    def complete(self, prompt: str, config: LLMConfig, seed=None):

        messages = [LLMMessage(role="user", text=prompt)]
        return self.chat(messages, config, seed)


class CrynuxLLMApiNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "crynux_api_key": ("STRING", {"multiline": False, "default": ""}),
                "endpoint": ("STRING", {"multiline": False, "default": DEFAULT_CRYNUX_ENDPOINT}),
            },
        }

    RETURN_TYPES = ("LLM_API",)
    FUNCTION = "create_api"
    CATEGORY = "ArtVenture/LLM"

    def create_api(self, crynux_api_key=None, endpoint=DEFAULT_CRYNUX_ENDPOINT):
        # Use environment variable if key is not provided but exists
        if not crynux_api_key or crynux_api_key == "":
            crynux_api_key = os.environ.get("CRYNUX_API_KEY", None)

        # If endpoint is empty, use default
        if not endpoint or endpoint == "":
            endpoint = DEFAULT_CRYNUX_ENDPOINT

        return (CrynuxApi(api_key=crynux_api_key, endpoint=endpoint),)


class CrynuxLLMApiConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "Qwen/Qwen2.5-7B-Instruct", "multiline": False}),
                "max_token": ("INT", {"default": 1024, "min": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LLM_CONFIG",)
    RETURN_NAMES = ("llm_config",)
    FUNCTION = "make_config"
    CATEGORY = "ArtVenture/LLM"

    def make_config(self, model: str, max_token: int, temperature: float):
        return (LLMConfig(model=model, max_token=max_token, temperature=temperature),)


NODE_CLASS_MAPPINGS = {
    "AV_CrynuxLLMApi": CrynuxLLMApiNode,
    "AV_CrynuxLLMApiConfig": CrynuxLLMApiConfigNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AV_CrynuxLLMApi": "Crynux LLM API",
    "AV_CrynuxLLMApiConfig": "Crynux LLM API Config",
}
