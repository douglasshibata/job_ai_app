import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Optional, Any, Dict, List, Tuple

import requests
from pydantic import BaseModel, ValidationError

try:
    from job_ai_app.config import (
        GROQ_API_KEY, GROQ_MODEL,
        GEMINI_API_KEY, GEMINI_MODEL,
        OLLAMA_HOST, OLLAMA_MODEL,
        PRIMARY_LLM_PROVIDER
    )
except ImportError:
    from config import (
        GROQ_API_KEY, GROQ_MODEL,
        GEMINI_API_KEY, GEMINI_MODEL,
        OLLAMA_HOST, OLLAMA_MODEL,
        PRIMARY_LLM_PROVIDER
    )

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def extract_json_from_text(text: str) -> str:
    """Extracts valid JSON string from text, stripping markdown code blocks if present."""
    text = text.strip()

    # Check for markdown json block
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if json_match:
        return json_match.group(1).strip()

    # Check if text starts with { and ends with }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace:last_brace + 1].strip()

    return text


class BaseLLMProvider(ABC):
    """Abstract Base Class for LLM Providers."""

    @abstractmethod
    def is_available(self) -> bool:
        """Checks if provider has required configuration or is reachable."""
        pass

    @abstractmethod
    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates raw text response."""
        pass

    def generate_structured(
        self,
        prompt: str,
        schema_class: Type[T],
        system_prompt: Optional[str] = None
    ) -> T:
        """Generates structured Pydantic output."""
        json_schema_str = json.dumps(schema_class.model_json_schema(), indent=2)
        full_system = (system_prompt or "") + (
            f"\n\nYou MUST respond strictly with a single valid JSON object matching the following JSON Schema:\n{json_schema_str}\nDo NOT include any commentary, intro, or markdown formatting outside the JSON."
        )
        raw_response = self.generate_text(prompt, system_prompt=full_system)
        clean_json_str = extract_json_from_text(raw_response)

        try:
            parsed_dict = json.loads(clean_json_str)
            return schema_class.model_validate(parsed_dict)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Failed to parse response into schema {schema_class.__name__}: {e}. Raw response: {raw_response[:200]}")
            raise ValueError(f"Provider response invalid for schema {schema_class.__name__}: {e}")


class GroqProvider(BaseLLMProvider):
    """Groq API Provider."""

    def __init__(self, api_key: str = GROQ_API_KEY, model: str = GROQ_MODEL):
        self.api_key = api_key
        self.model = model
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def _get_client(self):
        if self._client is None and self.is_available():
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        client = self._get_client()
        if not client:
            raise RuntimeError("Groq API key not configured or invalid.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"} if "json" in (system_prompt or "").lower() else None
        )
        return response.choices[0].message.content or ""


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API Provider."""

    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = GEMINI_MODEL):
        self.api_key = api_key
        self.model = model
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def _get_client(self):
        if self._client is None and self.is_available():
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Error initializing Google Gemini Client: {e}")
                self._client = None
        return self._client

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        client = self._get_client()
        if not client:
            raise RuntimeError("Google Gemini API key not configured or client initialization failed.")

        full_prompt = f"{system_prompt}\n\nUser Request:\n{prompt}" if system_prompt else prompt

        response = client.models.generate_content(
            model=self.model,
            contents=full_prompt,
        )
        return response.text or ""

    def generate_structured(
        self,
        prompt: str,
        schema_class: Type[T],
        system_prompt: Optional[str] = None
    ) -> T:
        client = self._get_client()
        if not client:
            return super().generate_structured(prompt, schema_class, system_prompt)

        full_prompt = f"{system_prompt}\n\nUser Request:\n{prompt}" if system_prompt else prompt
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema_class,
                },
            )
            return schema_class.model_validate_json(response.text)
        except Exception as e:
            logger.warning(f"Gemini structured response fallback to text parsing due to: {e}")
            return super().generate_structured(prompt, schema_class, system_prompt)


class OllamaProvider(BaseLLMProvider):
    """Local Ollama Provider."""

    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        self.host = host.rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.host}/api/chat"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2}
        }

        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_request()
        data = resp.json()
        return data.get("message", {}).get("content", "")


class UnifiedLLMClient:
    """
    Unified LLM Client managing multiple providers with automated fallback logic.
    """

    def __init__(self, primary_provider: str = PRIMARY_LLM_PROVIDER):
        self.providers: Dict[str, BaseLLMProvider] = {
            "groq": GroqProvider(),
            "gemini": GeminiProvider(),
            "ollama": OllamaProvider()
        }
        self.primary_name = primary_provider.lower()

    def get_provider_chain(self) -> List[Tuple[str, BaseLLMProvider]]:
        """Returns ordered list of (name, provider) starting with primary."""
        chain = []
        if self.primary_name in self.providers:
            chain.append((self.primary_name, self.providers[self.primary_name]))

        for name, provider in self.providers.items():
            if name != self.primary_name:
                chain.append((name, provider))
        return chain

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        errors = []
        for name, provider in self.get_provider_chain():
            if not provider.is_available():
                logger.info(f"Provider '{name}' is not available (key/service missing).")
                continue
            try:
                logger.info(f"Attempting text generation with provider '{name}'...")
                return provider.generate_text(prompt, system_prompt=system_prompt)
            except Exception as e:
                logger.warning(f"Provider '{name}' failed with error: {e}. Trying fallback...")
                errors.append(f"{name}: {e}")

        raise RuntimeError(f"All LLM providers failed or were unavailable. Errors: {'; '.join(errors)}")

    def generate_structured(
        self,
        prompt: str,
        schema_class: Type[T],
        system_prompt: Optional[str] = None
    ) -> T:
        errors = []
        for name, provider in self.get_provider_chain():
            if not provider.is_available():
                logger.info(f"Provider '{name}' is not available.")
                continue
            try:
                logger.info(f"Attempting structured generation with provider '{name}'...")
                return provider.generate_structured(prompt, schema_class, system_prompt=system_prompt)
            except Exception as e:
                logger.warning(f"Provider '{name}' structured generation failed: {e}. Trying fallback...")
                errors.append(f"{name}: {e}")

        raise RuntimeError(f"All LLM providers failed or were unavailable for structured output. Errors: {'; '.join(errors)}")
