import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Optional, Any, Dict, List, Tuple

from pydantic import BaseModel, ValidationError

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

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


class LLMFactory:
    """
    Factory for instantiating LangChain BaseChatModel instances
    supporting OpenAI, DeepInfra, Anthropic, Ollama, Groq, and Gemini.
    """

    @staticmethod
    def create_chat_model(
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2
    ) -> BaseChatModel:
        provider = provider.lower().strip()

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            key = api_key or os.getenv("OPENAI_API_KEY")
            return ChatOpenAI(
                model=model_name,
                api_key=key,
                temperature=temperature
            )

        elif provider == "deepinfra":
            from langchain_openai import ChatOpenAI
            model_name = model or os.getenv("DEEPINFRA_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")
            key = api_key or os.getenv("DEEPINFRA_API_KEY")
            url = base_url or os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
            return ChatOpenAI(
                model=model_name,
                api_key=key,
                base_url=url,
                temperature=temperature
            )

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            model_name = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            return ChatAnthropic(
                model=model_name,
                api_key=key,
                temperature=temperature
            )

        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            model_name = model or os.getenv("GEMINI_MODEL", GEMINI_MODEL or "gemini-2.5-flash")
            key = api_key or os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=key,
                temperature=temperature
            )

        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            model_name = model or os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
            url = base_url or os.getenv("OLLAMA_HOST", OLLAMA_HOST)
            return ChatOllama(
                model=model_name,
                base_url=url,
                temperature=temperature
            )

        elif provider == "groq":
            from langchain_openai import ChatOpenAI
            model_name = model or GROQ_MODEL
            key = api_key or GROQ_API_KEY
            return ChatOpenAI(
                model=model_name,
                api_key=key,
                base_url="https://api.groq.com/openai/v1",
                temperature=temperature
            )

        else:
            raise ValueError(f"Unsupported LLM provider: '{provider}'")


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


class LangChainProvider(BaseLLMProvider):
    """
    Unified LangChain-backed Provider wrapping BaseChatModel created via LLMFactory.
    """

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.provider_name = provider.lower().strip()
        self.model_name = model
        self.api_key = api_key
        self.base_url = base_url
        self._chat_model: Optional[BaseChatModel] = None

    def is_available(self) -> bool:
        if self.provider_name in ["openai", "deepinfra", "anthropic", "groq", "gemini"]:
            key = self.api_key or os.getenv(f"{self.provider_name.upper()}_API_KEY", "")
            if self.provider_name == "groq" and not key:
                key = GROQ_API_KEY
            elif self.provider_name == "gemini" and not key:
                key = GEMINI_API_KEY
            return bool(key and len(key.strip()) > 5)
        elif self.provider_name == "ollama":
            import requests
            url = self.base_url or os.getenv("OLLAMA_HOST", OLLAMA_HOST)
            try:
                r = requests.get(f"{url.rstrip('/')}/api/tags", timeout=2)
                return r.status_code == 200
            except Exception:
                return False
        return False

    def _get_model(self) -> BaseChatModel:
        if self._chat_model is None:
            self._chat_model = LLMFactory.create_chat_model(
                provider=self.provider_name,
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._chat_model

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        model = self._get_model()
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        response = model.invoke(messages)
        return str(response.content or "")

    def generate_structured(
        self,
        prompt: str,
        schema_class: Type[T],
        system_prompt: Optional[str] = None
    ) -> T:
        try:
            model = self._get_model()
            structured_model = model.with_structured_output(schema_class)
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            res = structured_model.invoke(messages)
            if isinstance(res, schema_class):
                return res
        except Exception as e:
            logger.warning(f"LangChain with_structured_output fallback to text schema parsing due to: {e}")

        return super().generate_structured(prompt, schema_class, system_prompt)


class GroqProvider(LangChainProvider):
    def __init__(self, api_key: str = GROQ_API_KEY, model: str = GROQ_MODEL):
        super().__init__(provider="groq", model=model, api_key=api_key)


class GeminiProvider(LangChainProvider):
    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = GEMINI_MODEL):
        super().__init__(provider="gemini", model=model, api_key=api_key)


class OllamaProvider(LangChainProvider):
    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        super().__init__(provider="ollama", model=model, base_url=host)


class UnifiedLLMClient:
    """
    Unified LLM Client managing multiple LangChain providers with automated fallback logic.
    Supports OpenAI, DeepInfra, Anthropic, Ollama, Groq, and Gemini.
    """

    def __init__(self, primary_provider: str = PRIMARY_LLM_PROVIDER):
        self.providers: Dict[str, BaseLLMProvider] = {
            "groq": GroqProvider(),
            "gemini": GeminiProvider(),
            "ollama": OllamaProvider()
        }
        self.primary_name = primary_provider.lower()

        # Dynamically register additional providers if specified as primary or available in environment
        if self.primary_name not in self.providers and self.primary_name in ["deepinfra", "openai", "anthropic"]:
            self.providers[self.primary_name] = LangChainProvider(self.primary_name)

    def register_provider(self, name: str, provider: BaseLLMProvider):
        """Registers or replaces an LLM provider dynamically."""
        self.providers[name.lower()] = provider

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
