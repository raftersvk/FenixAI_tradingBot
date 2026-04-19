"""
LLM Factory for creating LangChain LLM instances from provider configurations.
Supports multiple providers with fallback logic and stub mode for development.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from config.llm_provider_config import LLMProvidersConfig, AgentProviderConfig


class LLMFactory:
    """LLM Factory supporting multiple providers with fallback and stub."""

    def __init__(self, config: Optional[LLMProvidersConfig] = None):
        # If no explicit config is passed, attempt to use the LLMProviderLoader
        if config is None:
            try:
                from src.config.llm_provider_loader import get_provider_loader

                loader = get_provider_loader()
                config = loader.get_config() or LLMProvidersConfig()
            except Exception:
                pass

        self.config = config or LLMProvidersConfig()
        self._llm_cache: dict[str, Any] = {}

    def get_llm_for_agent(self, agent_type: str) -> Any:
        """Gets the configured LLM for an agent type."""
        if agent_type in self._llm_cache:
            return self._llm_cache[agent_type]

        agent_config = self.config.get_agent_config(agent_type)
        llm = self.create_llm_from_config(agent_config)
        self._llm_cache[agent_type] = llm
        return llm

    def create_llm_from_config(self, agent_config: AgentProviderConfig) -> Any:
        """Create an LLM instance from an AgentProviderConfig."""
        return self._create_llm(agent_config)

    def _create_llm(self, config: AgentProviderConfig) -> Any:
        """Internal: creates an LLM instance based on configuration."""
        provider = config.provider_type
        model = config.model_name
        temperature = config.temperature
        api_key = config.api_key.get_secret_value() if config.api_key else None
        api_base = config.api_base

        try:
            if provider == "openai":
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    api_key=api_key,
                    max_tokens=config.max_tokens,
                    timeout=config.timeout,
                )

            elif provider == "anthropic":
                from langchain_anthropic import ChatAnthropic

                return ChatAnthropic(
                    model=model,
                    temperature=temperature,
                    api_key=api_key,
                    max_tokens=config.max_tokens,
                )

            elif provider == "groq":
                from langchain_groq import ChatGroq

                return ChatGroq(
                    model=model,
                    temperature=temperature,
                    api_key=api_key,
                    max_tokens=config.max_tokens,
                )

            elif provider in ("ollama_local", "ollama_cloud"):
                from langchain_ollama import ChatOllama

                return ChatOllama(
                    model=model,
                    temperature=temperature,
                    base_url=api_base
                    or (
                        "http://localhost:11434"
                        if provider == "ollama_local"
                        else "https://api.ollama.ai"
                    ),
                    num_predict=config.max_tokens,
                )

            elif provider == "huggingface_inference":
                from langchain_huggingface import ChatHuggingFace
                from langchain_huggingface import HuggingFaceEndpoint

                # Determine token from env or passed api_key
                hf_token = (
                    api_key
                    or os.getenv("HUGGINGFACE_API_KEY")
                    or os.getenv("HF_TOKEN")
                    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
                )
                endpoint = HuggingFaceEndpoint(
                    repo_id=model,
                    huggingfacehub_api_token=hf_token,
                    max_new_tokens=config.max_tokens,
                    temperature=temperature,
                )
                return ChatHuggingFace(llm=endpoint)

            else:
                raise ValueError(f"Provider '{provider}' not supported")

        except ImportError as e:
            # If the provider package is not installed, attempt fallback immediately if configured
            if config.fallback_provider_type and config.fallback_model_name:
                fallback_config = AgentProviderConfig(
                    provider_type=config.fallback_provider_type,
                    model_name=config.fallback_model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    timeout=config.timeout,
                )
                return self._create_llm(fallback_config)
            raise

        except Exception as e:
            # Attempt fallback if configured
            if config.fallback_provider_type and config.fallback_model_name:
                fallback_config = AgentProviderConfig(
                    provider_type=config.fallback_provider_type,
                    model_name=config.fallback_model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    timeout=config.timeout,
                )
                return self._create_llm(fallback_config)

            # If fallback fails or not configured, return a stub in dev mode
            allow_stub = os.getenv("LLM_ALLOW_NOOP_STUB", "1") == "1"
            if allow_stub:

                class NoopStub:
                    def __init__(self, name="noop"):
                        self.name = name

                    def invoke(self, messages):
                        return type(
                            "R",
                            (),
                            {
                                "content": '{"action": "HOLD", "confidence": 0.0, "reason": "LLM unavailable (stub)"}'
                            },
                        )

                    def ainvoke(self, messages):
                        # Simple sync result wrapped in a coroutine-like object for compatibility
                        return self.invoke(messages)

                return NoopStub(name=f"noop_{provider}")
            raise
