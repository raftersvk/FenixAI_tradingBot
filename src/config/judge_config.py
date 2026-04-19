from __future__ import annotations

"""
Configuration helpers for the ReasoningBank LLM-as-a-Judge component.

The judge configuration is loaded from llm_providers.yaml.
"""

from dataclasses import dataclass


@dataclass
class JudgeModelConfig:
    """Small container describing which provider/model the judge should use."""

    provider: str
    model_id: str
    temperature: float = 0.1
    max_tokens: int = 512
    system_prompt: str = (
        "You are ReasoningBank-Judge, a strict auditor that labels agent "
        "reasoning traces as APPROVE/REJECT/INCONCLUSIVE and highlights risks."
    )


def get_judge_model_config() -> JudgeModelConfig:
    """Read judge configuration from LLM provider config (YAML) only."""

    try:
        from src.config.llm_provider_loader import get_provider_loader

        loader = get_provider_loader()
        cfg = loader.get_config()
        if cfg and cfg.judge:
            agent_cfg = cfg.judge
            return JudgeModelConfig(
                provider=agent_cfg.provider_type,
                model_id=agent_cfg.model_name,
                temperature=agent_cfg.temperature,
                max_tokens=agent_cfg.max_tokens,
                system_prompt=JudgeModelConfig.system_prompt,
            )
    except Exception as e:
        pass

    return JudgeModelConfig(
        provider="ollama_local",
        model_id="qwen3:8b",
        temperature=0.1,
        max_tokens=512,
        system_prompt=JudgeModelConfig.system_prompt,
    )
