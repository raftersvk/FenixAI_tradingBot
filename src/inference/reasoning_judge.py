from __future__ import annotations

import json
import logging
import threading
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config.judge_config import JudgeModelConfig, get_judge_model_config
from src.inference.llm_factory import LLMFactory
from src.inference.model_id_normalizer import normalize_model_id_for_provider
from config.llm_provider_config import AgentProviderConfig
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


@dataclass
class ReasoningJudgePayload:
    """Context passed to the LLM-as-a-judge."""

    agent_name: str
    prompt: str
    normalized_result: Dict[str, Any]
    raw_response: str
    backend: str
    metadata: Dict[str, Any]
    latency_ms: Optional[float] = None


@dataclass
class JudgeVerdict:
    """Structured verdict returned by the judge."""

    verdict: str
    score: float
    confidence: float
    critique: str
    success_estimate: Optional[bool] = None
    risks: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    raw_response: str = ""

    def as_entry_payload(self) -> Dict[str, Any]:
        """Flatten verdict so ReasoningBank can persist it."""

        return {
            "verdict": self.verdict,
            "score": self.score,
            "confidence": self.confidence,
            "notes": self.critique,
            "success_estimate": self.success_estimate,
            "tags": self.tags,
            "metadata": {
                "risks": self.risks,
                "improvements": self.improvements,
                "raw_response": self.raw_response,
            },
        }


class ReasoningLLMJudge:
    """LLM-powered judge that scores reasoning traces via LangChain LLMs."""

    def __init__(self, config: Optional[JudgeModelConfig] = None):
        self.config = config or get_judge_model_config()
        # Normalize model ID once
        self.model_id = normalize_model_id_for_provider(self.config.model_id, self.config.provider)
        # Map 'ollama' to 'ollama_local' for compatibility with AgentProviderConfig
        provider_type = self.config.provider
        if provider_type == "ollama":
            provider_type = "ollama_local"
        # Build a minimal AgentProviderConfig for the judge
        agent_config = AgentProviderConfig(
            provider_type=provider_type,
            model_name=self.model_id,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        # Create LLM via LLMFactory
        factory = LLMFactory()
        try:
            self._llm = factory.create_llm_from_config(agent_config)
        except Exception as exc:
            logger.debug("ReasoningLLMJudge: failed to create LLM: %s", exc)
            self._llm = None

    def evaluate(self, payload: ReasoningJudgePayload) -> Optional[JudgeVerdict]:
        if self._llm is None:
            logger.debug("ReasoningLLMJudge: LLM not available, skipping")
            return None

        prompt = self._build_prompt(payload)
        messages = []
        if self.config.system_prompt:
            messages.append(SystemMessage(content=self.config.system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            response = self._llm.invoke(messages)
            # Extract text content
            if hasattr(response, "content"):
                text = response.content
            else:
                text = str(response)
            text = text.strip()
        except Exception as exc:
            logger.debug("ReasoningLLMJudge LLM error: %s", exc)
            return None

        parsed = self._parse_response(text)
        if not parsed:
            return None

        success_estimate = parsed.get("success_estimate")
        if isinstance(success_estimate, str):
            lowered = success_estimate.strip().lower()
            if lowered in ("true", "yes", "approve", "approved"):
                success_estimate = True
            elif lowered in ("false", "no", "reject", "rejected"):
                success_estimate = False
            else:
                success_estimate = None

        verdict = JudgeVerdict(
            verdict=parsed.get("verdict", "inconclusive").lower(),
            score=float(parsed.get("score", 0.5)),
            confidence=float(parsed.get("confidence", 0.5)),
            critique=parsed.get("critique", "").strip(),
            success_estimate=success_estimate,
            risks=list(parsed.get("risks") or []),
            improvements=list(parsed.get("improvements") or []),
            tags=list(parsed.get("tags") or []),
            raw_response=text,
        )

        return verdict

    def _build_prompt(self, payload: ReasoningJudgePayload) -> str:
        norm_json = json.dumps(payload.normalized_result, ensure_ascii=False, indent=2)
        metadata_json = json.dumps(payload.metadata or {}, ensure_ascii=False, indent=2)
        raw = payload.raw_response
        if not isinstance(raw, str):
            raw = json.dumps(raw, ensure_ascii=False, indent=2)

        instruction = (
            "You are ReasoningBank-Judge. Evaluate the trading decision below and "
            "decide if the reasoning should be APPROVED, REJECTED or marked "
            "INCONCLUSIVE. Focus on internal coherence, signal alignment and risk "
            "controls. Output STRICT JSON with the schema described."
        )

        schema = (
            "{\n"
            '  "verdict": "approve|reject|inconclusive",\n'
            '  "success_estimate": true|false|null,\n'
            '  "score": 0.0-1.0,\n'
            '  "confidence": 0.0-1.0,\n'
            '  "critique": "short analysis",\n'
            '  "risks": ["list of concrete risks"],\n'
            '  "improvements": ["list of actionable improvements"],\n'
            '  "tags": ["optional labels"]\n'
            "}"
        )

        context = (
            f"### Agent Context\n"
            f"- Agent: {payload.agent_name}\n"
            f"- Backend: {payload.backend}\n"
            f"- LatencyMs: {payload.latency_ms}\n\n"
            f"### Task Prompt\n{payload.prompt}\n\n"
            f"### Normalized Result\n{norm_json}\n\n"
            f"### Raw Response\n{raw}\n\n"
            f"### Additional Metadata\n{metadata_json}\n\n"
            f"Return ONLY valid JSON matching this schema:\n{schema}\n"
        )

        return f"{instruction}\n\n{context}"

    def _parse_response(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        # First try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strip thinking markers if present (e.g., "Thinking...\n...done thinking.\n")
        clean_text = text
        if "...done thinking" in text:
            parts = text.split("...done thinking")
            if len(parts) > 1:
                clean_text = parts[-1].strip()
                if clean_text.startswith("."):
                    clean_text = clean_text[1:].strip()

        # Try to find JSON in code blocks first (```json ... ```)
        import re

        json_block_match = re.search(r"```json\s*\n?(.*?)\n?```", clean_text, re.DOTALL)
        if json_block_match:
            try:
                return json.loads(json_block_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Find all potential JSON objects by matching balanced braces
        def find_json_objects(s: str) -> list:
            objects = []
            depth = 0
            start = None
            for i, char in enumerate(s):
                if char == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0 and start is not None:
                        objects.append(s[start : i + 1])
                        start = None
            return objects

        json_objects = find_json_objects(clean_text)

        # Try parsing from last to first (most recent JSON is likely the answer)
        for obj_str in reversed(json_objects):
            try:
                parsed = json.loads(obj_str)
                # Validate it has expected fields
                if isinstance(parsed, dict) and "verdict" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue

        # Fallback: try any valid JSON object
        for obj_str in reversed(json_objects):
            try:
                parsed = json.loads(obj_str)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        logger.debug("ReasoningLLMJudge: no valid JSON found in response.")
        return None


_judge_instance: Optional[ReasoningLLMJudge] = None
_judge_lock = threading.Lock()


def get_reasoning_judge() -> Optional[ReasoningLLMJudge]:
    """Return a singleton judge instance, or None if initialization fails."""

    global _judge_instance
    if _judge_instance is None:
        with _judge_lock:
            if _judge_instance is None:
                try:
                    _judge_instance = ReasoningLLMJudge()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Unable to initialize ReasoningLLMJudge: %s", exc)
                    _judge_instance = None
    return _judge_instance
