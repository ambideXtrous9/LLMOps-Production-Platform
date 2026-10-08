"""
config/pii_guardrail.py
Gateway PII masking guardrail for LiteLLM (Plane 1).

Masks e-mail addresses, payment card numbers, US SSNs and API secrets in every
chat message *before* the request reaches the model, so raw PII never lands in
model context, prompt caches, spend logs or traces. Patterns mirror
models/retention_policy.yaml (pii_redaction_policy.entities_masked).

Mounted into the LiteLLM container at /app/pii_guardrail.py and enabled in
config/litellm.yaml:

    guardrails:
      - guardrail_name: pii-masking
        litellm_params:
          guardrail: pii_guardrail.PIIMaskingGuardrail
          mode: pre_call
          default_on: true
"""

import re
from typing import Any, List, Optional, Tuple

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail

# Order matters: secrets first so an "sk-..." key is never half-matched as a card.
PII_PATTERNS: List[Tuple[str, "re.Pattern[str]", str]] = [
    ("api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "[REDACTED_SECRET]"),
    ("email", re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+"), "[REDACTED_EMAIL]"),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){12,15}\d\b"), "[REDACTED_CARD]"),
]


def mask_text(text: str) -> Tuple[str, List[str]]:
    """Returns the masked text and the entity types that were found."""
    found: List[str] = []
    for name, pattern, replacement in PII_PATTERNS:
        text, count = pattern.subn(replacement, text)
        if count:
            found.append(name)
    return text, found


def _mask_content(content: Any) -> Tuple[Any, List[str]]:
    """Handles both plain-string content and OpenAI content-part lists (vision requests)."""
    if isinstance(content, str):
        return mask_text(content)
    if isinstance(content, list):
        found: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                part["text"], part_found = mask_text(part["text"])
                found.extend(part_found)
        return content, found
    return content, []


class PIIMaskingGuardrail(CustomGuardrail):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Optional[dict]:
        found: List[str] = []
        for message in data.get("messages") or []:
            if isinstance(message, dict) and "content" in message:
                message["content"], message_found = _mask_content(message["content"])
                found.extend(message_found)
        if isinstance(data.get("prompt"), str):
            data["prompt"], prompt_found = mask_text(data["prompt"])
            found.extend(prompt_found)
        if found:
            verbose_proxy_logger.info("pii-masking guardrail redacted entities: %s", sorted(set(found)))
        return data
