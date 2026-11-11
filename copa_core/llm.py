"""Thin, swappable LLM layer.

The instructor's starter code handles credentials, so the agent never hard-codes a provider:
  * call `set_llm(fn)` with any `fn(system: str, user: str) -> str` (e.g. the starter's client), or
  * set ANTHROPIC_API_KEY / OPENAI_API_KEY and it will auto-detect, or
  * set COPA_LLM=off to force the deterministic fallback (used by the test suite).

Every LLM output is parsed as JSON, validated, and re-checked by the deterministic layer.
If the LLM is unavailable or returns junk, callers fall back to deterministic text.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
_custom = None


def set_llm(fn):
    global _custom
    _custom = fn


def _auto():
    if os.environ.get("COPA_LLM", "auto").lower() == "off":
        return None
    model = os.environ.get("COPA_LLM_MODEL")
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()

            def call(system, user):
                r = client.messages.create(model=model or "claude-sonnet-5-5", max_tokens=1500, system=system,
                                           messages=[{"role": "user", "content": user}])
                return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return call
        except Exception:
            return None
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()

            def call(system, user):
                r = client.chat.completions.create(model=model or "gpt-4.1-mini",
                                                   messages=[{"role": "system", "content": system},
                                                             {"role": "user", "content": user}])
                return r.choices[0].message.content
            return call
        except Exception:
            return None
    return None


def available() -> bool:
    return os.environ.get("COPA_LLM", "auto").lower() != "off" and (_custom is not None or _auto() is not None)


def ask_json(prompt_name: str, data: dict) -> dict | None:
    """Fill prompts/<prompt_name>.txt with `data` (as JSON) and return the parsed JSON reply, or None."""
    if os.environ.get("COPA_LLM", "auto").lower() == "off":
        return None
    fn = _custom or _auto()
    if fn is None:
        return None
    text = (PROMPTS / f"{prompt_name}.txt").read_text()
    system, _, user_tmpl = text.partition("\n---\n")
    user = user_tmpl.replace("{{DATA}}", json.dumps(data, indent=1))
    try:
        reply = fn(system.strip(), user.strip())
        m = re.search(r"\{.*\}", reply or "", re.S)
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None
