"""Register the harness's Z.ai provider in ZCode's personal provider config.

ZCode 0.16.x only exposes models to the app-server registry through the
personal provider config (``~/.zcode/v2/provider_config.json``). The schema
is strict and under-documented; the shapes below are what the desktop itself
writes, verified against the runtime's own Zod schema and load logs:

- personal ``providerRules`` entries must use ``group: "standard-personal"``
  (not the family groups the built-in config uses) and must not declare
  ``builtinModelIds`` — their models come from ``personalModelIds``.
- each model needs a ``manualProviderModelRules`` entry whose ``config``
  carries ``properties`` and ``optionSpecs``; ``optionSpecs.reasoningLevel``
  supplies the reasoning dial values and its ``map`` is a *JSON string* of
  per-level request patches (empty patches keep the provider default).

The harness never owns this file: it merges its entries in and leaves
everything the user (or the desktop app) wrote untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

PROVIDER_ID = "zai-direct"
REASONING_LEVELS = ("low", "high", "max")

_EMPTY_LEVEL_PATCHES = json.dumps({level: {} for level in REASONING_LEVELS})

_DEFAULT_PROPERTIES = {
    "contextWindow": 200000,
    "supportsJsonSchemaOutput": False,
    "supportsNativeWebSearch": False,
    "supportsMidConversationSystem": True,
    "inputFormat": {"supportsImage": False, "supportsVideo": False, "supportsPdf": False},
}

_DEFAULT_BASE_URL = "https://api.z.ai/api/anthropic"


def default_provider_config_path() -> Path:
    return Path.home() / ".zcode" / "v2" / "provider_config.json"


def _provider_rule(api_key: str, base_url: str, model_ids: Sequence[str]) -> dict:
    return {
        "providerId": PROVIDER_ID,
        "providerName": "Z.ai Direct (lhht)",
        "enabled": True,
        "config": {
            "group": "standard-personal",
            "access": {"type": "zhipu-coding-plan-api-key", "apiKey": api_key},
            "api": {"type": "anthropic-messages", "baseUrl": base_url},
            "visibility": "visible",
            "personalModelIds": list(model_ids),
        },
    }


def _model_rule(model_id: str) -> dict:
    return {
        "providerId": PROVIDER_ID,
        "modelId": model_id,
        "config": {
            "enabled": True,
            "properties": dict(_DEFAULT_PROPERTIES),
            "optionSpecs": {
                "reasoningLevel": {
                    "values": list(REASONING_LEVELS),
                    "map": _EMPTY_LEVEL_PATCHES,
                },
                "maxOutputTokens": {"max": 32768},
            },
        },
    }


def ensure_provider_config(
    api_key: str,
    model_id: str,
    *,
    base_url: str = _DEFAULT_BASE_URL,
    path: Path | None = None,
) -> Path:
    """Merge the harness provider into ZCode's personal provider config.

    Idempotent: an existing ``zai-direct`` entry is refreshed in place (key,
    endpoint) and the matching model rule is upserted. Any other rules in the
    file are preserved. Returns the config path.

    The model list is *merged*, never replaced: the file is machine-global, so
    two lhht runs using different models concurrently (say glm-5.3 and
    glm-5.3-flash) each re-register on every episode, and a replace here made
    the last writer evict the other run's model — its next ``session/create``
    then died with ``Provider Registry 中不存在 Model``. Models of runs that
    no longer exist stay registered; that is harmless (an id is only consulted
    when requested) and self-heals if the user ever resets the file.
    """
    model_id = (model_id or "").strip()
    if not model_id:
        raise ValueError("ZCode provider config needs a non-empty model id")
    config_path = Path(path) if path else default_provider_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
    else:
        payload = {}

    if not isinstance(payload, dict) or payload.get("schemaVersion") != 1:
        payload = {"schemaVersion": 1}
    config = payload.get("config")
    if not isinstance(config, dict):
        config = {}

    rules = config.setdefault("providerConfigRules", {})
    if not isinstance(rules, dict):
        rules = {}
        config["providerConfigRules"] = rules
    provider_rules = rules.get("providerRules")
    if not isinstance(provider_rules, list):
        provider_rules = []
        rules["providerRules"] = provider_rules
    existing_model_ids: list[str] = []
    for rule in provider_rules:
        if isinstance(rule, dict) and rule.get("providerId") == PROVIDER_ID:
            rule_config = rule.get("config")
            ids = (
                rule_config.get("personalModelIds")
                if isinstance(rule_config, dict)
                else None
            )
            if isinstance(ids, list):
                existing_model_ids = [
                    m for m in ids if isinstance(m, str) and m.strip() and m != model_id
                ]
            break
    new_rule = _provider_rule(
        api_key, base_url, [model_id, *existing_model_ids]
    )
    for index, rule in enumerate(provider_rules):
        if isinstance(rule, dict) and rule.get("providerId") == PROVIDER_ID:
            provider_rules[index] = new_rule
            break
    else:
        provider_rules.append(new_rule)

    model_rules = config.setdefault("modelConfigRules", {})
    if not isinstance(model_rules, dict):
        model_rules = {}
        config["modelConfigRules"] = model_rules
    manual_rules = model_rules.get("manualProviderModelRules")
    if not isinstance(manual_rules, list):
        manual_rules = []
        model_rules["manualProviderModelRules"] = manual_rules
    new_model_rule = _model_rule(model_id)
    for index, rule in enumerate(manual_rules):
        if (
            isinstance(rule, dict)
            and rule.get("providerId") == PROVIDER_ID
            and rule.get("modelId") == model_id
        ):
            manual_rules[index] = new_model_rule
            break
    else:
        manual_rules.append(new_model_rule)

    payload["config"] = config
    temp_path = config_path.with_name(f"{config_path.name}.lhht.tmp")
    with open(temp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    try:
        import os

        os.replace(temp_path, config_path)
    except OSError:
        # Windows can refuse os.replace across watchers; fall back to rewrite.
        config_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp_path.unlink(missing_ok=True)
    return config_path
