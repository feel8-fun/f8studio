"""One-time migration of JSON authoring documents to explicit payload contracts.

Usage: pixi run python scripts/migrate_authoring_contracts.py path/to/document.json ...
The runtime intentionally does not perform this migration on reads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


def migrate_port(port: dict[str, Any]) -> None:
    schema = port.pop("valueSchema", None)
    kind = port.pop("payloadKind", None)
    delivery = port.pop("delivery", None)
    if "payload" not in port:
        if kind is None or kind == "json":
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            if {"width", "height", "pitch", "frameId", "tsMs"} <= props.keys():
                kind = "video_frame"
            elif {"sampleRate", "channels", "frames", "seq", "tsMs"} <= props.keys():
                kind = "audio_chunk"
            else:
                kind = "json"
        port["payload"] = {"kind": kind, "valueSchema" if kind == "json" else "metadataSchema": schema or {"type": "any"}}
    elif kind is not None and kind != port["payload"]["kind"]:
        raise ValueError(f"conflicting payload kinds on port {port.get('name')!r}")
    if delivery is not None:
        if "stream" in port and port["stream"].get("delivery", delivery) != delivery:
            raise ValueError(f"conflicting delivery on port {port.get('name')!r}")
        port.setdefault("stream", {})["delivery"] = delivery
    elif "stream" not in port and port["payload"]["kind"] in {"video_frame", "audio_chunk"}:
        port["stream"] = {"delivery": "latest"}


def migrate_descriptor(item: dict[str, Any], *, required_key: str) -> None:
    if "required" in item:
        item[required_key] = item.pop("required")
    if "uiControl" in item:
        raw = item.pop("uiControl")
        if raw and "control" not in item:
            match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[([^\]]+)\])?", raw)
            if match is None:
                raise ValueError(f"invalid UI control: {raw!r}")
            kind, argument = match.groups()
            if kind in {"wave_preview", "wave_pattern_editor", "wave_heatmap"}:
                control = {"kind": "custom", "rendererKey": kind}
            else:
                kind = {"wrapline": "textarea", "dropdown": "select", "dropbox": "select", "combo": "select", "combobox": "select"}.get(kind.lower(), kind.lower())
                control = {"kind": kind}
                if argument:
                    if kind in {"select", "multiselect"}:
                        control["optionsFromState"] = argument
                    elif kind in {"code", "textarea"}:
                        control["language"] = argument
                    else:
                        raise ValueError(f"control {kind!r} cannot take an argument")
            item["control"] = control
    policy = item.get("editPolicy")
    if isinstance(policy, dict) and "canEditRequired" in policy:
        policy["canEditValueRequired"] = policy.pop("canEditRequired")


def migrate(value: Any) -> None:
    """Visit only document/spec containers, never application state or JSON schemas."""
    if not isinstance(value, dict):
        raise ValueError("authoring document must be an object")
    if "serviceClass" in value or "operatorClass" in value:
        if value.get("schemaVersion") == "f8service/1":
            value.pop("launch", None)
        for key, required_key in (("stateFields", "valueRequired"), ("dataInPorts", "definitionProtected"), ("dataOutPorts", "definitionProtected"), ("commands", "definitionProtected")):
            for item in value.get(key, []):
                migrate_descriptor(item, required_key=required_key)
                if key in {"dataInPorts", "dataOutPorts"}:
                    migrate_port(item)
                if key == "commands":
                    for param in item.get("params", []):
                        migrate_descriptor(param, required_key="valueRequired")
        if value.get("schemaVersion") == "f8operator/1":
            for key in ("execInPorts", "execOutPorts"):
                if key in value:
                    value[key] = [{"name": item} if isinstance(item, str) else item for item in value[key]]
    for key in ("service", "spec", "graph"):
        if isinstance(value.get(key), dict):
            migrate(value[key])
    for key in ("operators", "nodes"):
        items = value.get(key)
        if isinstance(items, list):
            for item in items:
                migrate(item)
    for port in value.get("ports", []):
        if isinstance(port.get("dataSpec"), dict):
            migrate_descriptor(port["dataSpec"], required_key="definitionProtected")
            migrate_port(port["dataSpec"])
    if value.get("format") == "f8graph":
        _migrate_exchange_definitions(value)


def _migrate_exchange_definitions(value: dict[str, Any]) -> None:
    import msgspec
    from f8pysdk.specs import F8ServiceSpec, F8OperatorSpec
    from f8studio_core.graph.exchange import _definition_ref, import_graph

    if value.get("formatVersion") != 3:
        raise ValueError("only f8graph formatVersion 3 can be migrated")
    for kind in ("services", "operators"):
        definitions: dict[str, Any] = {}
        references: dict[str, str] = {}
        for old_ref, spec in value["definitions"][kind].items():
            migrate(spec)
            typed = msgspec.convert(spec, type=F8ServiceSpec) if kind == "services" else msgspec.convert(spec, type=F8OperatorSpec)
            new_ref = _definition_ref(typed)
            references[old_ref] = new_ref
            definitions[new_ref] = msgspec.to_builtins(typed)
        value["definitions"][kind] = definitions
        for instance in value[kind].values():
            instance["definitionRef"] = references[instance["definitionRef"]]
    import_graph(msgspec.json.encode(value))


def migrate_providers(providers: dict[str, Any]) -> None:
    """Move the old default-model image flag into its model capability once."""
    for provider_id, config in providers.items():
        supported = config.pop("supportsImage", False)
        model = config["model"]
        models = list(dict.fromkeys(([model] if model else []) + config.get("models", [])))
        config["models"] = models
        capabilities = config.setdefault("modelCapabilities", [])
        if not supported or not model or not (provider_id.startswith("connection_") or provider_id == "systemone_local"):
            continue
        capability = next((item for item in capabilities if item["modelId"] == model), None)
        if capability is None:
            capability = {"modelId": model, "source": "legacy"}
            capabilities.append(capability)
        if capability.get("imageInput") is None:
            capability["imageInput"] = True
            capability["imageSource"] = "legacy"
            if capability.get("thinking") is not None:
                capability.setdefault("thinkingSource", capability.get("source", "catalog"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", action="store_true", help="Migrate a provider settings file instead of a graph/describe document")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        value = json.loads(path.read_text())
        if args.providers:
            migrate_providers(value)
        else:
            migrate(value)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
