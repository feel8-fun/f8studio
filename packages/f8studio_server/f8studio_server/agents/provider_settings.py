from __future__ import annotations

from f8studio_server.errors import InvalidRequestError

import os
import tempfile
from pathlib import Path
from threading import RLock
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

import msgspec

AgentProtocol = Literal["openai_responses", "openai_chat", "anthropic", "systemone"]
_BUILTIN_PROTOCOLS: dict[str, AgentProtocol] = {
    "openai": "openai_responses", "anthropic": "anthropic",
    "google_gemini": "openai_chat", "ollama": "openai_chat",
    "typesafe": "systemone", "systemone_local": "systemone",
}


class ModelCapabilities(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    model_id: str
    image_input: bool | None = None
    thinking: bool | None = None
    source: Literal["catalog", "legacy"] = "catalog"
    image_source: Literal["catalog", "legacy", "manual"] | None = None
    thinking_source: Literal["catalog", "legacy", "manual"] | None = None


class ProviderConfig(msgspec.Struct, frozen=True, kw_only=True, rename="camel", forbid_unknown_fields=True):
    model: str
    endpoint: str = ""
    api_key: str = ""
    display_name: str = ""
    protocol: AgentProtocol | None = None
    models: tuple[str, ...] = ()
    model_capabilities: tuple[ModelCapabilities, ...] = ()
    disabled: bool = False


class UpdateProviderSettings(msgspec.Struct, frozen=True, kw_only=True, rename="camel", forbid_unknown_fields=True):
    model: str
    endpoint: str = ""
    api_key: str | None = None
    clear_api_key: bool = False
    display_name: str = ""
    models: tuple[str, ...] = ()
    model_capabilities: tuple[ModelCapabilities, ...] | None = None


class CreateProviderConnection(msgspec.Struct, frozen=True, kw_only=True, rename="camel", forbid_unknown_fields=True):
    display_name: str
    protocol: AgentProtocol
    endpoint: str
    model: str = ""
    models: tuple[str, ...] = ()
    api_key: str = ""
    model_capabilities: tuple[ModelCapabilities, ...] = ()


class ProviderSettingsView(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    provider_id: str
    display_name: str
    model: str
    endpoint: str
    api_key_set: bool
    requires_api_key: bool
    configured: bool
    source: str
    kind: Literal["agent", "decision"] = "agent"
    input_modalities: tuple[Literal["text", "image"], ...] = ("text",)
    supports_image: bool = False
    protocol: AgentProtocol | None = None
    models: tuple[str, ...] = ()
    custom: bool = False
    model_capabilities: tuple[ModelCapabilities, ...] = ()


_PROVIDER_NAMES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google_gemini": "Google Gemini",
    "ollama": "Ollama (local)",
    "typesafe": "TypeSafe Jev (decisions)",
    "systemone_local": "System-One compatible host",
}


class ProviderSettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = RLock()
        self._defaults = {
            "openai": ProviderConfig(
                model=os.environ.get("F8STUDIO_OPENAI_MODEL", "gpt-5.4-mini").strip(),
                endpoint=os.environ.get("F8STUDIO_OPENAI_ENDPOINT", "").strip().rstrip("/"),
                api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
            ),
            "anthropic": ProviderConfig(
                model=os.environ.get("F8STUDIO_ANTHROPIC_MODEL", "claude-sonnet-4-5").strip(),
                endpoint=os.environ.get("F8STUDIO_ANTHROPIC_ENDPOINT", "").strip().rstrip("/"),
                api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip(),
            ),
            "google_gemini": ProviderConfig(
                model=os.environ.get("F8STUDIO_GEMINI_MODEL", "gemini-2.5-pro-preview-03-25").strip(),
                endpoint=os.environ.get("F8STUDIO_GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/openai").strip().rstrip("/"),
                api_key=(os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()),
            ),
            "ollama": ProviderConfig(
                model=os.environ.get("F8STUDIO_OLLAMA_MODEL", "").strip(),
                endpoint=os.environ.get("F8STUDIO_OLLAMA_ENDPOINT", "http://127.0.0.1:11434/v1").strip().rstrip("/"),
            ),
            "typesafe": ProviderConfig(
                model=os.environ.get("F8STUDIO_TYPESAFE_MODEL", "jev-latest").strip(),
                endpoint=os.environ.get("F8STUDIO_TYPESAFE_ENDPOINT", "https://api.typesafe.ai/v1").strip().rstrip("/"),
                api_key=os.environ.get("TYPESAFE_API_KEY", "").strip(),
            ),
            "systemone_local": ProviderConfig(
                model=os.environ.get("F8STUDIO_SYSTEMONE_MODEL", "local-decision-model").strip(),
                endpoint=os.environ.get("F8STUDIO_SYSTEMONE_ENDPOINT", "http://127.0.0.1:8001/v1").strip().rstrip("/"),
                api_key=os.environ.get("F8STUDIO_SYSTEMONE_API_KEY", "").strip(),
                models=(os.environ.get("F8STUDIO_SYSTEMONE_MODEL", "local-decision-model").strip(),),
                model_capabilities=(ModelCapabilities(
                    model_id=os.environ.get("F8STUDIO_SYSTEMONE_MODEL", "local-decision-model").strip(),
                    image_input=True, image_source="manual",
                ),) if os.environ.get("F8STUDIO_SYSTEMONE_SUPPORTS_IMAGE", "").strip().lower() in {"1", "true", "yes"} else (),
            ),
        }
        self._saved: dict[str, ProviderConfig] = {}
        if path is not None and path.exists():
            self._saved = msgspec.json.decode(path.read_bytes(), type=dict[str, ProviderConfig])
            for provider_id, config in self._saved.items():
                self._validate(provider_id, config)

    def get(self, provider_id: str) -> ProviderConfig:
        with self._lock:
            if provider_id not in self._defaults and provider_id not in self._saved:
                raise InvalidRequestError("Unknown configurable agent provider")
            if provider_id in self._saved:
                config = self._saved[provider_id]
                if config.disabled:
                    raise InvalidRequestError("Provider connection was deleted")
                return config
            return self._defaults[provider_id]

    def views(self) -> tuple[ProviderSettingsView, ...]:
        with self._lock:
            builtin = (provider_id for provider_id in _PROVIDER_NAMES
                      if (provider_id in self._saved or self._environment_configured(provider_id))
                      and not (provider_id in self._saved and self._saved[provider_id].disabled)
                      and self.view(provider_id).configured)
            custom = (key for key in self._saved if key.startswith("connection_"))
            return tuple(self.view(provider_id) for provider_id in (*builtin, *custom))

    def _environment_configured(self, provider_id: str) -> bool:
        config = self._defaults[provider_id]
        return bool(config.api_key or (provider_id == "ollama" and os.environ.get("F8STUDIO_OLLAMA_MODEL"))
                    or (provider_id == "systemone_local" and
                        (os.environ.get("F8STUDIO_SYSTEMONE_MODEL") or os.environ.get("F8STUDIO_SYSTEMONE_ENDPOINT"))))

    def view(self, provider_id: str) -> ProviderSettingsView:
        config = self.get(provider_id)
        custom = provider_id.startswith("connection_")
        requires_key = provider_id not in {"ollama", "systemone_local"} and (not custom or config.protocol not in {"openai_chat", "systemone"})
        models = tuple(dict.fromkeys((config.model, *config.models))) if config.model else config.models
        capabilities = list(config.model_capabilities)
        default_capability = next((item for item in capabilities if item.model_id == config.model), None)
        image_supported = (default_capability.image_input is True if default_capability is not None
                           else False)
        return ProviderSettingsView(
            provider_id=provider_id,
            display_name=config.display_name if custom else _PROVIDER_NAMES[provider_id],
            model=config.model,
            endpoint=config.endpoint or {
                "openai": "https://api.openai.com/v1",
                "anthropic": "https://api.anthropic.com/v1",
            }.get(provider_id, ""),
            api_key_set=bool(config.api_key),
            requires_api_key=requires_key,
            configured=bool(config.model and (config.endpoint or provider_id in {"openai", "anthropic"}) and (config.api_key or not requires_key)),
            source="saved" if provider_id in self._saved else "environment",
            kind="decision" if provider_id in {"typesafe", "systemone_local"} or config.protocol == "systemone" else "agent",
            input_modalities=("text", "image") if image_supported else ("text",),
            supports_image=image_supported,
            protocol=config.protocol if custom else _BUILTIN_PROTOCOLS.get(provider_id),
            models=models,
            custom=custom,
            model_capabilities=tuple(capabilities),
        )

    @staticmethod
    def _validate(provider_id: str, config: ProviderConfig) -> None:
        custom = provider_id.startswith("connection_")
        if provider_id not in _PROVIDER_NAMES and not custom:
            raise InvalidRequestError("Unknown configurable agent provider")
        if config.disabled:
            if custom or config != ProviderConfig(model="", disabled=True):
                raise InvalidRequestError("Invalid deleted provider marker")
            return
        if (not config.model and not custom) or len(config.model) > 256 or any(character.isspace() for character in config.model):
            raise InvalidRequestError("Model ID must be non-empty, at most 256 characters, and contain no whitespace")
        if (custom or provider_id in {"google_gemini", "ollama", "typesafe", "systemone_local"}) and not config.endpoint:
            raise InvalidRequestError("This provider requires an endpoint URL")
        if custom and (config.protocol not in {"openai_responses", "openai_chat", "anthropic", "systemone"}
                       or not config.display_name.strip() or len(config.display_name) > 80):
            raise InvalidRequestError("Custom connections require a protocol and a name of at most 80 characters")
        if len(config.models) > 500 or any(not model or len(model) > 256 or any(char.isspace() for char in model) for model in config.models):
            raise InvalidRequestError("Model list contains an invalid ID or exceeds 500 models")
        if len(config.model_capabilities) > 500 or any(item.model_id not in config.models for item in config.model_capabilities):
            raise InvalidRequestError("Model capabilities must refer to saved models")
        if len({item.model_id for item in config.model_capabilities}) != len(config.model_capabilities):
            raise InvalidRequestError("Model capabilities contain duplicate model IDs")
        if config.endpoint:
            endpoint = urlsplit(config.endpoint)
            if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname
                    or endpoint.username is not None or endpoint.password is not None
                    or endpoint.query or endpoint.fragment or any(character.isspace() for character in config.endpoint)):
                raise InvalidRequestError("Endpoint must be an HTTP(S) URL without credentials, query parameters, or fragments")
        if len(config.api_key) > 8192 or any(character.isspace() for character in config.api_key):
            raise InvalidRequestError("API key must contain no whitespace and be at most 8192 characters")

    @staticmethod
    def _models(default_model: str, models: tuple[str, ...]) -> tuple[str, ...]:
        default_model = default_model.strip()
        return tuple(dict.fromkeys((default_model, *models))) if default_model else tuple(dict.fromkeys(models))

    def update(self, provider_id: str, request: UpdateProviderSettings) -> ProviderSettingsView:
        with self._lock:
            current = self.get(provider_id)
            if request.clear_api_key and request.api_key:
                raise InvalidRequestError("Cannot replace and clear an API key at the same time")
            config = ProviderConfig(
                model=request.model.strip(),
                endpoint=request.endpoint.strip().rstrip("/"),
                api_key="" if request.clear_api_key else (request.api_key.strip() if request.api_key else current.api_key),
                display_name=request.display_name.strip() if provider_id.startswith("connection_") else "",
                protocol=current.protocol,
                models=self._models(request.model, request.models),
                model_capabilities=(tuple(item for item in current.model_capabilities
                                          if item.model_id in self._models(request.model, request.models))
                                    if request.model_capabilities is None else request.model_capabilities),
            )
            self._validate(provider_id, config)
            saved = {**self._saved, provider_id: config}
            self._persist(saved)
            self._saved = saved
            return self.view(provider_id)

    def create(self, request: CreateProviderConnection) -> ProviderSettingsView:
        with self._lock:
            provider_id = f"connection_{uuid4().hex}"
            config = ProviderConfig(
                display_name=request.display_name.strip(), protocol=request.protocol,
                endpoint=request.endpoint.strip().rstrip("/"), api_key=request.api_key.strip(),
                model=request.model.strip(), models=self._models(request.model, request.models),
                model_capabilities=request.model_capabilities,
            )
            self._validate(provider_id, config)
            saved = {**self._saved, provider_id: config}
            self._persist(saved)
            self._saved = saved
            return self.view(provider_id)

    def delete(self, provider_id: str) -> None:
        with self._lock:
            if provider_id.startswith("connection_"):
                if provider_id not in self._saved:
                    raise InvalidRequestError("Unknown provider connection")
                saved = {key: value for key, value in self._saved.items() if key != provider_id}
            elif provider_id in _PROVIDER_NAMES:
                if provider_id in self._saved and self._saved[provider_id].disabled:
                    raise InvalidRequestError("Provider connection was deleted")
                if provider_id not in self._saved and not self._environment_configured(provider_id):
                    raise InvalidRequestError("Provider connection is not configured")
                saved = {**self._saved, provider_id: ProviderConfig(model="", disabled=True)}
            else:
                raise InvalidRequestError("Unknown provider connection")
            self._persist(saved)
            self._saved = saved

    def _persist(self, saved: dict[str, ProviderConfig]) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(dir=self._path.parent, delete=False)
        temporary_path = Path(handle.name)
        try:
            with handle:
                _ = handle.write(msgspec.json.encode(saved))
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(self._path)
        finally:
            temporary_path.unlink(missing_ok=True)
