from __future__ import annotations

from f8studio_server.errors import InvalidRequestError

from pathlib import Path
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal, cast

from .models import AgentImage, AgentProviderSummary
from .provider_settings import (AgentProtocol, CreateProviderConnection, ProviderConfig,
                                ProviderSettingsStore, ProviderSettingsView, UpdateProviderSettings)
from .provider_probe import ProbeProviderRequest, ProviderProbeResult, probe_provider


class AgentProviderRegistry:
    def __init__(self, settings_path: Path | None = None) -> None:
        self._settings = ProviderSettingsStore(settings_path)

    def settings(self) -> tuple[ProviderSettingsView, ...]:
        return self._settings.views()

    def update_settings(self, provider_id: str, request: UpdateProviderSettings) -> ProviderSettingsView:
        return self._settings.update(provider_id, request)

    def create_connection(self, request: CreateProviderConnection) -> ProviderSettingsView:
        return self._settings.create(request)

    def delete_connection(self, provider_id: str) -> None:
        self._settings.delete(provider_id)

    async def probe(self, request: ProbeProviderRequest) -> ProviderProbeResult:
        saved_key = ""
        if request.provider_id:
            config, protocol = self._connection(request.provider_id)
            endpoint = self._settings.view(request.provider_id).endpoint
            if protocol == request.protocol and endpoint == request.endpoint.strip().rstrip("/"):
                saved_key = config.api_key
        return await probe_provider(request, saved_api_key=saved_key)

    def _connection(self, provider_id: str) -> tuple[ProviderConfig, AgentProtocol]:
        config = self._settings.get(provider_id)
        setting = self._settings.view(provider_id)
        if setting.protocol is None:
            raise InvalidRequestError(f"provider is not a conversational agent: {provider_id}")
        return config, setting.protocol

    def decision_config(self, provider_id: str) -> ProviderConfig:
        setting = next((item for item in self.settings() if item.provider_id == provider_id), None)
        if setting is None or setting.kind != "decision":
            raise InvalidRequestError("Provider does not support typed decisions")
        config = self._settings.get(provider_id)
        if not config.model or not config.endpoint or (provider_id == "typesafe" and not config.api_key):
            raise InvalidRequestError("Configure the decision provider in Studio Settings before evaluating decisions")
        return config

    def summaries(self) -> tuple[AgentProviderSummary, ...]:
        return (
            AgentProviderSummary(
                provider_id="deterministic", display_name="Deterministic graph agent",
                models=("graph-builder-v1",), configured=True, deterministic=True,
            ),
            *(AgentProviderSummary(
                provider_id=setting.provider_id, display_name=setting.display_name,
                models=setting.models, configured=setting.configured,
                supports_images=any(item.image_input is True for item in setting.model_capabilities),
                model_capabilities=setting.model_capabilities,
            ) for setting in self.settings() if setting.kind == "agent"),
        )

    def supports_image(self, provider_id: str, model_id: str) -> bool:
        setting = next((item for item in self.settings() if item.provider_id == provider_id), None)
        if setting is None:
            return False
        capability = next((item for item in setting.model_capabilities if item.model_id == model_id), None)
        if capability is not None and capability.image_input is not None:
            return capability.image_input
        return False

    def validate_selection(self, provider_id: str, model_id: str) -> None:
        summary = next((provider for provider in self.summaries() if provider.provider_id == provider_id), None)
        if summary is None:
            if provider_id in {"openai", "anthropic", "google_gemini", "ollama"}:
                raise InvalidRequestError(f"agent provider is not configured: {provider_id}")
            raise InvalidRequestError(f"unknown agent provider: {provider_id}")
        if not summary.configured:
            raise InvalidRequestError(f"agent provider is not configured: {provider_id}")
        if not model_id or len(model_id) > 256 or any(character.isspace() for character in model_id):
            raise InvalidRequestError("model ID must be non-empty, at most 256 characters, and contain no whitespace")
        if summary.deterministic and model_id not in summary.models:
            raise InvalidRequestError(f"unknown model for provider {provider_id}: {model_id}")

    async def complete(self, *, provider_id: str, model_id: str, prompt: str) -> str:
        self.validate_selection(provider_id, model_id)
        if provider_id == "deterministic":
            return prompt
        return await self._run_model(
            provider_id=provider_id, model_id=model_id, prompt=prompt,
            instructions=self._instructions(), max_tokens=1024,
        )

    async def run_with_tools(
        self,
        *,
        provider_id: str,
        model_id: str,
        prompt: str,
        tools: Sequence[Callable[..., Awaitable[str]]],
        images: Sequence[AgentImage] = (),
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
    ) -> str:
        self.validate_selection(provider_id, model_id)
        if provider_id == "deterministic":
            raise InvalidRequestError("the deterministic provider does not support model tool calls")
        if images and not self.supports_image(provider_id, model_id):
            raise InvalidRequestError(f"agent provider does not support image input: {provider_id}")
        return await self._run_model(
            provider_id=provider_id, model_id=model_id, prompt=prompt, tools=tools,
            images=images, reasoning_effort=reasoning_effort,
            instructions=self._tool_instructions(), max_tokens=4096,
        )

    async def _run_model(
        self, *, provider_id: str, model_id: str, prompt: str,
        instructions: str, max_tokens: int,
        tools: Sequence[Callable[..., Awaitable[str]]] = (),
        images: Sequence[AgentImage] = (),
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
    ) -> str:
        config, protocol = self._connection(provider_id)
        try:
            from agent_framework import Agent, AgentResponse, Content, Message
        except ModuleNotFoundError as exc:
            raise RuntimeError("Agent Framework dependencies are not installed") from exc

        input_message = Message(role="user", contents=[
            Content.from_text(prompt),
            *(Content.from_uri(image.data_url) for image in images),
        ]) if images else prompt

        if protocol == "openai_responses":
            from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions

            agent = Agent(
                OpenAIChatClient(model=model_id, api_key=config.api_key, base_url=config.endpoint or None),
                name="f8studio-agent", instructions=instructions, tools=tools,
            )
            options: OpenAIChatOptions[None] = {"max_tokens": max_tokens, "store": False}
            if reasoning_effort is not None:
                options["reasoning"] = {"effort": reasoning_effort}
            response = cast(AgentResponse[None], await agent.run(input_message, options=options))
            return response.text
        elif protocol == "anthropic":
            from agent_framework.anthropic import AnthropicChatOptions, AnthropicClient

            agent = Agent(
                AnthropicClient(model=model_id, api_key=config.api_key, base_url=config.endpoint or None),
                name="f8studio-agent", instructions=instructions, tools=tools,
            )
            anthropic_options: AnthropicChatOptions[None] = {"max_tokens": max_tokens}
            if reasoning_effort is not None:
                anthropic_options = cast(AnthropicChatOptions[None], {
                    "max_tokens": max_tokens, "output_config": {"effort": reasoning_effort},
                })
            response = cast(AgentResponse[None], await agent.run(input_message, options=anthropic_options))
            return response.text
        elif protocol == "openai_chat":
            from agent_framework.openai import OpenAIChatCompletionClient, OpenAIChatCompletionOptions

            agent = Agent(
                OpenAIChatCompletionClient(model=model_id, api_key=config.api_key or "local", base_url=config.endpoint),
                name="f8studio-agent", instructions=instructions, tools=tools,
            )
            chat_options: OpenAIChatCompletionOptions[None] = {"max_tokens": max_tokens}
            if reasoning_effort is not None:
                chat_options = cast(OpenAIChatCompletionOptions[None], {
                    "max_tokens": max_tokens, "reasoning_effort": reasoning_effort,
                })
            response = cast(AgentResponse[None], await agent.run(input_message, options=chat_options))
            return response.text

        raise InvalidRequestError(f"Protocol does not support conversational agents: {protocol}")

    @staticmethod
    def _tool_instructions() -> str:
        return (
            "You operate on the selected Feel8 Studio project through the supplied tools. "
            "List available skills and read the relevant graph, code, or game skill before acting. "
            "Read the current graph and search the catalog for relevant operators. Inspect exact operator specs and "
            "ports before editing. For graph construction use graph_propose_changes with compact node definitions, "
            "port-name connections, and state updates; the server builds the full patch. "
            "Then call graph_apply_proposal: that tool displays the approval UI and waits for the user. "
            "Do not end a graph-editing task after inspection or ask for approval in prose; prepare the proposal "
            "and invoke the approval tool within the same run. catalog_create_node only returns a template and "
            "does not add a node to the graph. "
            "Choose reasonable defaults for unspecified sampling rates and explain them; do not stop for routine choices. "
            "For Python code, read the exact node, analyze the proposed code, then write using its revision "
            "and content hash. Inspect deployment, logs, and monitor evidence before claiming success. "
            "Never claim a game installation or runtime behavior was verified without tool evidence. "
            "Discover installed extension tools, skills, and resources for game workflows. "
            "Use the generic extension execution tool and its approval flow; only offer operations provided by enabled extensions."
        )

    @staticmethod
    def _instructions() -> str:
        return (
            "Summarize the supplied Feel8 Studio tool evidence. State exactly what changed, "
            "whether validation and deployment succeeded, and cite graph revisions."
        )


__all__ = ["AgentProviderRegistry"]
