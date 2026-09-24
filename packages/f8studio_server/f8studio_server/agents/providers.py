from __future__ import annotations

import os
from typing import cast

from .models import AgentProviderSummary


class AgentProviderRegistry:
    def __init__(self) -> None:
        self._openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self._openai_model = os.environ.get("F8STUDIO_OPENAI_MODEL", "gpt-5.4-mini").strip()
        self._openai_endpoint = os.environ.get("F8STUDIO_OPENAI_ENDPOINT", "").strip().rstrip("/")
        self._anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        self._anthropic_model = os.environ.get(
            "F8STUDIO_ANTHROPIC_MODEL", "claude-sonnet-4-5"
        ).strip()
        self._anthropic_endpoint = os.environ.get(
            "F8STUDIO_ANTHROPIC_ENDPOINT", ""
        ).strip().rstrip("/")
        self._gemini_api_key = (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )
        self._gemini_model = os.environ.get(
            "F8STUDIO_GEMINI_MODEL", "gemini-2.5-pro-preview-03-25"
        ).strip()
        self._gemini_endpoint = os.environ.get(
            "F8STUDIO_GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/openai"
        ).strip().rstrip("/")
        self._ollama_model = os.environ.get("F8STUDIO_OLLAMA_MODEL", "").strip()
        self._ollama_endpoint = os.environ.get(
            "F8STUDIO_OLLAMA_ENDPOINT", "http://127.0.0.1:11434/v1"
        ).strip().rstrip("/")

    def summaries(self) -> tuple[AgentProviderSummary, ...]:
        return (
            AgentProviderSummary(
                provider_id="deterministic",
                display_name="Deterministic graph agent",
                models=("graph-builder-v1",),
                configured=True,
                deterministic=True,
            ),
            AgentProviderSummary(
                provider_id="openai",
                display_name="OpenAI",
                models=(self._openai_model,) if self._openai_model else (),
                configured=bool(self._openai_api_key and self._openai_model),
            ),
            AgentProviderSummary(
                provider_id="anthropic",
                display_name="Anthropic",
                models=(self._anthropic_model,) if self._anthropic_model else (),
                configured=bool(self._anthropic_api_key and self._anthropic_model),
            ),
            AgentProviderSummary(
                provider_id="google_gemini",
                display_name="Google Gemini",
                models=(self._gemini_model,) if self._gemini_model else (),
                configured=bool(self._gemini_api_key and self._gemini_model and self._gemini_endpoint),
            ),
            AgentProviderSummary(
                provider_id="ollama",
                display_name="Ollama (local)",
                models=(self._ollama_model,) if self._ollama_model else (),
                configured=bool(self._ollama_model and self._ollama_endpoint),
            ),
        )

    def validate_selection(self, provider_id: str, model_id: str) -> None:
        summary = next((provider for provider in self.summaries() if provider.provider_id == provider_id), None)
        if summary is None:
            raise ValueError(f"unknown agent provider: {provider_id}")
        if not summary.configured:
            raise ValueError(f"agent provider is not configured: {provider_id}")
        if model_id not in summary.models:
            raise ValueError(f"unknown model for provider {provider_id}: {model_id}")

    async def complete(self, *, provider_id: str, model_id: str, prompt: str) -> str:
        self.validate_selection(provider_id, model_id)
        if provider_id == "deterministic":
            return prompt
        if provider_id == "openai":
            return await self._complete_openai(model_id=model_id, prompt=prompt)
        if provider_id == "anthropic":
            return await self._complete_anthropic(model_id=model_id, prompt=prompt)
        if provider_id == "google_gemini":
            return await self._complete_openai_compatible(
                model_id=model_id,
                prompt=prompt,
                api_key=self._gemini_api_key,
                base_url=self._gemini_endpoint,
            )
        if provider_id == "ollama":
            return await self._complete_openai_compatible(
                model_id=model_id,
                prompt=prompt,
                api_key="ollama",
                base_url=self._ollama_endpoint,
            )
        raise ValueError(f"unsupported agent provider: {provider_id}")

    @staticmethod
    def _instructions() -> str:
        return (
            "Summarize the supplied Feel8 Studio tool evidence. State exactly what changed, "
            "whether validation and deployment succeeded, and cite graph revisions."
        )

    async def _complete_openai(self, *, model_id: str, prompt: str) -> str:
        try:
            from agent_framework import Agent, AgentResponse
            from agent_framework.openai import OpenAIChatClient
        except ModuleNotFoundError as exc:
            raise RuntimeError("OpenAI Agent Framework dependencies are not installed") from exc

        client = OpenAIChatClient(
            model=model_id,
            api_key=self._openai_api_key,
            base_url=self._openai_endpoint or None,
        )
        agent = Agent(
            client,
            name="f8studio-agent",
            instructions=self._instructions(),
        )
        response = cast(
            AgentResponse[None],
            await agent.run(prompt, options={"store": False, "max_tokens": 1024}),
        )
        return response.text

    async def _complete_openai_compatible(
        self,
        *,
        model_id: str,
        prompt: str,
        api_key: str,
        base_url: str,
    ) -> str:
        try:
            from agent_framework import Agent, AgentResponse
            from agent_framework.openai import OpenAIChatCompletionClient
        except ModuleNotFoundError as exc:
            raise RuntimeError("OpenAI-compatible Agent Framework dependencies are not installed") from exc

        client = OpenAIChatCompletionClient(
            model=model_id,
            api_key=api_key,
            base_url=base_url,
        )
        agent = Agent(
            client,
            name="f8studio-agent",
            instructions=self._instructions(),
        )
        response = cast(
            AgentResponse[None],
            await agent.run(prompt, options={"max_tokens": 1024}),
        )
        return response.text

    async def _complete_anthropic(self, *, model_id: str, prompt: str) -> str:
        try:
            from agent_framework import Agent, AgentResponse
            from agent_framework.anthropic import AnthropicClient
        except ModuleNotFoundError as exc:
            raise RuntimeError("Anthropic Agent Framework dependencies are not installed") from exc

        client = AnthropicClient(
            model=model_id,
            api_key=self._anthropic_api_key,
            base_url=self._anthropic_endpoint or None,
        )
        agent = Agent(
            client,
            name="f8studio-agent",
            instructions=self._instructions(),
        )
        response = cast(
            AgentResponse[None],
            await agent.run(prompt, options={"max_tokens": 1024}),
        )
        return response.text


__all__ = ["AgentProviderRegistry"]
