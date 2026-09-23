from __future__ import annotations

import argparse
import asyncio
import os

from f8studio_server.agents.providers import AgentProviderRegistry


async def run_probe(model_id: str) -> None:
    registry = AgentProviderRegistry()
    response = await registry.complete(
        provider_id="openai",
        model_id=model_id,
        prompt=(
            "Summarize this Feel8 Studio diagnostic evidence in one sentence: "
            '{"graphRevision":3,"valid":true,"nodeCount":2,"deployment":"succeeded"}'
        ),
    )
    if not response.strip():
        raise RuntimeError("OpenAI agent provider returned an empty response")
    print(f"OpenAI agent smoke passed: model={model_id}; response_chars={len(response)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the conditional Web Studio OpenAI provider smoke test")
    parser.add_argument("--model", default=os.environ.get("F8STUDIO_OPENAI_MODEL", "gpt-5.4-mini"))
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        print("OpenAI agent smoke skipped: OPENAI_API_KEY is not configured")
        return
    asyncio.run(run_probe(args.model))


if __name__ == "__main__":
    main()
