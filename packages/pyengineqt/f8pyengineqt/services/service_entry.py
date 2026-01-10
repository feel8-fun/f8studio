from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class F8ServiceDescribeEntryLaunch(BaseModel):
    """
    Minimal launch definition used for service discovery.

    This intentionally mirrors `F8ServiceLaunchSpec` (from f8pysdk) but is kept
    separate because discovery entries are not full service specs.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    command: str = Field(..., description="Executable path or name")
    args: list[str] = []
    env: dict[str, str] = {}
    workdir: str = "./"


class F8ServiceDescribeEntry(BaseModel):
    """
    Discovery entry stored in `service.yml`.

    - `serviceClass` is recommended (for stability/caching), but optional.
    - Full `F8ServiceSpec` + operator specs are fetched via `{launch} --describe`.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    schemaVersion: Literal["f8serviceEntry/1"] = "f8serviceEntry/1"

    serviceClass: str | None = Field(
        None, description="Recommended stable id. If omitted, must be provided by --describe output."
    )
    label: str | None = None
    version: str | None = None

    launch: F8ServiceDescribeEntryLaunch

    describeArgs: list[str] = Field(default_factory=lambda: ["--describe"])
    timeoutMs: int = 4000


class F8DescribePayload(BaseModel):
    """
    JSON payload returned by `{entrypoint} --describe`.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    schemaVersion: Literal["f8describe/1"] | None = None
    service: dict[str, Any]
    operators: list[dict[str, Any]] = []

