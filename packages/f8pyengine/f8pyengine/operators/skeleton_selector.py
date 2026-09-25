from __future__ import annotations

from typing import Any

from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry
from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    array_schema,
    boolean_schema,
    complex_object_schema,
    integer_schema,
    string_schema,
)

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.skeleton_selector"


class SkeletonSelectorRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[port.name for port in (node.dataInPorts or [])],
            data_out_ports=[port.name for port in (node.dataOutPorts or [])],
            state_fields=[state.name for state in (node.stateFields or [])],
        )
        state = dict(initial_state or {})
        self._profile_id = str(state.get("profileId") or "").strip()
        self._role = str(state.get("role") or "").strip().lower()
        self._role_index = _integer_or_default(state.get("roleIndex"), 0)
        self._fallback_model_name = str(state.get("fallbackModelName") or "").strip()
        self._allow_legacy_fallback = _boolean_or_default(state.get("allowLegacyFallback"), True)
        self._available_keys: list[str] = []
        self._available_keys_synced = False

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name == "profileId":
            self._profile_id = str(value or "").strip()
        elif name == "role":
            self._role = str(value or "").strip().lower()
        elif name == "roleIndex":
            self._role_index = _integer_or_default(value, self._role_index)
        elif name == "fallbackModelName":
            self._fallback_model_name = str(value or "").strip()
        elif name == "allowLegacyFallback":
            self._allow_legacy_fallback = _boolean_or_default(value, self._allow_legacy_fallback)

    async def validate_state(
        self,
        field: str,
        value: Any,
        *,
        ts_ms: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == "roleIndex":
            role_index = _integer_or_default(value, -1)
            if role_index < 0 or role_index > 1024:
                raise ValueError("roleIndex must be in range 0..1024")
            return role_index
        if name == "allowLegacyFallback":
            return _boolean_or_default(value, False)
        if name == "role":
            return str(value or "").strip().lower()
        if name in {"profileId", "fallbackModelName"}:
            return str(value or "").strip()
        return value

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        port_name = str(port or "").strip()
        if port_name not in {"skeleton", "stableKey", "status"}:
            return None
        raw_skeletons = await self.pull("skeletons", ctx_id=ctx_id)
        candidates = _skeleton_candidates(raw_skeletons)
        available_keys = [_stable_key(candidate) for candidate in candidates]
        await self._sync_available_keys(available_keys)
        selected = self._select(candidates)
        if selected is None:
            if port_name == "status":
                return {
                    "valid": False,
                    "stableKey": "",
                    "profileId": self._profile_id,
                    "role": self._role,
                    "roleIndex": self._role_index,
                    "reason": "no_matching_skeleton",
                }
            return None
        stable_key = _stable_key(selected)
        if port_name == "skeleton":
            return selected
        if port_name == "stableKey":
            return stable_key
        return {
            "valid": True,
            "stableKey": stable_key,
            "profileId": _identity_text(selected, "profileId"),
            "role": _identity_text(selected, "role").lower(),
            "roleIndex": _identity_integer(selected, "roleIndex", -1),
            "reason": "stable_identity" if _has_stable_identity(selected) else "legacy_model_name",
        }

    def _select(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        for candidate in candidates:
            if not _has_stable_identity(candidate):
                continue
            profile_matches = not self._profile_id or _identity_text(candidate, "profileId").casefold() == self._profile_id.casefold()
            role_matches = not self._role or _identity_text(candidate, "role").casefold() == self._role.casefold()
            index_matches = _identity_integer(candidate, "roleIndex", -1) == self._role_index
            if profile_matches and role_matches and index_matches:
                return candidate
        if self._allow_legacy_fallback and self._fallback_model_name:
            for candidate in candidates:
                if str(candidate.get("modelName") or "").strip() == self._fallback_model_name:
                    return candidate
        if not self._profile_id and not self._role and not self._fallback_model_name and len(candidates) == 1:
            return candidates[0]
        return None

    async def _sync_available_keys(self, available_keys: list[str]) -> None:
        if self._available_keys_synced and available_keys == self._available_keys:
            return
        self._available_keys = list(available_keys)
        self._available_keys_synced = True
        await self.set_state("availableKeys", list(available_keys))


def _skeleton_candidates(value: Any) -> list[dict[str, Any]]:
    raw_items = value if isinstance(value, list) else [value]
    candidates: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict) or not isinstance(item.get("bones"), list):
            continue
        candidates.append({str(key): field_value for key, field_value in item.items()})
    return candidates


def _trailer(skeleton: dict[str, Any]) -> dict[str, Any]:
    value = skeleton.get("trailer")
    if not isinstance(value, dict):
        return {}
    return {str(key): field_value for key, field_value in value.items()}


def _identity_text(skeleton: dict[str, Any], field_name: str) -> str:
    return str(_trailer(skeleton).get(field_name) or "").strip()


def _identity_integer(skeleton: dict[str, Any], field_name: str, default: int) -> int:
    return _integer_or_default(_trailer(skeleton).get(field_name), default)


def _has_stable_identity(skeleton: dict[str, Any]) -> bool:
    return bool(
        _identity_text(skeleton, "profileId")
        and _identity_text(skeleton, "role")
        and _identity_integer(skeleton, "roleIndex", -1) >= 0
    )


def _stable_key(skeleton: dict[str, Any]) -> str:
    explicit = str(skeleton.get("stableKey") or "").strip()
    if explicit:
        return explicit
    if _has_stable_identity(skeleton):
        profile_id = _identity_text(skeleton, "profileId")
        role = _identity_text(skeleton, "role").lower()
        role_index = _identity_integer(skeleton, "roleIndex", -1)
        return f"{profile_id}:{role}:{role_index}"
    return str(skeleton.get("modelName") or "").strip()


def _integer_or_default(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _boolean_or_default(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


SkeletonSelectorRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.motion",
    operatorClass=OPERATOR_CLASS,
    version="0.1.0",
    label="Skeleton Selector",
    description="Select a character by stable exporter profile, role, and role index.",
    tags=["skeleton", "character", "stable", "select", "unity"],
    dataInPorts=[F8DataPortSpec(name="skeletons", description="Decoded skeleton list.", valueSchema=any_schema())],
    dataOutPorts=[
        F8DataPortSpec(name="skeleton", description="Selected skeleton.", valueSchema=any_schema()),
        F8DataPortSpec(name="stableKey", description="Stable profile/role/index key.", valueSchema=string_schema()),
        F8DataPortSpec(
            name="status",
            description="Selection status on the data channel.",
            valueSchema=complex_object_schema(
                properties={
                    "valid": boolean_schema(),
                    "stableKey": string_schema(),
                    "profileId": string_schema(),
                    "role": string_schema(),
                    "roleIndex": integer_schema(),
                    "reason": string_schema(),
                }
            ),
        ),
    ],
    stateFields=[
        F8StateSpec(
            name="profileId",
            label="Profile ID",
            description="Exporter game profile ID. Empty accepts any profile.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="role",
            label="Role",
            description="Stable character role.",
            valueSchema=string_schema(default="", enum=["", "male", "female", "other"]),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="roleIndex",
            label="Role Index",
            description="Zero-based index within the selected role.",
            valueSchema=integer_schema(default=0, minimum=0, maximum=1024),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="fallbackModelName",
            label="Legacy Model",
            description="Exact modelName used only for LMEX v1 streams.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="allowLegacyFallback",
            label="Legacy Fallback",
            description="Allow exact modelName fallback for LMEX v1 packets.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="availableKeys",
            label="Available Characters",
            description="Low-frequency list of currently available stable keys.",
            valueSchema=array_schema(items=string_schema()),
            access=F8StateAccess.ro,
            valueRequired=True,
            showOnNode=False,
        ),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(SkeletonSelectorRuntimeNode.SPEC, SkeletonSelectorRuntimeNode, overwrite=True)
    return registry


__all__ = ["SkeletonSelectorRuntimeNode", "register_operator"]
