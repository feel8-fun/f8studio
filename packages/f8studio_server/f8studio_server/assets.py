from __future__ import annotations

import enum
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

import msgspec

from f8pysdk.f8_naming import ensure_token
from f8pysdk.specs import F8JsonValue
from f8studio_core.graph import GraphEdge, NodeLayout, OperatorNode, ServiceNode, StudioDocument, validate_document
from f8studio_core.graph.models import GraphNode


ASSET_SCHEMA_VERSION = "f8studio-asset/1"
COMPONENT_SCHEMA_VERSION = "f8studio-component/1"
VARIANT_SCHEMA_VERSION = "f8studio-variant/1"


class AssetKind(str, enum.Enum):
    component = "component"
    variant = "variant"
    modding_recipe = "modding_recipe"


class ComponentContent(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    schema_version: str = COMPONENT_SCHEMA_VERSION
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    layout: tuple[NodeLayout, ...] = ()


class VariantContent(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    schema_version: str = VARIANT_SCHEMA_VERSION
    service_class: str
    operator_class: str | None = None
    state_values: dict[str, F8JsonValue] = msgspec.field(default_factory=dict)


class AssetSummary(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    asset_id: str
    kind: AssetKind
    name: str
    description: str
    tags: tuple[str, ...]
    current_version: int
    created_at: str
    updated_at: str


class AssetRecord(AssetSummary, frozen=True, kw_only=True, rename="camel"):
    content: F8JsonValue


class AssetVersion(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    asset_id: str
    version: int
    created_at: str
    content: F8JsonValue


class CreateAssetRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    kind: AssetKind
    name: str
    content: F8JsonValue
    description: str = ""
    tags: tuple[str, ...] = ()
    asset_id: str | None = None


class UpdateAssetRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    name: str
    content: F8JsonValue
    description: str = ""
    tags: tuple[str, ...] = ()


class AssetExport(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    schema_version: str
    asset: AssetRecord
    versions: tuple[AssetVersion, ...]


class ProjectVersion(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    version_id: str
    project_id: str
    name: str
    created_at: str
    document: StudioDocument


class CreateProjectVersionRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    name: str = "Snapshot"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected database text, got {type(value).__name__}")
    return value


def _integer(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError(f"expected database integer, got {type(value).__name__}")
    return value


def _bytes(value: object) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"expected database bytes, got {type(value).__name__}")
    return value


def _json(value: object) -> F8JsonValue:
    return cast(F8JsonValue, msgspec.to_builtins(value, str_keys=True))


def _normalize_tags(tags: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))


def _validate_content(kind: AssetKind, content: F8JsonValue) -> F8JsonValue:
    encoded = msgspec.json.encode(content)
    if kind is AssetKind.component:
        component = msgspec.json.decode(encoded, type=ComponentContent)
        if component.schema_version != COMPONENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported component schema: {component.schema_version}")
        node_ids = {node.node_id for node in component.nodes}
        if len(node_ids) != len(component.nodes):
            raise ValueError("component node ids must be unique")
        if any(edge.from_node_id not in node_ids or edge.to_node_id not in node_ids for edge in component.edges):
            raise ValueError("component edges must reference component nodes")
        layout_ids = [layout.node_id for layout in component.layout]
        if len(set(layout_ids)) != len(layout_ids):
            raise ValueError("component layout node ids must be unique")
        if any(node_id not in node_ids for node_id in layout_ids):
            raise ValueError("component layout must reference component nodes")
        service_classes = {
            node.service_id: node.service_class
            for node in component.nodes
            if isinstance(node, ServiceNode)
        }
        if len(service_classes) != sum(isinstance(node, ServiceNode) for node in component.nodes):
            raise ValueError("component service ids must be unique")
        for node in component.nodes:
            if not isinstance(node, OperatorNode):
                continue
            service_class = service_classes.get(node.service_id)
            if service_class is None:
                raise ValueError("component operators must reference component services")
            if service_class != node.service_class:
                raise ValueError("component operator and service classes must match")
        return _json(component)
    if kind is AssetKind.variant:
        variant = msgspec.json.decode(encoded, type=VariantContent)
        if variant.schema_version != VARIANT_SCHEMA_VERSION:
            raise ValueError(f"unsupported variant schema: {variant.schema_version}")
        if not variant.service_class.strip():
            raise ValueError("variant serviceClass must be non-empty")
        return _json(variant)
    if not isinstance(content, dict):
        raise ValueError("modding recipe content must be an object")
    schema = content.get("schemaVersion")
    if not isinstance(schema, str) or not schema.strip():
        raise ValueError("modding recipe content requires schemaVersion")
    return content


class AssetRepository:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path.resolve()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=10.0)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS local_assets (
                    asset_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    tags BLOB NOT NULL,
                    current_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS local_asset_versions (
                    asset_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    content BLOB NOT NULL,
                    PRIMARY KEY(asset_id, version),
                    FOREIGN KEY(asset_id) REFERENCES local_assets(asset_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS project_versions (
                    version_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    document BLOB NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS project_versions_project
                    ON project_versions(project_id, created_at DESC);
                """
            )

    def list_assets(self, kind: AssetKind | None = None) -> tuple[AssetSummary, ...]:
        query = """SELECT asset_id, kind, name, description, tags, current_version, created_at, updated_at
                   FROM local_assets"""
        params: tuple[str, ...] = ()
        if kind is not None:
            query += " WHERE kind = ?"
            params = (kind.value,)
        query += " ORDER BY lower(name), asset_id"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return tuple(self._summary(row) for row in rows)

    def create(self, request: CreateAssetRequest) -> AssetRecord:
        asset_id = ensure_token(request.asset_id or uuid4().hex, label="asset_id")
        name = request.name.strip()
        if not name:
            raise ValueError("asset name must be non-empty")
        content = _validate_content(request.kind, request.content)
        timestamp = _now()
        tags = _normalize_tags(request.tags)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """INSERT INTO local_assets(
                           asset_id, kind, name, description, tags, current_version, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, 1, ?, ?)""",
                    (asset_id, request.kind.value, name, request.description, msgspec.json.encode(tags), timestamp, timestamp),
                )
                connection.execute(
                    "INSERT INTO local_asset_versions(asset_id, version, created_at, content) VALUES (?, 1, ?, ?)",
                    (asset_id, timestamp, msgspec.json.encode(content)),
                )
        except sqlite3.IntegrityError as exc:
            raise FileExistsError(f"asset already exists: {asset_id}") from exc
        return self.get(asset_id)

    def get(self, asset_id: str) -> AssetRecord:
        asset_id = ensure_token(asset_id, label="asset_id")
        with self._connect() as connection:
            row = connection.execute(
                """SELECT a.asset_id, a.kind, a.name, a.description, a.tags, a.current_version,
                          a.created_at, a.updated_at, v.content
                   FROM local_assets a JOIN local_asset_versions v
                     ON v.asset_id = a.asset_id AND v.version = a.current_version
                   WHERE a.asset_id = ?""",
                (asset_id,),
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"asset not found: {asset_id}")
        summary = self._summary(row[:8])
        return AssetRecord(
            asset_id=summary.asset_id,
            kind=summary.kind,
            name=summary.name,
            description=summary.description,
            tags=summary.tags,
            current_version=summary.current_version,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
            content=msgspec.json.decode(_bytes(row[8]), type=F8JsonValue),
        )

    def update(self, asset_id: str, request: UpdateAssetRequest) -> AssetRecord:
        asset_id = ensure_token(asset_id, label="asset_id")
        name = request.name.strip()
        if not name:
            raise ValueError("asset name must be non-empty")
        timestamp = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT kind, current_version FROM local_assets WHERE asset_id = ?",
                (asset_id,),
            ).fetchone()
            if current is None:
                raise FileNotFoundError(f"asset not found: {asset_id}")
            content = _validate_content(AssetKind(_text(current[0])), request.content)
            version = _integer(current[1]) + 1
            connection.execute(
                """UPDATE local_assets SET name = ?, description = ?, tags = ?,
                       current_version = ?, updated_at = ? WHERE asset_id = ?""",
                (name, request.description, msgspec.json.encode(_normalize_tags(request.tags)), version, timestamp, asset_id),
            )
            connection.execute(
                "INSERT INTO local_asset_versions(asset_id, version, created_at, content) VALUES (?, ?, ?, ?)",
                (asset_id, version, timestamp, msgspec.json.encode(content)),
            )
        return self.get(asset_id)

    def delete(self, asset_id: str) -> None:
        asset_id = ensure_token(asset_id, label="asset_id")
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM local_assets WHERE asset_id = ?", (asset_id,))
            if cursor.rowcount != 1:
                raise FileNotFoundError(f"asset not found: {asset_id}")

    def versions(self, asset_id: str) -> tuple[AssetVersion, ...]:
        _ = self.get(asset_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT version, created_at, content FROM local_asset_versions WHERE asset_id = ? ORDER BY version DESC",
                (asset_id,),
            ).fetchall()
        return tuple(
            AssetVersion(
                asset_id=asset_id,
                version=_integer(row[0]),
                created_at=_text(row[1]),
                content=msgspec.json.decode(_bytes(row[2]), type=F8JsonValue),
            )
            for row in rows
        )

    def export(self, asset_id: str) -> AssetExport:
        return AssetExport(schema_version=ASSET_SCHEMA_VERSION, asset=self.get(asset_id), versions=self.versions(asset_id))

    def import_asset(self, payload: AssetExport) -> AssetRecord:
        if payload.schema_version != ASSET_SCHEMA_VERSION:
            raise ValueError(f"unsupported asset export schema: {payload.schema_version}")
        asset = payload.asset
        asset_id = ensure_token(asset.asset_id, label="asset_id")
        if not asset.name.strip():
            raise ValueError("asset name must be non-empty")
        versions = tuple(sorted(payload.versions, key=lambda item: item.version))
        if not versions or versions[-1].version != asset.current_version:
            raise ValueError("asset export does not contain its current version")
        if len({version.version for version in versions}) != len(versions):
            raise ValueError("asset export contains duplicate versions")
        for version in versions:
            if version.asset_id != asset_id or version.version < 1:
                raise ValueError("asset export version identity is invalid")
        normalized = tuple(
            (version, _validate_content(asset.kind, version.content)) for version in versions
        )
        if normalized[-1][1] != _validate_content(asset.kind, asset.content):
            raise ValueError("asset current content does not match current version")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """INSERT INTO local_assets(
                           asset_id, kind, name, description, tags, current_version, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        asset_id,
                        asset.kind.value,
                        asset.name.strip(),
                        asset.description,
                        msgspec.json.encode(_normalize_tags(asset.tags)),
                        asset.current_version,
                        asset.created_at,
                        asset.updated_at,
                    ),
                )
                connection.executemany(
                    "INSERT INTO local_asset_versions(asset_id, version, created_at, content) VALUES (?, ?, ?, ?)",
                    tuple(
                        (asset_id, version.version, version.created_at, msgspec.json.encode(content))
                        for version, content in normalized
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise FileExistsError(f"asset already exists or export is invalid: {asset_id}") from exc
        return self.get(asset_id)

    def create_project_version(self, project_id: str, name: str, document: StudioDocument) -> ProjectVersion:
        validate_document(document)
        if document.project_id != project_id:
            raise ValueError("version document projectId does not match route project id")
        clean_name = name.strip() or "Snapshot"
        version = ProjectVersion(
            version_id=uuid4().hex,
            project_id=project_id,
            name=clean_name,
            created_at=_now(),
            document=document,
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO project_versions(version_id, project_id, name, created_at, document) VALUES (?, ?, ?, ?, ?)",
                (version.version_id, project_id, version.name, version.created_at, msgspec.json.encode(document)),
            )
        return version

    def list_project_versions(self, project_id: str) -> tuple[ProjectVersion, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT version_id, name, created_at, document FROM project_versions WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        return tuple(
            ProjectVersion(
                version_id=_text(row[0]),
                project_id=project_id,
                name=_text(row[1]),
                created_at=_text(row[2]),
                document=msgspec.json.decode(_bytes(row[3]), type=StudioDocument),
            )
            for row in rows
        )

    def get_project_version(self, project_id: str, version_id: str) -> ProjectVersion:
        versions = self.list_project_versions(project_id)
        found = next((version for version in versions if version.version_id == version_id), None)
        if found is None:
            raise FileNotFoundError(f"project version not found: {version_id}")
        return found

    @staticmethod
    def _summary(row: tuple[object, ...]) -> AssetSummary:
        tags = msgspec.json.decode(_bytes(row[4]), type=tuple[str, ...])
        return AssetSummary(
            asset_id=_text(row[0]),
            kind=AssetKind(_text(row[1])),
            name=_text(row[2]),
            description=_text(row[3]),
            tags=tags,
            current_version=_integer(row[5]),
            created_at=_text(row[6]),
            updated_at=_text(row[7]),
        )


__all__ = [
    "ASSET_SCHEMA_VERSION",
    "AssetExport",
    "AssetKind",
    "AssetRecord",
    "AssetRepository",
    "AssetSummary",
    "AssetVersion",
    "ComponentContent",
    "CreateAssetRequest",
    "CreateProjectVersionRequest",
    "ProjectVersion",
    "UpdateAssetRequest",
    "VariantContent",
]
