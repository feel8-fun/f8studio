from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LaunchConfig:
    command: str
    args: list[str]
    env: dict[str, str]
    workdir: str | None


@dataclass(frozen=True)
class ServiceEntry:
    service_class: str
    label: str
    version: str
    launch: LaunchConfig
    source_dir: Path


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    description: str
    access: str
    required: bool
    show_on_node: bool
    value_schema: dict[str, Any]


@dataclass(frozen=True)
class CommandParamSpec:
    name: str
    description: str
    required: bool
    value_schema: dict[str, Any]


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    show_on_node: bool
    params: list[CommandParamSpec]


@dataclass(frozen=True)
class PortSpec:
    name: str
    description: str
    required: bool
    show_on_node: bool
    value_schema: dict[str, Any]


@dataclass(frozen=True)
class OperatorSpec:
    operator_class: str
    label: str
    description: str
    exec_in_ports: list[str]
    exec_out_ports: list[str]
    state_fields: list[FieldSpec]
    data_in_ports: list[PortSpec]
    data_out_ports: list[PortSpec]


@dataclass(frozen=True)
class ServiceDescribe:
    service_class: str
    label: str
    description: str
    tags: list[str]
    state_fields: list[FieldSpec]
    commands: list[CommandSpec]
    data_in_ports: list[PortSpec]
    data_out_ports: list[PortSpec]
    operators: list[OperatorSpec]


@dataclass(frozen=True)
class ServiceDocBundle:
    entry: ServiceEntry
    describe: ServiceDescribe
    slug: str
    manual_text: str


def _as_dict(value: Any, *, ctx: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{ctx}: expected object, got {type(value).__name__}")
    return value


def _as_list(value: Any, *, ctx: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{ctx}: expected list, got {type(value).__name__}")
    return value


def _req_str(obj: dict[str, Any], key: str, *, ctx: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{ctx}: missing or invalid '{key}'")
    return value.strip()


def _opt_str(obj: dict[str, Any], key: str, *, default: str = "") -> str:
    value = obj.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _opt_bool(obj: dict[str, Any], key: str, *, default: bool = False) -> bool:
    value = obj.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"invalid bool for '{key}'")


def _parse_launch(raw: Any, *, ctx: str) -> LaunchConfig:
    launch_obj = _as_dict(raw, ctx=f"{ctx}.launch")
    command = _req_str(launch_obj, "command", ctx=f"{ctx}.launch")

    args_raw = _as_list(launch_obj.get("args"), ctx=f"{ctx}.launch.args")
    args: list[str] = []
    for index, value in enumerate(args_raw):
        if isinstance(value, str):
            args.append(value)
        else:
            args.append(str(value))

    env_obj = _as_dict(launch_obj.get("env") or {}, ctx=f"{ctx}.launch.env")
    env: dict[str, str] = {}
    for key, value in env_obj.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{ctx}.launch.env: invalid env key")
        env[key] = str(value)

    workdir_value = launch_obj.get("workdir")
    workdir: str | None
    if workdir_value is None:
        workdir = None
    elif isinstance(workdir_value, str):
        workdir = workdir_value.strip() or None
    else:
        workdir = str(workdir_value)

    return LaunchConfig(command=command, args=args, env=env, workdir=workdir)


def _parse_field(field_obj: dict[str, Any], *, ctx: str) -> FieldSpec:
    value_schema = _as_dict(field_obj.get("valueSchema") or {}, ctx=f"{ctx}.valueSchema")
    return FieldSpec(
        name=_req_str(field_obj, "name", ctx=ctx),
        label=_opt_str(field_obj, "label", default=""),
        description=_opt_str(field_obj, "description", default=""),
        access=_opt_str(field_obj, "access", default="rw") or "rw",
        required=_opt_bool(field_obj, "required", default=False),
        show_on_node=_opt_bool(field_obj, "showOnNode", default=False),
        value_schema=value_schema,
    )


def _parse_command_param(param_obj: dict[str, Any], *, ctx: str) -> CommandParamSpec:
    value_schema = _as_dict(param_obj.get("valueSchema") or {}, ctx=f"{ctx}.valueSchema")
    return CommandParamSpec(
        name=_req_str(param_obj, "name", ctx=ctx),
        description=_opt_str(param_obj, "description", default=""),
        required=_opt_bool(param_obj, "required", default=False),
        value_schema=value_schema,
    )


def _parse_command(command_obj: dict[str, Any], *, ctx: str) -> CommandSpec:
    params_raw = _as_list(command_obj.get("params"), ctx=f"{ctx}.params")
    params: list[CommandParamSpec] = []
    for index, param_raw in enumerate(params_raw):
        param_obj = _as_dict(param_raw, ctx=f"{ctx}.params[{index}]")
        params.append(_parse_command_param(param_obj, ctx=f"{ctx}.params[{index}]"))

    return CommandSpec(
        name=_req_str(command_obj, "name", ctx=ctx),
        description=_opt_str(command_obj, "description", default=""),
        show_on_node=_opt_bool(command_obj, "showOnNode", default=False),
        params=params,
    )


def _parse_port(port_obj: dict[str, Any], *, ctx: str) -> PortSpec:
    value_schema = _as_dict(port_obj.get("valueSchema") or {}, ctx=f"{ctx}.valueSchema")
    return PortSpec(
        name=_req_str(port_obj, "name", ctx=ctx),
        description=_opt_str(port_obj, "description", default=""),
        required=_opt_bool(port_obj, "required", default=False),
        show_on_node=_opt_bool(port_obj, "showOnNode", default=False),
        value_schema=value_schema,
    )


def _parse_operator(operator_obj: dict[str, Any], *, ctx: str) -> OperatorSpec:
    exec_in_raw = _as_list(operator_obj.get("execInPorts"), ctx=f"{ctx}.execInPorts")
    exec_out_raw = _as_list(operator_obj.get("execOutPorts"), ctx=f"{ctx}.execOutPorts")

    exec_in_ports: list[str] = []
    for index, value in enumerate(exec_in_raw):
        if not isinstance(value, str):
            raise ValueError(f"{ctx}.execInPorts[{index}]: expected string")
        exec_in_ports.append(value)

    exec_out_ports: list[str] = []
    for index, value in enumerate(exec_out_raw):
        if not isinstance(value, str):
            raise ValueError(f"{ctx}.execOutPorts[{index}]: expected string")
        exec_out_ports.append(value)

    state_fields_raw = _as_list(operator_obj.get("stateFields"), ctx=f"{ctx}.stateFields")
    state_fields: list[FieldSpec] = []
    for index, field_raw in enumerate(state_fields_raw):
        field_obj = _as_dict(field_raw, ctx=f"{ctx}.stateFields[{index}]")
        state_fields.append(_parse_field(field_obj, ctx=f"{ctx}.stateFields[{index}]"))

    data_in_raw = _as_list(operator_obj.get("dataInPorts"), ctx=f"{ctx}.dataInPorts")
    data_in_ports: list[PortSpec] = []
    for index, port_raw in enumerate(data_in_raw):
        port_obj = _as_dict(port_raw, ctx=f"{ctx}.dataInPorts[{index}]")
        data_in_ports.append(_parse_port(port_obj, ctx=f"{ctx}.dataInPorts[{index}]"))

    data_out_raw = _as_list(operator_obj.get("dataOutPorts"), ctx=f"{ctx}.dataOutPorts")
    data_out_ports: list[PortSpec] = []
    for index, port_raw in enumerate(data_out_raw):
        port_obj = _as_dict(port_raw, ctx=f"{ctx}.dataOutPorts[{index}]")
        data_out_ports.append(_parse_port(port_obj, ctx=f"{ctx}.dataOutPorts[{index}]"))

    return OperatorSpec(
        operator_class=_req_str(operator_obj, "operatorClass", ctx=ctx),
        label=_opt_str(operator_obj, "label", default=""),
        description=_opt_str(operator_obj, "description", default=""),
        exec_in_ports=exec_in_ports,
        exec_out_ports=exec_out_ports,
        state_fields=state_fields,
        data_in_ports=data_in_ports,
        data_out_ports=data_out_ports,
    )


def _slugify_service_class(service_class: str) -> str:
    return service_class.strip().replace(".", "-")


def _load_service_entry(service_dir: Path) -> ServiceEntry:
    service_path = service_dir / "service.yml"
    if not service_path.exists():
        raise ValueError(f"missing service.yml for service directory: {service_dir}")

    raw = yaml.safe_load(service_path.read_text(encoding="utf-8"))
    service_obj = _as_dict(raw, ctx=str(service_path))

    service_class = _req_str(service_obj, "serviceClass", ctx=str(service_path))
    label = _req_str(service_obj, "label", ctx=str(service_path))
    version = _req_str(service_obj, "version", ctx=str(service_path))
    launch = _parse_launch(service_obj.get("launch"), ctx=str(service_path))

    return ServiceEntry(
        service_class=service_class,
        label=label,
        version=version,
        launch=launch,
        source_dir=service_dir,
    )


def _load_service_describe(service_dir: Path) -> ServiceDescribe:
    describe_path = service_dir / "describe.json"
    if not describe_path.exists():
        raise ValueError(f"missing describe.json for service directory: {service_dir}")

    try:
        raw = json.loads(describe_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {describe_path}: {exc}") from exc

    root_obj = _as_dict(raw, ctx=str(describe_path))
    service_obj = _as_dict(root_obj.get("service"), ctx=f"{describe_path}.service")

    state_fields_raw = _as_list(service_obj.get("stateFields"), ctx=f"{describe_path}.service.stateFields")
    state_fields: list[FieldSpec] = []
    for index, field_raw in enumerate(state_fields_raw):
        field_obj = _as_dict(field_raw, ctx=f"{describe_path}.service.stateFields[{index}]")
        state_fields.append(_parse_field(field_obj, ctx=f"{describe_path}.service.stateFields[{index}]"))

    commands_raw = _as_list(service_obj.get("commands"), ctx=f"{describe_path}.service.commands")
    commands: list[CommandSpec] = []
    for index, command_raw in enumerate(commands_raw):
        command_obj = _as_dict(command_raw, ctx=f"{describe_path}.service.commands[{index}]")
        commands.append(_parse_command(command_obj, ctx=f"{describe_path}.service.commands[{index}]"))

    data_in_raw = _as_list(service_obj.get("dataInPorts"), ctx=f"{describe_path}.service.dataInPorts")
    data_in_ports: list[PortSpec] = []
    for index, port_raw in enumerate(data_in_raw):
        port_obj = _as_dict(port_raw, ctx=f"{describe_path}.service.dataInPorts[{index}]")
        data_in_ports.append(_parse_port(port_obj, ctx=f"{describe_path}.service.dataInPorts[{index}]"))

    data_out_raw = _as_list(service_obj.get("dataOutPorts"), ctx=f"{describe_path}.service.dataOutPorts")
    data_out_ports: list[PortSpec] = []
    for index, port_raw in enumerate(data_out_raw):
        port_obj = _as_dict(port_raw, ctx=f"{describe_path}.service.dataOutPorts[{index}]")
        data_out_ports.append(_parse_port(port_obj, ctx=f"{describe_path}.service.dataOutPorts[{index}]"))

    operators_raw = _as_list(root_obj.get("operators"), ctx=f"{describe_path}.operators")
    operators: list[OperatorSpec] = []
    for index, operator_raw in enumerate(operators_raw):
        operator_obj = _as_dict(operator_raw, ctx=f"{describe_path}.operators[{index}]")
        operators.append(_parse_operator(operator_obj, ctx=f"{describe_path}.operators[{index}]"))

    tags_raw = _as_list(service_obj.get("tags"), ctx=f"{describe_path}.service.tags")
    tags: list[str] = []
    for index, tag in enumerate(tags_raw):
        if not isinstance(tag, str):
            raise ValueError(f"{describe_path}.service.tags[{index}]: expected string")
        tags.append(tag)

    return ServiceDescribe(
        service_class=_req_str(service_obj, "serviceClass", ctx=f"{describe_path}.service"),
        label=_req_str(service_obj, "label", ctx=f"{describe_path}.service"),
        description=_opt_str(service_obj, "description", default=""),
        tags=tags,
        state_fields=state_fields,
        commands=commands,
        data_in_ports=data_in_ports,
        data_out_ports=data_out_ports,
        operators=operators,
    )


def _discover_service_dirs(services_root: Path) -> list[Path]:
    if not services_root.exists():
        raise ValueError(f"services root does not exist: {services_root}")

    service_dirs: list[Path] = []
    for service_path in sorted(services_root.rglob("service.yml")):
        service_dirs.append(service_path.parent)

    if not service_dirs:
        raise ValueError(f"no service.yml files found under: {services_root}")
    return service_dirs


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def _schema_summary(schema: dict[str, Any]) -> str:
    schema_type = schema.get("type")
    enum_value = schema.get("enum")
    default_value = schema.get("default")

    parts: list[str] = []
    if isinstance(schema_type, str) and schema_type:
        if schema_type == "array":
            items_obj = schema.get("items")
            if isinstance(items_obj, dict):
                item_type = items_obj.get("type")
                if isinstance(item_type, str) and item_type:
                    parts.append(f"array[{item_type}]")
                else:
                    parts.append("array")
            else:
                parts.append("array")
        elif schema_type == "object":
            properties_obj = schema.get("properties")
            if isinstance(properties_obj, dict) and properties_obj:
                property_names = sorted([name for name in properties_obj.keys() if isinstance(name, str)])
                if property_names:
                    preview = ", ".join(property_names[:4])
                    if len(property_names) > 4:
                        preview = f"{preview}, ..."
                    parts.append(f"object{{{preview}}}")
                else:
                    parts.append("object")
            else:
                parts.append("object")
        else:
            parts.append(schema_type)
    else:
        parts.append("any")

    if isinstance(enum_value, list) and enum_value:
        enum_preview = ", ".join(str(item) for item in enum_value[:5])
        if len(enum_value) > 5:
            enum_preview = f"{enum_preview}, ..."
        parts.append(f"enum[{enum_preview}]")

    if default_value is not None:
        parts.append(f"default={default_value}")

    return " / ".join(parts)


def _render_field_table(fields: list[FieldSpec]) -> str:
    if not fields:
        return "_None_\n"

    lines = [
        "| Name | Access | Required | On Node | Schema | Description |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for field in fields:
        lines.append(
            "| "
            f"`{_md_escape(field.name)}` | "
            f"`{_md_escape(field.access)}` | "
            f"`{str(field.required).lower()}` | "
            f"`{str(field.show_on_node).lower()}` | "
            f"`{_md_escape(_schema_summary(field.value_schema))}` | "
            f"{_md_escape(field.description or field.label)} |"
        )
    return "\n".join(lines) + "\n"


def _render_port_table(ports: list[PortSpec]) -> str:
    if not ports:
        return "_None_\n"

    lines = [
        "| Name | Required | On Node | Schema | Description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for port in ports:
        lines.append(
            "| "
            f"`{_md_escape(port.name)}` | "
            f"`{str(port.required).lower()}` | "
            f"`{str(port.show_on_node).lower()}` | "
            f"`{_md_escape(_schema_summary(port.value_schema))}` | "
            f"{_md_escape(port.description)} |"
        )
    return "\n".join(lines) + "\n"


def _render_command_table(commands: list[CommandSpec]) -> str:
    if not commands:
        return "_None_\n"

    lines: list[str] = []
    for command in commands:
        lines.append(f"### `{command.name}`")
        if command.description:
            lines.append(command.description)
        lines.append("")
        lines.append(f"- Show on node: `{str(command.show_on_node).lower()}`")
        if not command.params:
            lines.append("- Params: none")
            lines.append("")
            continue

        lines.append("")
        lines.append("| Param | Required | Schema | Description |")
        lines.append("| --- | --- | --- | --- |")
        for param in command.params:
            lines.append(
                "| "
                f"`{_md_escape(param.name)}` | "
                f"`{str(param.required).lower()}` | "
                f"`{_md_escape(_schema_summary(param.value_schema))}` | "
                f"{_md_escape(param.description)} |"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_launch_block(entry: ServiceEntry) -> str:
    args_text = " ".join(entry.launch.args)
    command_line = entry.launch.command
    if args_text:
        command_line = f"{command_line} {args_text}"

    lines = [
        "```bash",
        command_line,
        "```",
        "",
        f"- Workdir: `{entry.launch.workdir or '.'}`",
    ]

    if entry.launch.env:
        lines.append("- Environment overrides:")
        for key in sorted(entry.launch.env.keys()):
            lines.append(f"  - `{key}={entry.launch.env[key]}`")
    else:
        lines.append("- Environment overrides: none")

    return "\n".join(lines) + "\n"


def _render_operator_section(operators: list[OperatorSpec]) -> str:
    if not operators:
        return "## Operators\n\n_None_\n"

    parts: list[str] = ["## Operators", ""]
    for operator in operators:
        title = operator.label or operator.operator_class
        parts.append(f"### {title} (`{operator.operator_class}`)")
        parts.append(operator.description or "No description.")
        parts.append("")
        exec_in = ", ".join(f"`{name}`" for name in operator.exec_in_ports) or "none"
        exec_out = ", ".join(f"`{name}`" for name in operator.exec_out_ports) or "none"
        parts.append(f"- Exec in ports: {exec_in}")
        parts.append(f"- Exec out ports: {exec_out}")
        parts.append("")
        parts.append("#### State Fields")
        parts.append("")
        parts.append(_render_field_table(operator.state_fields).rstrip())
        parts.append("")
        parts.append("#### Data Input Ports")
        parts.append("")
        parts.append(_render_port_table(operator.data_in_ports).rstrip())
        parts.append("")
        parts.append("#### Data Output Ports")
        parts.append("")
        parts.append(_render_port_table(operator.data_out_ports).rstrip())
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def _render_service_page(bundle: ServiceDocBundle, services_root: Path) -> str:
    entry = bundle.entry
    describe = bundle.describe

    source_rel = entry.source_dir.relative_to(services_root)
    title = f"# {describe.label or entry.label} (`{entry.service_class}`)"

    tags_text = ", ".join(f"`{tag}`" for tag in describe.tags) if describe.tags else "none"

    parts: list[str] = [
        "<!-- AUTO-GENERATED by scripts/generate_service_docs.py. DO NOT EDIT DIRECTLY. -->",
        title,
        "",
        describe.description or "No description.",
        "",
        f"- Service class: `{entry.service_class}`",
        f"- Version: `{entry.version}`",
        f"- Source directory: `{source_rel.as_posix()}`",
        f"- Tags: {tags_text}",
        "",
        "## How to Run",
        "",
        _render_launch_block(entry).rstrip(),
        "",
        "## Service State Fields",
        "",
        _render_field_table(describe.state_fields).rstrip(),
        "",
        "## Service Commands",
        "",
        _render_command_table(describe.commands).rstrip(),
        "",
        "## Service Data Input Ports",
        "",
        _render_port_table(describe.data_in_ports).rstrip(),
        "",
        "## Service Data Output Ports",
        "",
        _render_port_table(describe.data_out_ports).rstrip(),
        "",
        _render_operator_section(describe.operators).rstrip(),
        "",
        "## Usage Guide (Manual)",
        "",
        bundle.manual_text.strip(),
        "",
    ]

    return "\n".join(parts)


def _family_name(service_class: str) -> str:
    if service_class.startswith("f8.cvkit."):
        return "CVKit"
    if service_class.startswith("f8.dl."):
        return "Deep Learning"
    if service_class.startswith("f8.mp."):
        return "MediaPipe"
    return "Core"


def _render_modules_index(bundles: list[ServiceDocBundle]) -> str:
    groups: dict[str, list[ServiceDocBundle]] = {"Core": [], "CVKit": [], "Deep Learning": [], "MediaPipe": []}
    for bundle in bundles:
        family = _family_name(bundle.entry.service_class)
        groups[family].append(bundle)

    lines: list[str] = [
        "<!-- AUTO-GENERATED by scripts/generate_service_docs.py. DO NOT EDIT DIRECTLY. -->",
        "# Modules Overview",
        "",
        "Service pages are generated from `services/**/service.yml` and `services/**/describe.json`.",
        "Manual usage guidance is merged from `docs/modules/manual/*.md`.",
        "",
    ]

    for family in ["Core", "CVKit", "Deep Learning", "MediaPipe"]:
        items = sorted(groups[family], key=lambda item: item.entry.service_class)
        if not items:
            continue
        lines.append(f"## {family}")
        lines.append("")
        lines.append("| Service | Label | Operators | State Fields | Link |")
        lines.append("| --- | --- | --- | --- | --- |")
        for bundle in items:
            operators_count = len(bundle.describe.operators)
            state_count = len(bundle.describe.state_fields)
            link = f"services/{bundle.slug}.md"
            lines.append(
                "| "
                f"`{bundle.entry.service_class}` | "
                f"{_md_escape(bundle.describe.label)} | "
                f"`{operators_count}` | "
                f"`{state_count}` | "
                f"[{bundle.describe.label}]({link}) |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _load_manual_text(manual_root: Path, slug: str, service_class: str) -> str:
    manual_path = manual_root / f"{slug}.md"
    if not manual_path.exists():
        raise ValueError(f"missing manual section for {service_class}: {manual_path}")
    text = manual_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty manual section for {service_class}: {manual_path}")
    return text


def _build_bundles(services_root: Path, manual_root: Path) -> list[ServiceDocBundle]:
    bundles: list[ServiceDocBundle] = []
    service_dirs = _discover_service_dirs(services_root)

    for service_dir in service_dirs:
        entry = _load_service_entry(service_dir)
        describe = _load_service_describe(service_dir)
        if entry.service_class != describe.service_class:
            raise ValueError(
                "serviceClass mismatch: "
                f"entry={entry.service_class} describe={describe.service_class} path={service_dir}"
            )
        slug = _slugify_service_class(entry.service_class)
        manual_text = _load_manual_text(manual_root, slug, entry.service_class)
        bundles.append(
            ServiceDocBundle(entry=entry, describe=describe, slug=slug, manual_text=manual_text)
        )

    return sorted(bundles, key=lambda item: item.entry.service_class)


def _build_expected_files(
    bundles: list[ServiceDocBundle],
    *,
    services_root: Path,
    output_root: Path,
    index_path: Path,
) -> dict[Path, str]:
    expected: dict[Path, str] = {}
    for bundle in bundles:
        output_path = output_root / f"{bundle.slug}.md"
        expected[output_path] = _render_service_page(bundle, services_root)
    expected[index_path] = _render_modules_index(bundles)
    return expected


def _check_mode(expected: dict[Path, str], output_root: Path) -> int:
    mismatches: list[str] = []

    for path, content in expected.items():
        if not path.exists():
            mismatches.append(f"MISSING: {path}")
            continue
        current = path.read_text(encoding="utf-8")
        if current != content:
            mismatches.append(f"DIFFERS: {path}")

    existing_generated = sorted(output_root.glob("*.md"))
    expected_generated = {path for path in expected.keys() if path.parent == output_root}
    for existing_path in existing_generated:
        if existing_path not in expected_generated:
            mismatches.append(f"STALE: {existing_path}")

    if mismatches:
        print("documentation is out of date:")
        for mismatch in mismatches:
            print(f"- {mismatch}")
        return 1

    print("documentation is up to date")
    return 0


def _write_files(expected: dict[Path, str], output_root: Path) -> None:
    for path, content in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    existing_generated = sorted(output_root.glob("*.md"))
    expected_generated = {path for path in expected.keys() if path.parent == output_root}
    for existing_path in existing_generated:
        if existing_path not in expected_generated:
            existing_path.unlink()


def build_docs(
    *,
    services_root: Path,
    output_root: Path,
    manual_root: Path,
    index_path: Path,
    check: bool,
) -> int:
    bundles = _build_bundles(services_root, manual_root)
    expected = _build_expected_files(
        bundles,
        services_root=services_root,
        output_root=output_root,
        index_path=index_path,
    )

    if check:
        return _check_mode(expected, output_root)

    _write_files(expected, output_root)
    print(f"generated {len(bundles)} service page(s) and modules index")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate docs/modules service pages from service metadata")
    parser.add_argument("--services-root", default="services", help="Root containing service directories")
    parser.add_argument("--output-root", default="docs/modules/services", help="Output directory for generated pages")
    parser.add_argument("--manual-root", default="docs/modules/manual", help="Directory with per-service manual sections")
    parser.add_argument("--index-path", default="docs/modules/index.md", help="Output path for generated module index")
    parser.add_argument("--check", action="store_true", help="Validate generated files without writing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return build_docs(
            services_root=Path(args.services_root).resolve(),
            output_root=Path(args.output_root).resolve(),
            manual_root=Path(args.manual_root).resolve(),
            index_path=Path(args.index_path).resolve(),
            check=bool(args.check),
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
