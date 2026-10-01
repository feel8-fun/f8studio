from f8pyscript.script_runtime_values import build_script_output_ports, extract_script_outputs


def _ports(*names: str):
    return build_script_output_ports(str(name) for name in names)


def test_extract_script_outputs_preserves_explicit_output_keys() -> None:
    outputs = extract_script_outputs(
        {"outputs": {"out": 1, "extra": 2, 3: "three"}},
        ports=_ports("out"),
    )

    assert outputs == {"out": 1, "extra": 2, "3": "three"}


def test_extract_script_outputs_maps_plain_value_only_when_out_exists() -> None:
    assert extract_script_outputs("ok", ports=_ports("out")) == {"out": "ok"}
    assert extract_script_outputs("ok", ports=_ports("other")) == {}


def test_extract_script_outputs_rejects_invalid_outputs_field() -> None:
    try:
        extract_script_outputs({"outputs": 1}, ports=_ports("out"))
    except ValueError as exc:
        assert str(exc) == "script return field 'outputs' must be a dict"
        return
    raise AssertionError("expected ValueError")


def test_extract_script_outputs_rejects_dict_without_outputs_field() -> None:
    try:
        extract_script_outputs({"out": 1}, ports=_ports("out"))
    except ValueError as exc:
        assert str(exc) == "script dict return must include an 'outputs' dict"
        return
    raise AssertionError("expected ValueError")
