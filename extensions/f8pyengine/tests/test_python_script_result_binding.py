from f8pyengine.operators.script_utils.result_binding import ScriptOutputPorts, extract_script_outputs


def _ports(*names: str) -> ScriptOutputPorts:
    data_out_ports = frozenset(str(name) for name in names)
    single_data_out_port = str(names[0]) if len(names) == 1 else None
    return ScriptOutputPorts(
        data_out_ports=data_out_ports,
        single_data_out_port=single_data_out_port,
        has_out_port="out" in data_out_ports,
    )


def test_extract_script_outputs_filters_to_known_ports() -> None:
    outputs = extract_script_outputs(
        {"outputs": {"out": 1, "extra": 2, 3: "three"}},
        ports=_ports("out", "3"),
    )

    assert outputs == {"out": 1, "3": "three"}


def test_extract_script_outputs_maps_plain_value_to_out() -> None:
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
