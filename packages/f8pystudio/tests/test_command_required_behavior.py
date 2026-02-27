from __future__ import annotations

from qtpy import QtWidgets

from f8pysdk import F8Command, F8ServiceSpec
from f8pystudio.widgets.f8_spec_ops import delete_command
from f8pystudio.widgets.node_property_widgets import _F8EditCommandDialog, _F8SpecCommandEditor


class _FakeModel:
    def __init__(self) -> None:
        self.f8_sys: dict[str, object] = {}


class _FakeNode:
    def __init__(self, spec: F8ServiceSpec) -> None:
        self.spec = spec
        self.model = _FakeModel()
        self.id = "svc.test"

    def effective_commands(self) -> list[F8Command]:
        return list(self.spec.commands or [])


def _ensure_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is not None:
        return app
    return QtWidgets.QApplication([])


def _make_spec(commands: list[F8Command]) -> F8ServiceSpec:
    return F8ServiceSpec(serviceClass="f8.test", label="Test", editableCommands=True, commands=commands)


def test_spec_delete_command_keeps_required() -> None:
    spec = _make_spec(
        [
            F8Command(name="required_cmd", required=True, params=[]),
            F8Command(name="optional_cmd", required=False, params=[]),
        ]
    )
    spec2 = delete_command(spec, name="required_cmd")
    names = [str(c.name or "") for c in list(spec2.commands or [])]
    assert names == ["required_cmd", "optional_cmd"]


def test_spec_delete_command_removes_optional() -> None:
    spec = _make_spec(
        [
            F8Command(name="required_cmd", required=True, params=[]),
            F8Command(name="optional_cmd", required=False, params=[]),
        ]
    )
    spec2 = delete_command(spec, name="optional_cmd")
    names = [str(c.name or "") for c in list(spec2.commands or [])]
    assert names == ["required_cmd"]


def test_edit_command_dialog_preserves_required_flag() -> None:
    _ensure_app()
    dialog = _F8EditCommandDialog(
        None,
        title="Edit command",
        cmd=F8Command(name="required_cmd", required=True, params=[]),
        ui_only=False,
    )
    edited = dialog.command()
    assert edited.required is True


def test_command_editor_hides_delete_for_required_command(monkeypatch) -> None:
    _ensure_app()
    spec = _make_spec(
        [
            F8Command(name="required_cmd", required=True, params=[]),
            F8Command(name="optional_cmd", required=False, params=[]),
        ]
    )
    node = _FakeNode(spec)
    editor = _F8SpecCommandEditor(None, node=node, on_apply=None)

    required_row = editor._cmd_rows["required_cmd"]
    optional_row = editor._cmd_rows["optional_cmd"]
    assert required_row._btn_del.isHidden() is True
    assert optional_row._btn_del.isHidden() is False

    asked = {"called": False}

    def _fake_question(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        asked["called"] = True
        return QtWidgets.QMessageBox.Yes

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", _fake_question)

    editor._delete_command("required_cmd")
    assert asked["called"] is False
    names = [str(c.name or "") for c in list(node.spec.commands or [])]
    assert names == ["required_cmd", "optional_cmd"]

    editor._delete_command("optional_cmd")
    assert asked["called"] is True
    names = [str(c.name or "") for c in list(node.spec.commands or [])]
    assert names == ["required_cmd"]
