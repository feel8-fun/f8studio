import os
import sys
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import pytest


PKG_PYDL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PKG_PYDL not in sys.path:
    sys.path.insert(0, PKG_PYDL)


from f8pydl.onnx_runtime import (  # noqa: E402
    OnnxYoloDetectorRuntime,
    _available_ort_providers,
    _choose_ort_providers_from_available,
    _create_ort_session,
)
from f8pydl.model_config import ModelSpec  # noqa: E402


@dataclass(frozen=True)
class _FakeSession:
    providers: list[str]


class _FakeOrt:
    def __init__(
        self,
        *,
        available_providers: list[str],
        fail_first_session: bool = False,
        fail_provider_query: bool = False,
    ) -> None:
        self._available_providers = available_providers
        self._fail_first_session = fail_first_session
        self._fail_provider_query = fail_provider_query
        self.session_providers: list[list[str]] = []

    def get_available_providers(self) -> list[str]:
        if self._fail_provider_query:
            raise RuntimeError("provider query failed")
        return list(self._available_providers)

    def InferenceSession(self, _model_path: str, *, providers: list[str]) -> _FakeSession:  # noqa: N802
        self.session_providers.append(list(providers))
        if self._fail_first_session and len(self.session_providers) == 1:
            raise RuntimeError("cuda unavailable")
        return _FakeSession(providers=list(providers))


class _ObbHarness:
    _obb_angle_unit = OnnxYoloDetectorRuntime._obb_angle_unit
    _cv2_box_points = OnnxYoloDetectorRuntime._cv2_box_points

    def __init__(self, *, meta: dict[str, Any] | None) -> None:
        self.spec = ModelSpec(
            model_id="m",
            display_name="M",
            provider="onnx",
            task="yolo_obb",
            onnx_path=Path("model.onnx"),
            input_width=640,
            input_height=640,
            conf_threshold=0.25,
            iou_threshold=0.45,
            classes=[],
            meta=meta,
        )


class _FakeCv2:
    class error(Exception):
        ...

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def boxPoints(self, box: tuple[tuple[float, float], tuple[float, float], float]) -> list[list[float]]:  # noqa: N802
        if self.fail:
            raise self.error("bad box")
        (cx, cy), (w, h), _angle = box
        return [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]


def test_choose_ort_providers_prefers_cuda_when_available() -> None:
    providers = _choose_ort_providers_from_available(
        ["CPUExecutionProvider", "CUDAExecutionProvider"],
        prefer="auto",
    )

    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_choose_ort_providers_uses_cpu_when_cuda_unavailable() -> None:
    providers = _choose_ort_providers_from_available(["CPUExecutionProvider"], prefer="cuda")

    assert providers == ["CPUExecutionProvider"]


def test_available_ort_providers_returns_empty_list_on_query_failure(caplog: pytest.LogCaptureFixture) -> None:
    ort = _FakeOrt(available_providers=[], fail_provider_query=True)

    caplog.set_level(logging.DEBUG, logger="f8pydl.onnx_runtime")
    assert _available_ort_providers(ort) == []
    assert "failed to query ONNX Runtime providers" in caplog.text


def test_create_ort_session_falls_back_to_cpu_for_auto_provider(caplog: pytest.LogCaptureFixture) -> None:
    ort = _FakeOrt(
        available_providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
        fail_first_session=True,
    )

    result = _create_ort_session(ort, "model.onnx", ort_provider="auto")

    assert isinstance(result.session, _FakeSession)
    assert result.session.providers == ["CPUExecutionProvider"]
    assert ort.session_providers == [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]
    assert "falling back to CPUExecutionProvider" in result.provider_warning
    assert "failed to initialize ONNX Runtime session" in caplog.text


def test_create_ort_session_keeps_cpu_provider_fail_fast() -> None:
    class _FailingCpuOrt(_FakeOrt):
        def InferenceSession(self, _model_path: str, *, providers: list[str]) -> Any:  # noqa: N802
            self.session_providers.append(list(providers))
            raise RuntimeError("cpu init failed")

    ort = _FailingCpuOrt(available_providers=["CPUExecutionProvider"])

    with pytest.raises(RuntimeError, match="cpu init failed"):
        _create_ort_session(ort, "model.onnx", ort_provider="cpu")

    assert ort.session_providers == [["CPUExecutionProvider"]]


def test_obb_angle_unit_reads_explicit_yolo_metadata() -> None:
    harness = _ObbHarness(meta={"yolo": {"angleUnit": "rad"}})

    assert harness._obb_angle_unit() == "rad"


def test_obb_angle_unit_uses_default_for_missing_metadata() -> None:
    harness = _ObbHarness(meta={"other": {"angleUnit": "rad"}})

    assert harness._obb_angle_unit() == "deg"


def test_cv2_box_points_returns_none_when_opencv_rejects_box() -> None:
    pts = OnnxYoloDetectorRuntime._cv2_box_points(
        _FakeCv2(fail=True),
        cx=1.0,
        cy=2.0,
        w=3.0,
        h=4.0,
        angle_deg=5.0,
    )

    assert pts is None
