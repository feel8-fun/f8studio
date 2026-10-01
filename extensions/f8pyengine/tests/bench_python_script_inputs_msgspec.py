from __future__ import annotations

import argparse
from time import perf_counter
from typing import Any

import msgspec


class _InputView:
    __slots__ = ("_data", "_attr_to_key")

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self._attr_to_key: dict[str, str] | None = None

    @staticmethod
    def _build_attr_to_key(data: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for raw_key in data.keys():
            key = str(raw_key or "")
            if key.isidentifier():
                out[key] = key
        return out

    @classmethod
    def _wrap_value(cls, value: Any) -> Any:
        if type(value) in (str, int, float, bool, type(None)):
            return value
        if isinstance(value, _InputView):
            return value
        if type(value) is dict:
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._wrap_value(item) for item in value)
        return value

    def __getattr__(self, name: str) -> Any:
        attr_to_key = self._attr_to_key
        if attr_to_key is None:
            attr_to_key = self._build_attr_to_key(self._data)
            self._attr_to_key = attr_to_key
        key = attr_to_key.get(name)
        if key is None:
            raise AttributeError(name)
        return self._wrap_value(self._data.get(key))


class Bone(msgspec.Struct, kw_only=True):
    name: str
    position: list[float]


class Msg(msgspec.Struct, kw_only=True):
    modelName: str | None = None
    bones: list[Bone]


class Inputs(msgspec.Struct, kw_only=True):
    msg: Msg


PAYLOAD = {
    "msg": {
        "modelName": "m",
        "bones": [
            {"name": "Head", "position": [0.0, 1.0, 2.0]},
            {"name": "Hips", "position": [3.0, 4.0, 5.0]},
            {"name": "Foot", "position": [6.0, 7.0, 8.0]},
        ],
    }
}


def _run_input_view(iterations: int) -> tuple[float, float]:
    t0 = perf_counter()
    acc = 0.0
    for _ in range(iterations):
        inputs = _InputView(PAYLOAD)
        for bone in inputs.msg.bones:
            if bone.name == "Hips":
                acc += float(bone.position[0])
                break
    elapsed = perf_counter() - t0
    return elapsed, acc


def _run_msgspec(iterations: int) -> tuple[float, float]:
    t0 = perf_counter()
    acc = 0.0
    for _ in range(iterations):
        inputs = msgspec.convert(PAYLOAD, type=Inputs)
        for bone in inputs.msg.bones:
            if bone.name == "Hips":
                acc += float(bone.position[0])
                break
    elapsed = perf_counter() - t0
    return elapsed, acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark input_view vs msgspec Struct inputs")
    parser.add_argument("--iterations", type=int, default=300_000)
    args = parser.parse_args()

    iterations = max(1, int(args.iterations))
    input_view_elapsed, input_view_acc = _run_input_view(iterations)
    msgspec_elapsed, msgspec_acc = _run_msgspec(iterations)

    if abs(input_view_acc - msgspec_acc) > 1e-9:
        raise AssertionError("benchmark validation failed: accumulators differ")

    input_view_ops = iterations / input_view_elapsed if input_view_elapsed > 0 else 0.0
    msgspec_ops = iterations / msgspec_elapsed if msgspec_elapsed > 0 else 0.0
    speedup = (msgspec_ops / input_view_ops) if input_view_ops > 0 else 0.0

    print(f"input_view  : {input_view_ops:,.0f} ops/s ({input_view_elapsed:.4f}s)")
    print(f"msgspec     : {msgspec_ops:,.0f} ops/s ({msgspec_elapsed:.4f}s)")
    print(f"speedup     : {speedup:.2f}x")


if __name__ == "__main__":
    main()
