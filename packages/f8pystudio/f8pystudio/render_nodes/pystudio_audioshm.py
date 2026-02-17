from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pyqtgraph as pg  # type: ignore[import-not-found]
from qtpy import QtCore, QtWidgets
from NodeGraphQt.nodes.base_node import NodeBaseWidget

from f8pysdk.shm import AudioShmReader, read_audio_header, SAMPLE_FORMAT_F32LE

from ..nodegraph.operator_basenode import F8StudioOperatorBaseNode
from ..nodegraph.viz_operator_nodeitem import F8StudioVizOperatorNodeItem
from ..ui_bus import UiCommand


class _AudioShmPane(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._title = QtWidgets.QLabel("AudioSHM")
        self._title.setStyleSheet("color: rgb(225, 225, 225);")

        self._plot = pg.PlotWidget()
        self._plot.setBackground((20, 20, 20))
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        # Compact viz-style: no axis captions or tick labels.
        try:
            axb = self._plot.getAxis("bottom")
            axl = self._plot.getAxis("left")
            axb.setLabel("")
            axl.setLabel("")
            axb.setStyle(showValues=False)
            axl.setStyle(showValues=False)
        except (AttributeError, RuntimeError, TypeError):
            pass
        try:
            pi = self._plot.getPlotItem()
            if pi is not None:
                try:
                    pi.layout.setContentsMargins(2, 2, 2, 2)
                except (AttributeError, RuntimeError, TypeError):
                    pass
                try:
                    pi.setDefaultPadding(0.02)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
        except (AttributeError, RuntimeError, TypeError):
            pass
        self._curve = self._plot.plot([], [], pen=pg.mkPen((120, 200, 255), width=1))

        self._status = QtWidgets.QLabel("")
        self._status.setStyleSheet("color: rgb(160, 160, 160);")

        layout.addWidget(self._title)
        layout.addWidget(self._plot, 1)
        layout.addWidget(self._status)

        # Node label + state fields already provide context; keep pane compact by default.
        self._title.setVisible(False)
        self._status.setVisible(False)
        self.setMinimumWidth(200)
        self.setMinimumHeight(120)
        self.setMaximumWidth(200)
        self.setMaximumHeight(150)


        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)  # type: ignore[attr-defined]
        self._timer.setInterval(20)

        self._reader: AudioShmReader | None = None
        self._shm_name = ""
        self._last_seq = 0

        self._history_ms = 250
        self._channel = 0
        self._sample_rate = 48000
        self._window_frames = 0
        self._x = np.zeros((0,), dtype=np.float32)
        self._y = np.zeros((0,), dtype=np.float32)

    def set_config(self, *, shm_name: str, throttle_ms: int, history_ms: int, channel: int) -> None:
        shm_name = str(shm_name or "").strip()
        self._history_ms = max(20, int(history_ms))
        self._channel = max(0, int(channel))
        throttle_ms = max(0, int(throttle_ms))
        self._timer.setInterval(max(1, throttle_ms) if throttle_ms > 0 else 1)
        if shm_name != self._shm_name:
            self._shm_name = shm_name
            self._reset_reader()
        if not self._timer.isActive():
            self._timer.start()

    def detach(self) -> None:
        try:
            self._timer.stop()
        except (AttributeError, RuntimeError, TypeError):
            pass
        self._reset_reader()

    def _reset_reader(self) -> None:
        try:
            if self._reader is not None:
                self._reader.close()
        except (AttributeError, RuntimeError, TypeError):
            pass
        self._reader = None
        self._last_seq = 0
        self._status.setText("")
        self._title.setText("AudioSHM")

    def _ensure_reader(self) -> bool:
        if self._reader is not None:
            return True
        if not self._shm_name:
            self._status.setText("missing shmName")
            return False
        try:
            r = AudioShmReader(self._shm_name)
            r.open(use_event=False)
            self._reader = r
            self._status.setText(f"{self._shm_name}")
            return True
        except Exception as exc:
            self._status.setText(f"open failed: {exc}")
            self._reader = None
            return False

    def _rebuild_window(self, sample_rate: int) -> None:
        self._sample_rate = int(sample_rate)
        self._window_frames = max(1, int(math.ceil(float(self._sample_rate) * float(self._history_ms) / 1000.0)))
        self._x = np.linspace(-float(self._history_ms) / 1000.0, 0.0, self._window_frames, dtype=np.float32)
        self._y = np.zeros((self._window_frames,), dtype=np.float32)
        self._curve.setData(self._x, self._y)

    def _tick(self) -> None:
        if not self._ensure_reader():
            return
        assert self._reader is not None
        buf = self._reader.buf
        hdr = read_audio_header(buf)
        if hdr is None or hdr.magic != 0xF8A11A02 or hdr.version != 1:
            return
        if int(hdr.fmt) != int(SAMPLE_FORMAT_F32LE):
            self._status.setText(f"unsupported fmt={hdr.fmt}")
            return
        if hdr.sample_rate <= 0 or hdr.channels <= 0:
            return
        if int(hdr.sample_rate) != int(self._sample_rate) or self._window_frames == 0:
            self._rebuild_window(int(hdr.sample_rate))

        seq = int(hdr.write_seq)
        if seq <= 0 or seq == int(self._last_seq):
            return

        h2, ch, payload = self._reader.read_chunk_f32(seq)
        if h2 is None or ch is None or payload is None:
            return
        frames = int(ch.frames)
        if frames <= 0:
            self._last_seq = seq
            return

        channels = int(h2.channels)
        samples = np.frombuffer(payload, dtype=np.float32).copy()
        try:
            samples = samples.reshape((frames, channels))
        except (TypeError, ValueError):
            self._last_seq = seq
            return
        idx = min(max(0, int(self._channel)), max(0, channels - 1))
        y = samples[:, idx]

        if frames >= self._window_frames:
            self._y[:] = y[-self._window_frames :]
        else:
            self._y[:-frames] = self._y[frames:]
            self._y[-frames:] = y[:frames]

        peak = float(np.max(np.abs(y))) if y.size else 0.0
        self._curve.setData(self._x, self._y)
        self._title.setText(f"AudioSHM  {int(h2.sample_rate)}Hz  ch={channels}  seq={seq}  peak={peak:.3f}")
        self._status.setText(f"{self._shm_name}  ts={int(ch.ts_ms)}")
        self._last_seq = seq


class _AudioShmWidget(NodeBaseWidget):
    def __init__(self, parent=None, name: str = "__audioshm", label: str = "") -> None:
        super().__init__(parent=parent, name=name, label=label)
        self._pane = _AudioShmPane()
        self.set_custom_widget(self._pane)

    def get_value(self) -> object:
        return {}

    def set_value(self, value: object) -> None:
        return

    def set_config(self, *, shm_name: str, throttle_ms: int, history_ms: int, channel: int) -> None:
        self._pane.set_config(shm_name=shm_name, throttle_ms=throttle_ms, history_ms=history_ms, channel=channel)

    def detach(self) -> None:
        self._pane.detach()


class PyStudioAudioShmNode(F8StudioOperatorBaseNode):
    """
    Render node for `f8.audio_shm_view`.
    """

    def __init__(self):
        super().__init__(qgraphics_item=F8StudioVizOperatorNodeItem)
        try:
            self.add_custom_widget(_AudioShmWidget(self.view, name="__audioshm", label=""))
        except (AttributeError, RuntimeError, TypeError):
            pass

    def apply_ui_command(self, cmd: UiCommand) -> None:
        c = str(cmd.command or "")
        if c == "audioshm.detach":
            try:
                w = self.get_widget("__audioshm")
                if not w:
                    return
                w.detach()
            except (AttributeError, RuntimeError, TypeError):
                pass
            return
        if c != "audioshm.set":
            return
        try:
            payload = dict(cmd.payload or {})
            shm_name = str(payload.get("shmName") or "").strip()
            throttle_ms = int(payload.get("throttleMs") or 20)
            history_ms = int(payload.get("historyMs") or 250)
            channel = int(payload.get("channel") or 0)
        except (AttributeError, TypeError, ValueError):
            return
        try:
            w = self.get_widget("__audioshm")
            if not w:
                return
            w.set_config(shm_name=shm_name, throttle_ms=throttle_ms, history_ms=history_ms, channel=channel)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return

