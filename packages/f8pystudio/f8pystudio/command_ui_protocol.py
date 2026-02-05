from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from qtpy import QtWidgets


class CommandUiSource(str, Enum):
    NODEGRAPH = "nodegraph"
    PROPERTIES_BIN = "properties_bin"


class CommandUiHandler(ABC):
    """
    PyStudio-only protocol for nodes that want to override command invocation with custom UI.

    Implementations should return True when the command has been handled (ie. the
    default command invocation should NOT run).
    """

    @abstractmethod
    def handle_command_ui(
        self,
        cmd: Any,
        *,
        parent: QtWidgets.QWidget | None,
        source: CommandUiSource,
    ) -> bool:
        raise NotImplementedError

