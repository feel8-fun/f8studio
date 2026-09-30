"""
Explicit internal-only service bus boundary.

Import concrete owner modules directly:
- `f8pysdk.service_bus.internal.control_endpoints`
- `f8pysdk.service_bus.internal.command`
- `f8pysdk.service_bus.internal.cache`
- `f8pysdk.service_bus.internal.logging`

Public state protocol types live under:
- `f8pysdk.state`

State runtime owners live under:
- `f8pysdk.service_bus.state.pipeline`
- `f8pysdk.service_bus.state.store`
- `f8pysdk.service_bus.state.router`
- `f8pysdk.service_bus.state.helpers`
- `f8pysdk.service_bus.state.options`

Data runtime owners live under:
- `f8pysdk.service_bus.data.emit`
- `f8pysdk.service_bus.data.flow`
- `f8pysdk.service_bus.data.router`
"""
from __future__ import annotations
