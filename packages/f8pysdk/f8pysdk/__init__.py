"""
Package init.

Keep this lightweight so utility submodules (e.g. `f8pysdk.shm`) can be imported
without pulling in optional/extra dependencies used by generated schemas.
"""

try:
    from .generated import *  # type: ignore
    from .schema_helpers import *  # type: ignore

    def _proxy_getattr(self, name):
        return self.root.__getattribute__(name)

    F8DataTypeSchema.__getattr__ = _proxy_getattr  # type: ignore[name-defined]
    F8JsonValue.__getattr__ = _proxy_getattr  # type: ignore[name-defined]
except Exception:
    # Optional deps for generated schemas (e.g. pydantic) may be missing in some
    # runtime environments; allow importing lightweight helpers regardless.
    pass
