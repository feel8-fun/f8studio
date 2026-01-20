from .constants import SERVICE_CLASS, STUDIO_SERVICE_ID

__all__ = [
    "SERVICE_CLASS",
    "STUDIO_SERVICE_ID",
    "ServiceHostRegistry",
    "register_pystudio_specs",
]


def __getattr__(name: str):  # pragma: no cover
    # Keep backward-compatible imports like:
    # - `from f8pystudio.service_host import ServiceHostRegistry`
    #
    # Avoid importing `service_host_registry` eagerly to prevent circular imports
    # with `f8pystudio.runtime_nodes.*`.
    if name in ("ServiceHostRegistry", "register_pystudio_specs"):
        from .service_host_registry import ServiceHostRegistry, register_pystudio_specs

        return {"ServiceHostRegistry": ServiceHostRegistry, "register_pystudio_specs": register_pystudio_specs}[name]
    raise AttributeError(name)
