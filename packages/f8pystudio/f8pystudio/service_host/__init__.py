from .constants import SERVICE_CLASS, STUDIO_SERVICE_ID

__all__ = [
    "SERVICE_CLASS",
    "STUDIO_SERVICE_ID",
    "register_pystudio_specs",
]


def __getattr__(name: str):  # pragma: no cover
    # Avoid importing `service_host_registry` eagerly to prevent circular imports
    # with `f8pystudio.runtime_nodes.*`.
    if name == "register_pystudio_specs":
        from .service_host_registry import register_pystudio_specs

        return register_pystudio_specs
    raise AttributeError(name)
