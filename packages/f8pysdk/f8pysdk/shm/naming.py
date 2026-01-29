from __future__ import annotations


DEFAULT_VIDEO_SHM_BYTES = 256 * 1024 * 1024
DEFAULT_VIDEO_SHM_SLOTS = 2

DEFAULT_AUDIO_SHM_BYTES = 8 * 1024 * 1024


def video_shm_name(service_id: str) -> str:
    return f"shm.{service_id}.video"


def audio_shm_name(service_id: str) -> str:
    return f"shm.{service_id}.audio"


def frame_event_name(shm_name: str) -> str:
    return shm_name + "_evt"

