from __future__ import annotations


COCO17_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

MEDIAPIPE_POSE_33_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]

HUMAN36M_17_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
]

_SKELETON_EDGES_BY_PROTOCOL: dict[str, list[tuple[int, int]]] = {
    "coco17": COCO17_EDGES,
    "mediapipe_pose_33": MEDIAPIPE_POSE_33_EDGES,
    "human36m_17": HUMAN36M_17_EDGES,
}


def skeleton_edges_for_protocol(skeleton_protocol: str) -> list[tuple[int, int]] | None:
    protocol = str(skeleton_protocol or "").strip().lower()
    if not protocol:
        return None
    edges = _SKELETON_EDGES_BY_PROTOCOL.get(protocol)
    if edges is None:
        return None
    return list(edges)
