from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from f8pyfallendoll.stream import (
    parse_latest_frame,
    read_appended,
    resolve_runtime_dir,
    select_frame,
)


def _bone(name: str, x: float = 0.0) -> dict[str, Any]:
    return {"name": name, "pos": [x, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]}


def _payload(
    *,
    timestamp_ms: int,
    role: str,
    role_index: int,
    priority: int,
    preferred_bones: list[str],
    bones: list[dict[str, Any]],
) -> dict[str, Any]:
    stable_key = f"fallen-doll:{role}:{role_index}"
    return {
        "type": "skeleton_binary",
        "schema": "fallen-doll-ue-world-v1",
        "modelName": stable_key,
        "stableKey": stable_key,
        "timestampMs": timestamp_ms,
        "bones": bones,
        "trailer": {
            "profileId": "fallen-doll",
            "hanimeActive": True,
            "hanimeId": "Hand02",
            "hanimeAsset": "/Game/HAnime/Hand02",
            "hanimeCategory": "hand",
            "role": role,
            "roleIndex": role_index,
            "participantPriority": priority,
            "preferredBones": preferred_bones,
        },
    }


def test_resolve_runtime_dir_uses_specific_override_first(tmp_path: Path) -> None:
    exact = tmp_path / "exact"
    games = tmp_path / "games"
    resolved = resolve_runtime_dir(
        {"FD_TCODE_RUNTIME_DIR": str(exact), "F8STUDIO_GAMES_DIR": str(games)},
        home=tmp_path / "home",
    )
    assert resolved == exact.resolve()


def test_parse_latest_frame_filters_profile_gate_and_old_frames() -> None:
    old = _payload(
        timestamp_ms=100,
        role="male",
        role_index=0,
        priority=0,
        preferred_bones=["Penis02"],
        bones=[_bone("Penis02")],
    )
    male = {**old, "timestampMs": 120}
    female = _payload(
        timestamp_ms=120,
        role="female",
        role_index=0,
        priority=0,
        preferred_bones=["R_Hand"],
        bones=[_bone("R_Hand")],
    )
    inactive = _payload(
        timestamp_ms=120,
        role="female",
        role_index=1,
        priority=1,
        preferred_bones=["L_Hand"],
        bones=[_bone("L_Hand")],
    )
    inactive["trailer"]["hanimeActive"] = False

    parsed = parse_latest_frame(
        [json.dumps(old), json.dumps(male), json.dumps(female), json.dumps(inactive), "not-json"],
        arrival_timestamp_ms=999,
    )

    assert [item["stableKey"] for item in parsed.skeletons] == ["fallen-doll:female:0", "fallen-doll:male:0"]
    assert all(item["timestampMs"] == 999 for item in parsed.skeletons)
    assert all(item["sourceTimestampMs"] == 120 for item in parsed.skeletons)
    assert parsed.dropped_payloads == 1
    assert parsed.rejected_lines == 2


def test_select_frame_uses_participant_priority_and_preferred_bone_order() -> None:
    male_low_priority = _payload(
        timestamp_ms=120,
        role="male",
        role_index=0,
        priority=5,
        preferred_bones=["Penis01"],
        bones=[_bone("Penis01")],
    )
    male_primary = _payload(
        timestamp_ms=120,
        role="male",
        role_index=1,
        priority=0,
        preferred_bones=["Penis02", "Penis01"],
        bones=[_bone("Penis01"), _bone("Penis02")],
    )
    female = _payload(
        timestamp_ms=120,
        role="female",
        role_index=0,
        priority=0,
        preferred_bones=["R_Hand", "L_Hand"],
        bones=[_bone("L_Hand"), _bone("R_Hand")],
    )

    selected = select_frame(
        [male_low_priority, male_primary, female],
        reference_role="male",
        target_role="female",
        enabled_reference_participants=["fallen-doll:male:0", "fallen-doll:male:1"],
        enabled_target_participants=["fallen-doll:female:0"],
        enabled_reference_bones=["Penis01", "Penis02"],
        enabled_target_bones=["L_Hand", "R_Hand"],
    )

    assert selected.status["valid"] is True
    assert selected.status["referenceKey"] == "fallen-doll:male:1"
    assert selected.reference_bone is not None and selected.reference_bone["name"] == "Penis02"
    assert selected.target_bone is not None and selected.target_bone["name"] == "R_Hand"


def test_select_frame_adds_mouth_specific_target_basis() -> None:
    male = _payload(
        timestamp_ms=120,
        role="male",
        role_index=0,
        priority=0,
        preferred_bones=["Penis01"],
        bones=[_bone("Penis01")],
    )
    female = _payload(
        timestamp_ms=120,
        role="female",
        role_index=0,
        priority=0,
        preferred_bones=["M_Jaw"],
        bones=[_bone("M_Jaw")],
    )

    selected = select_frame(
        [male, female],
        reference_role="male",
        target_role="female",
        enabled_reference_participants=["fallen-doll:male:0"],
        enabled_target_participants=["fallen-doll:female:0"],
        enabled_reference_bones=["Penis01"],
        enabled_target_bones=["M_Jaw"],
    )

    assert selected.target_bone is not None
    assert selected.target_bone["basis"] == {"up": "+local_y", "right": "-local_z"}


def test_select_frame_builds_bilateral_target_when_pose_has_no_primary_foot() -> None:
    male = _payload(
        timestamp_ms=120,
        role="male",
        role_index=0,
        priority=0,
        preferred_bones=["Penis01"],
        bones=[_bone("Penis01")],
    )
    female = _payload(
        timestamp_ms=120,
        role="female",
        role_index=0,
        priority=0,
        preferred_bones=[],
        bones=[_bone("R_Foot"), _bone("L_Foot")],
    )
    female["trailer"]["hanimeCategory"] = "foot"

    selected = select_frame(
        [male, female],
        reference_role="male",
        target_role="female",
        enabled_reference_participants=["fallen-doll:male:0"],
        enabled_target_participants=["fallen-doll:female:0"],
        enabled_reference_bones=["Penis01"],
        enabled_target_bones=["R_Foot", "L_Foot"],
    )

    assert selected.status["valid"] is True
    assert selected.target_bone is not None
    assert selected.target_bone["name"] == "R_Foot+L_Foot"
    assert selected.target_bone["targetMode"] == "bilateral_reference_axis"
    assert selected.target_bone["pairPositions"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_select_frame_uses_nearest_single_foot_when_pair_is_not_near_reference() -> None:
    male = _payload(
        timestamp_ms=120,
        role="male",
        role_index=0,
        priority=0,
        preferred_bones=["Penis01"],
        bones=[
            {"name": "Penis01", "pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]},
            {"name": "Penis09", "pos": [0.0, 0.0, 1.0], "rot": [1.0, 0.0, 0.0, 0.0]},
        ],
    )
    female = _payload(
        timestamp_ms=120,
        role="female",
        role_index=0,
        priority=0,
        preferred_bones=[],
        bones=[
            {"name": "R_Foot", "pos": [2.0, 0.0, 0.5], "rot": [1.0, 0.0, 0.0, 0.0]},
            {"name": "L_Foot", "pos": [0.1, 0.0, 0.5], "rot": [1.0, 0.0, 0.0, 0.0]},
        ],
    )
    female["trailer"]["hanimeCategory"] = "foot"

    selected = select_frame(
        [male, female],
        reference_role="male",
        target_role="female",
        enabled_reference_participants=["fallen-doll:male:0"],
        enabled_target_participants=["fallen-doll:female:0"],
        enabled_reference_bones=["Penis01"],
        enabled_target_bones=["R_Foot", "L_Foot"],
    )

    assert selected.target_bone is not None
    assert selected.target_bone["name"] == "L_Foot"
    assert "targetMode" not in selected.target_bone
    assert selected.target_bone["basis"] == {"up": "+local_z", "right": "-local_y"}


def test_select_frame_keeps_bilateral_feet_when_pair_midpoint_tracks_reference() -> None:
    male = _payload(
        timestamp_ms=120,
        role="male",
        role_index=0,
        priority=0,
        preferred_bones=["Penis01"],
        bones=[
            {"name": "Penis01", "pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]},
            {"name": "Penis09", "pos": [0.0, 0.0, 1.0], "rot": [1.0, 0.0, 0.0, 0.0]},
        ],
    )
    female = _payload(
        timestamp_ms=120,
        role="female",
        role_index=0,
        priority=0,
        preferred_bones=[],
        bones=[
            {"name": "R_Foot", "pos": [0.1, 0.0, 0.5], "rot": [1.0, 0.0, 0.0, 0.0]},
            {"name": "L_Foot", "pos": [-0.1, 0.0, 0.5], "rot": [1.0, 0.0, 0.0, 0.0]},
        ],
    )
    female["trailer"]["hanimeCategory"] = "foot"

    selected = select_frame(
        [male, female],
        reference_role="male",
        target_role="female",
        enabled_reference_participants=["fallen-doll:male:0"],
        enabled_target_participants=["fallen-doll:female:0"],
        enabled_reference_bones=["Penis01"],
        enabled_target_bones=["R_Foot", "L_Foot"],
    )

    assert selected.target_bone is not None
    assert selected.target_bone["name"] == "R_Foot+L_Foot"
    assert selected.target_bone["targetMode"] == "bilateral_reference_axis"


def test_select_frame_builds_bilateral_target_for_breast_contact() -> None:
    male = _payload(
        timestamp_ms=120,
        role="male",
        role_index=0,
        priority=0,
        preferred_bones=["Penis01"],
        bones=[_bone("Penis01")],
    )
    female = _payload(
        timestamp_ms=120,
        role="female",
        role_index=0,
        priority=0,
        preferred_bones=[],
        bones=[_bone("R_Breast_Nipple"), _bone("L_Breast_Nipple")],
    )
    female["trailer"]["hanimeCategory"] = "breast"

    selected = select_frame(
        [male, female],
        reference_role="male",
        target_role="female",
        enabled_reference_participants=["fallen-doll:male:0"],
        enabled_target_participants=["fallen-doll:female:0"],
        enabled_reference_bones=["Penis01"],
        enabled_target_bones=["R_Breast_Nipple", "L_Breast_Nipple"],
    )

    assert selected.status["valid"] is True
    assert selected.target_bone is not None
    assert selected.target_bone["name"] == "R_Breast_Nipple+L_Breast_Nipple"
    assert selected.target_bone["targetMode"] == "bilateral_reference_axis"


def test_select_frame_uses_enabled_fallback_when_preferred_side_is_disabled() -> None:
    male = _payload(
        timestamp_ms=120,
        role="male",
        role_index=0,
        priority=0,
        preferred_bones=["Penis01"],
        bones=[_bone("Penis01")],
    )
    female = _payload(
        timestamp_ms=120,
        role="female",
        role_index=0,
        priority=0,
        preferred_bones=["R_Hand"],
        bones=[_bone("R_Hand"), _bone("L_Hand")],
    )

    selected = select_frame(
        [male, female],
        reference_role="male",
        target_role="female",
        enabled_reference_participants=["fallen-doll:male:0"],
        enabled_target_participants=["fallen-doll:female:0"],
        enabled_reference_bones=["Penis01"],
        enabled_target_bones=["L_Hand"],
    )

    assert selected.status["valid"] is True
    assert selected.target_bone is not None
    assert selected.target_bone["name"] == "L_Hand"


def test_read_appended_detects_truncation(tmp_path: Path) -> None:
    spool = tmp_path / "fd-skeleton.ndjson"
    spool.write_bytes(b"first\nsecond\n")
    initial = read_appended(spool, 0)
    assert initial.text == "first\nsecond\n"
    assert initial.truncated is False

    spool.write_bytes(b"new\n")
    truncated = read_appended(spool, initial.offset)
    assert truncated.text == "new\n"
    assert truncated.truncated is True
