from __future__ import annotations

# pyright: reportMissingImports=false

import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def main() -> None:
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    source, destination = (Path(value) for value in arguments)
    bpy.ops.import_scene.fbx(filepath=str(source))
    armatures = [item for item in bpy.context.scene.objects if item.type == "ARMATURE"]
    if len(armatures) != 1:
        raise ValueError(f"Expected one animated armature in {source}, found {len(armatures)}")
    armature = armatures[0]
    action = armature.animation_data.action if armature.animation_data is not None else None
    if action is None:
        raise ValueError(f"No armature animation found in {source}")

    bones = list(armature.pose.bones)
    names = [bone.name.split(":", 1)[-1] for bone in bones]
    indexes = {bone.name: index for index, bone in enumerate(bones)}
    parents = [indexes[bone.parent.name] if bone.parent is not None else -1 for bone in bones]
    frame_start = int(action.frame_range[0])
    frame_end = int(action.frame_range[1])
    frame_rate = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    to_y_up = Matrix.Rotation(-1.5707963267948966, 4, "X")
    frames: list[list[list[float]]] = []
    for frame_number in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame_number)
        frame: list[list[float]] = []
        for bone in bones:
            matrix = to_y_up @ armature.matrix_world @ bone.matrix
            position = matrix.translation
            rotation = matrix.to_quaternion()
            frame.append([position.x, position.y, position.z, rotation.w, rotation.x, rotation.y, rotation.z])
        frames.append(frame)

    destination.write_text(json.dumps({
        "frameRate": frame_rate,
        "boneNames": names,
        "parents": parents,
        "frames": frames,
    }, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
