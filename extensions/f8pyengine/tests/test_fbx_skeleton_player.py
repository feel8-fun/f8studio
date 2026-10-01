from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from f8pysdk.specs import F8RuntimeNode
from f8pyengine.constants import SERVICE_CLASS
from f8pyengine.operators.fbx_skeleton_player import AnimationClip, FbxSkeletonPlayerRuntimeNode
from f8pyengine.pyengine_node_registry import create_pyengine_registry


def test_fbx_player_produces_animated_skeleton_frames() -> None:
    clip = AnimationClip(
        frame_rate=1.0,
        bone_names=["Hips", "Head"],
        parents=[-1, 0],
        frames=[
            [(0, 1, 0, 1, 0, 0, 0), (0, 2, 0, 1, 0, 0, 0)],
            [(1, 1, 0, 1, 0, 0, 0), (1, 2, 0, 1, 0, 0, 0)],
        ],
    )

    async def scenario() -> None:
        node = F8RuntimeNode(
            nodeId="fbx", serviceId="engine", serviceClass=SERVICE_CLASS,
            operatorClass=FbxSkeletonPlayerRuntimeNode.SPEC.operatorClass,
            dataInPorts=list(FbxSkeletonPlayerRuntimeNode.SPEC.dataInPorts),
            dataOutPorts=list(FbxSkeletonPlayerRuntimeNode.SPEC.dataOutPorts),
        )
        player = FbxSkeletonPlayerRuntimeNode(node_id="fbx", node=node, initial_state={"path": "tests/Excited.fbx"})
        assert player._loop is True
        with patch("f8pyengine.operators.fbx_skeleton_player.load_fbx_clip", return_value=clip) as loader:
            input_time = AsyncMock(return_value=None)
            with patch.object(player, "pull", new=input_time):
                assert await player.compute_output("skeletons") is None
                loader.assert_not_called()
                input_time.return_value = 0.0
                first = await player.compute_output("skeletons")
                input_time.return_value = 1.1
                second = await player.compute_output("skeletons")
                await player.on_state("loop", False)
                input_time.return_value = 2.0
                ended = await player.compute_output("skeletons")
            loader.assert_called_once_with(Path("tests/Excited.fbx"), "")
            assert first["bones"][1]["parent"] == "Hips"
            assert first["bones"][0]["pos"] == [0, 1, 0]
            assert second["bones"][0]["pos"] == [1, 1, 0]
            assert ended is None
            await player.close()

    asyncio.run(scenario())


def test_fbx_player_uses_external_tick_time_for_looping_and_completion() -> None:
    clip = AnimationClip(
        frame_rate=1.0, bone_names=["Hips"], parents=[-1],
        frames=[[(0, 0, 0, 1, 0, 0, 0)], [(1, 0, 0, 1, 0, 0, 0)]],
    )

    async def scenario() -> None:
        node = F8RuntimeNode(
            nodeId="fbx", serviceId="engine", serviceClass=SERVICE_CLASS,
            operatorClass=FbxSkeletonPlayerRuntimeNode.SPEC.operatorClass,
            dataInPorts=list(FbxSkeletonPlayerRuntimeNode.SPEC.dataInPorts),
            dataOutPorts=list(FbxSkeletonPlayerRuntimeNode.SPEC.dataOutPorts),
        )
        player = FbxSkeletonPlayerRuntimeNode(node_id="fbx", node=node, initial_state={"path": "tests/Excited.fbx"})
        assert player._loop is True
        with patch("f8pyengine.operators.fbx_skeleton_player.load_fbx_clip", return_value=clip):
            with patch.object(player, "pull", new=AsyncMock(return_value=3.25)):
                looped = await player.compute_output("skeletons")
            assert looped["bones"][0]["pos"] == [1, 0, 0]
            with patch.object(player, "pull", new=AsyncMock(return_value=0.25)):
                rewound = await player.compute_output("skeletons")
            assert rewound["bones"][0]["pos"] == [0, 0, 0]
            await player.on_state("loop", False)
            with patch.object(player, "pull", new=AsyncMock(return_value=2.0)):
                assert await player.compute_output("skeletons") is None
        await player.close()

    asyncio.run(scenario())


def test_fbx_player_is_registered_in_pyengine() -> None:
    registry = create_pyengine_registry()
    assert any(
        spec.operatorClass == FbxSkeletonPlayerRuntimeNode.SPEC.operatorClass
        for spec in registry.operator_specs(SERVICE_CLASS)
    )


def test_replacing_path_during_load_does_not_cancel_output_sampling() -> None:
    async def scenario() -> None:
        node = F8RuntimeNode(
            nodeId="fbx", serviceId="engine", serviceClass=SERVICE_CLASS,
            operatorClass=FbxSkeletonPlayerRuntimeNode.SPEC.operatorClass,
            dataInPorts=list(FbxSkeletonPlayerRuntimeNode.SPEC.dataInPorts),
            dataOutPorts=list(FbxSkeletonPlayerRuntimeNode.SPEC.dataOutPorts),
        )
        player = FbxSkeletonPlayerRuntimeNode(node_id="fbx", node=node, initial_state={"path": "old.fbx"})
        loading = asyncio.Event()
        release = asyncio.Event()

        async def load(_path: str, _blender_path: str) -> None:
            loading.set()
            await release.wait()

        with patch.object(player, "_load_clip", new=load):
            with patch.object(player, "pull", new=AsyncMock(return_value=0.0)):
                output = asyncio.create_task(player.compute_output("skeletons"))
                await loading.wait()
                await player.on_state("path", "new.fbx")
                assert await output is None
        release.set()
        await player.close()

    asyncio.run(scenario())
