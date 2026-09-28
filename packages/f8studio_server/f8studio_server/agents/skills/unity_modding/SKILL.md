# Unity Game Modding

Start with `modding_detect_target`. For a supported Unity target, use `modding_preview_unity_install` and review its blocking errors and exact files to write. The user must approve `modding_apply_unity_install`; do not guess an installer or write directly to the game directory.

After installation, ask the user to start the game and enter a scene with characters. Use `modding_verify_udp` on the previewed port (normally 39540). A received packet is not sufficient: require a complete decoded skeleton frame. Inspect logs and decoder errors when verification fails.

Use the returned graph build plan and current catalog to prepare a graph patch. Preview the patch, obtain approval, and verify `UDP In -> Skeleton Decoder -> Viz 3D` before adding downstream outputs. Keep physical serial output disabled until the user explicitly arms it.

Unreal installation is not available in Studio yet. Report the detected engine and missing installer rather than writing into a UE4SS directory.
