# Unity

Use the pinned `extensions/f8unitymods` Git submodule as the authoritative Unity
toolchain. Initialize it with `git submodule update --init --recursive`; do not
fall back to a sibling checkout.

PyStudio wraps:

- `game_setup.run_detect()`
- `game_setup.run_diagnose()`
- `game_setup.run_install()`

Default behavior:

- Exporter option is `auto`; profile metadata resolves to `F8SkeletonStreamer` or `F8Live2DStreamer`.
- Default UDP skeleton stream port is `39540`.
- Mono/IL2CPP backend detection chooses the matching BepInEx stack.
- Optional utility flags are explicit booleans: RuntimeUnityEditor, CinematicUnityExplorer, ConfigurationManager, UniversalUnityDemosaics.
- Unknown Unity games need a preview and approval before a generated custom profile is installed.

Expected graph:

`UDP In(port=39540) -> Skeleton Decoder -> Viz 3D`

The production exporter emits the LMEX v2 binary extension with stable
`profileId:role:roleIndex` character identity. Stream verification succeeds
only after at least one complete binary frame has decoded; receiving arbitrary
UDP datagrams is not sufficient.

Only add TCode branches after stream verification or when the recipe already
contains a TCode graph. The guided OSR graph must keep `Serial Out.enabled=false`
until the user explicitly arms it and must gate serial execution through the
250 ms stream watchdog.
