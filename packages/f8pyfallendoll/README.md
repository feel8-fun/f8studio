# Fallen Doll Source

`f8pyfallendoll` is a game-specific source service for Feel8 Studio. It tails
the latest skeleton frames exported by the Fallen Doll UE4SS Lua mod, discards
backlog, selects the configured interaction participants and functional bones,
and exposes standard skeleton/bone payloads to the rest of the graph.

The default spool location is:

```text
~/.f8/studio/games/fallen-doll/runtime/fd-skeleton.ndjson
```

`F8STUDIO_GAMES_DIR` overrides the shared games directory. The more specific
`FD_TCODE_RUNTIME_DIR` overrides only Fallen Doll's runtime directory.

This service does not contain device output and never arms a physical device.
