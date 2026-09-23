# Studio Runtime Nodes

The `f8.pystudio` runtime is built into the Web Studio server. These operators provide authoring and presentation behavior without a desktop GUI process.

| Family | Operators and behavior |
| --- | --- |
| Presentation | Text, wave, video, audio spectrum, track overlay, 3D scene and TCode views |
| Graph control | Control panel, patch hub and value stepper |
| Expressions | Typed data and state expressions using an AST allowlist |
| Annotation | Backdrop and note nodes stored with the graph |

Video previews appear inside graph nodes and may also be inspected in the Presentation workspace. 3D and Monaco assets are part of the local production bundle; they do not load code from a CDN.

Studio runtime operators are explicitly registered under `packages/f8studio_server/f8studio_server/studio_runtime`. Presentation renderers are explicitly mapped under `packages/f8studio_web/src/presentation`. There is no runtime plugin discovery path.
