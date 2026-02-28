# Scene 04: Functional TCode Generation

## Script

[Download JSON](scripts/scene-04-functional_tcode.json)

## Goal

Build multi-channel control signals from functional operators (`Phase`, `Tempest`, `Cosine`, `Expr`) and output as `TCode`.

## Main Pipeline

`Phase -> Tempest -> Expr`

`Phase 1 -> Cosine -> Expr`

`Expr/Cosine -> TCode -> Serial Out`

## Steps

1. Import the script and start `PyEngine`.
2. Confirm timing link `Tick.tickMs -> TCode.intervalMs` (both `100` by default).
3. Verify signal paths: `Phase -> Tempest -> Expr.x` and `Phase 1 -> Cosine -> Expr.y`.
4. Review `Expr.code` (current: `x*(y*0.5+0.5)`) to confirm mixing behavior.
5. Ensure output mapping is correct: `Expr.out -> TCode.L0/R0`, `Cosine.value -> TCode.L1`.
6. Configure and enable `Serial Out`; inspect behavior with `TCodeViz`, `TextViz`, and `WaveViz`.

## Key Parameters

- `Phase.hz`: `0.69` (main rhythm)
- `Phase 1.hz`: `0.1` (slow modulation)
- `Tempest.eccentric`: `0.77` (waveform shape)
- `TCode.intervalMs`: `100`

## Validation

- `WaveViz` shows clear phase/amplitude variation for `Tempest`, `Cosine`, and `Expr`.
- `TextViz` continuously refreshes `TCode` strings.
- `TCodeViz` and serial output stay synchronized without abrupt jumps.

## Image Placeholders

- `docs/media/images/scenarios/scene-04-overview.png`
- `docs/media/images/scenarios/scene-04-waveforms.png`
