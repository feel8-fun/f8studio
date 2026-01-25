# SHM Formats (Draft)

This repo uses Windows named shared memory (CreateFileMapping) for low-latency media exchange between services.

## Video SHM v1 (current)

Producer: `f8::implayer::VideoSharedMemorySink` (`packages/f8implayer/src/video_shared_memory_sink.cpp`)

- Mapping name: `shm.<serviceId>.video` (convention)
- Pixel format: BGRA32, interleaved (8-bit per channel), row-major
- Double-buffered (N slots), writer updates payload first, then updates header (frame_id last)
- Optional Windows event: writer pulses named event `<shmName>_evt` on each new frame (manual-reset; SetEvent+ResetEvent) so multiple consumers waiting are woken together.

### Header

Binary layout is little-endian, 8-byte aligned.

- `magic` (u32) = `0xF8A11A01`
- `version` (u32) = `1`
- `slot_count` (u32)
- `width` (u32)  (output width)
- `height` (u32) (output height)
- `pitch` (u32)  (bytes per row, equals `width * 4`)
- `format` (u32) = `1` for BGRA32
- padding (4 bytes)
- `frame_id` (u64) monotonic counter
- `ts_ms` (i64) wall-clock milliseconds since epoch
- `active_slot` (u32) which slot contains the newest frame
- `payload_capacity` (u32) capacity per slot in bytes

Total header size: 56 bytes.

### Payload

Payload begins immediately after header:

`payload_base = header_size + active_slot * payload_capacity`

Valid bytes for the frame are `pitch * height`.

## Audio SHM v1 (proposed)

Goal: Provide a stable, analysis-friendly PCM stream independent from playback quality.

Suggested defaults:
- sampleRate: 48000
- channels: 2
- format: float32 little-endian (interleaved)
- slot_count: 8 (more jitter tolerance than video)

### Header (proposal)

- `magic` (u32) = `0xF8A11A02`
- `version` (u32) = `1`
- `slot_count` (u32)
- `sample_rate` (u32)
- `channels` (u32)
- `format` (u32) = `1` for F32LE, `2` for S16LE
- `frames_per_slot` (u32) fixed block size per slot
- padding (0 or 4 bytes to align 8)
- `frame_id` (u64) monotonic block counter
- `ts_ms` (i64) wall-clock ms for the block start (or mid-point)
- `active_slot` (u32)
- `payload_capacity` (u32) bytes per slot

Payload: interleaved PCM, `frames_per_slot * channels * bytes_per_sample` bytes.
