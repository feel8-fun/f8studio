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
- ring buffer: 1–2 seconds of 10ms chunks

### Layout (proposal)

- Mapping name: `shm.<serviceId>.audio` (convention)
- Writer appends fixed-size chunks into a ring. Multiple consumers keep their own `last_seq` (no shared read cursor).
- Optional Windows event: writer pulses named event `<shmName>_evt` on each new chunk (manual-reset; SetEvent+ResetEvent).

### Header (proposal)

- `magic` (u32) = `0xF8A11A02`
- `version` (u32) = `1`
- `sample_rate` (u32)
- `channels` (u16)
- `format` (u16) = `1` for F32LE, `2` for S16LE
- `frames_per_chunk` (u32) fixed block size (e.g. 10ms at 48k = 480)
- `chunk_count` (u32) total chunks in ring
- `bytes_per_frame` (u32) = `channels * bytes_per_sample`
- `payload_bytes_per_chunk` (u32) = `frames_per_chunk * bytes_per_frame`
- `write_seq` (u64) monotonic chunk counter
- `write_frame_index` (u64) monotonic frame counter (optional but useful for A/V sync)
- `ts_ms` (i64) wall-clock ms for the latest chunk (start timestamp)

### Chunk record

Each chunk is `ChunkHeader + PCM payload`:

- `seq` (u64) chunk sequence number
- `ts_ms` (i64) wall-clock ms for this chunk
- `frames` (u32) valid frames in this chunk (<= frames_per_chunk)
- padding (u32)
- payload: interleaved PCM, `frames * bytes_per_frame` bytes, remaining bytes in the chunk are undefined.

Consumers:
- Read `write_seq`, pick the newest `seq` and compute ring index `seq % chunk_count`.
- Validate `ChunkHeader.seq` matches expected `seq` before using payload; if mismatch, retry.
