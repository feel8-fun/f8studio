# Runner (Headless)

`f8pysdk.headless_runner` executes a saved Studio session JSON without launching GUI.

## Launch

```bash
python -m f8pysdk.headless_runner --session path/to/session.json
```

## Command Arguments

- `--session` (required): path to Studio session JSON.
- `--nats-url`: NATS endpoint (default `nats://127.0.0.1:4222`).
- `--discovery-root`: additional service discovery root (repeatable).
- `--builtin-injector`: injector callable (`module:function`, repeatable).
- `--discovery-live`: disable static describe fast path.
- `--no-auto-start`: do not auto-start service processes.
- `--no-bootstrap`: disable local NATS bootstrap.

## Example

```bash
python -m f8pysdk.headless_runner \
  --session sessions/demo.json \
  --nats-url nats://127.0.0.1:4222 \
  --discovery-root services
```

## Studio vs Runner

- **Studio**: graph editing, interactive operation.
- **Runner**: deterministic execution of saved graph layouts for automation pipelines.

## Failure Boundaries

Headless runner exits with non-zero code on validation/runtime failures and logs traceback context for diagnosis.
