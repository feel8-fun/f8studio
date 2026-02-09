# Lovense Mock Server (f8) Notes

This repo includes a lightweight Lovense Local API mock server:
`packages/f8pyengine/f8pyengine/operators/lovense_mock_server.py`.

It is used to **capture** `/command` traffic and to return **spec-shaped** JSON responses for common commands so that
clients can proceed while you log/analyze requests.

## JSON formatting compatibility (important)

Some Lovense clients/SDKs are strict about JSON formatting. The mock server intentionally uses **compact JSON**:
no extra spaces after separators (`,`, `:`).

That includes:
- The outer HTTP JSON response body
- The `data.toys` field (which is itself a JSON string)

## Quick curl examples

Assuming the server is listening on `127.0.0.1:30010`:

```bash
# GetToys
curl -sS -X POST http://127.0.0.1:30010/command \
  -H 'Content-Type: application/json' \
  -H 'X-platform: f8' \
  --data '{"command":"GetToys"}'

# GetToyName
curl -sS -X POST http://127.0.0.1:30010/command \
  -H 'Content-Type: application/json' \
  -H 'X-platform: f8' \
  --data '{"command":"GetToyName"}'

# Function
curl -sS -X POST http://127.0.0.1:30010/command \
  -H 'Content-Type: application/json' \
  -H 'X-platform: f8' \
  --data '{"command":"Function","action":"Vibrate:10","timeSec":5,"apiVer":1}'

# Position
curl -sS -X POST http://127.0.0.1:30010/command \
  -H 'Content-Type: application/json' \
  -H 'X-platform: f8' \
  --data '{"command":"Position","value":"38","toy":"ff922f7fd345","apiVer":1}'
```

## Responses (high level)

- `GetToys` returns `{code:200,type:"OK",data:{toys:"{...}",platform:"pc",appType:"remote"}}` (+ compatibility fields).
- Common control commands (`Function`, `Position`, `Pattern`, `PatternV2`, `Preset`) return `{code:200,type:"ok"}` when parameters look valid.
- Unknown commands return `{code:400,type:"error",...}`.
- If a `toy` id is provided and not recognized, returns `{code:401,type:"error",...}`.

## Event state shape

The node writes a compact, analysis-friendly object to the `event` state field.

Key fields:
- `command.name` (e.g. `Pattern`, `Function`) and `command.kind` (derived classification)
- `toys.scope` (`all` / `selected`) and resolved `toys.ids` / `toys.names`
- `params` contains common request parameters (e.g. `action`, `timeSec`, `rule`, `strength`, `value`, `name`)

For deeper debugging you can enable:
- `eventIncludePayload`: adds the parsed payload into the event
- `eventIncludeRequest`: adds selected headers + bodyText into the event
