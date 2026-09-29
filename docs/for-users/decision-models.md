# Decision models in graphs

**Decision** is a PyEngine operator (`f8.decision`) for structured probabilistic decisions. Add a named **System-One decisions** connection in Studio Settings, then add the node from the operator catalog. Saved legacy TypeSafe Jev and local System-One settings continue to work. Decision providers are separate from conversational Agent providers.

Connect a text or JSON data source to `state`, and a Tick or another execution source to `exec`. Several questions can share one request; edit the `questions` JSON in the inspector. `routeQuestion` selects which answer drives the convenient value outputs and execution branch.

For a local host, set the model ID and base URL such as `http://127.0.0.1:8001/v1`, then put the connection ID from Studio Settings in the node's `providerId` field. The host must accept `POST /v1/systemone` with `model`, `state`, and `questions`, and return typed `answers`, `model`, and `usage` in the System-One response shape. The API key is optional for a local host. Model names such as Laya, Kev, and vjev-vision are host-provided IDs; Studio does not launch or download those models.

For a vision-capable host, its `/models` response must explicitly report image input for the selected model, for example `{"id":"vjev-vision","input_modalities":["text","image"]}`. Refresh models, save the connection, and connect a video-frame stream to `video`. The node subscribes to the latest stream frame and samples, resizes, and JPEG-encodes one frame only when `exec` fires; it does not infer on every video frame. An unavailable frame follows the `error` branch. The image is included as a JPEG data URL in `state.image`, with a 2 MB request limit. The external host must explicitly support this image convention; a text-only `/v1/systemone` implementation is insufficient. Existing legacy local settings retain their saved image flag.

```json
{
  "decision": {
    "type": "choice",
    "instructions": "Which action best matches the observation?",
    "criteria": {
      "moving": "The observation describes meaningful motion.",
      "idle": "The observation describes an idle scene.",
      "unknown": "There is insufficient evidence."
    }
  },
  "motion": {
    "type": "noul",
    "instructions": "Does the observation describe motion?"
  },
  "intensity": {
    "type": "score",
    "instructions": "Rate the motion intensity.",
    "criteria": ["None", "Low", "High"]
  }
}
```

| Output | Meaning |
| --- | --- |
| `answers` | Complete response, including every typed answer, actual model version, and token usage |
| `value` | Selected Choice label, weighted Score, or Noul boolean (yes probability ≥ 0.5) |
| `probabilities` | Full Choice/Score probability distribution |
| `probability` | Probability of the selected Choice option, or Noul's yes probability; null for Score |
| `confidence` | Provider's Choice/Score confidence; null for Noul |
| `accepted` | Whether the selected answer passes the configured thresholds |
| `metrics` | Processed, dropped, and failed counts; sample age and originating request ID |
| `error` | Most recent request error, cleared on the next successful result |

The `decided` execution output fires when the answer passes its thresholds; `uncertain` fires otherwise. `error` fires on request failure. `decided` means the answer is sufficiently certain under your configured policy, **not** that it is positive: a Noul probability of 0.05 is a confident “no”, and produces `value=false` through `decided` with the default thresholds. Downstream code should inspect `value` to choose the actual action. Choice labels such as `unknown` retain the meaning you assign to them, even if the model selects them confidently.

Choice requires both `minConfidence` and `minProbability`. Score uses `minConfidence`. Noul has no confidence statistic: it requires `max(p, 1-p) >= minProbability`. Thresholds are application policy, not guarantees of correctness.

Only one request per node runs at a time. New triggers overwrite the one pending sample rather than building a queue. `minIntervalMs` limits request frequency (default 100 ms); `maxAgeMs` discards old samples and results (default 2000 ms). Pause, configuration changes, and shutdown suppress obsolete results. Repeated failures back off for at least two seconds; identical errors are logged at most once every five seconds. The Studio gateway caps total concurrent upstream requests at four and returns a retryable status when full. Multiple nodes share your account's rate limit, so adjust their intervals accordingly. Telemetry stays on data ports, never service state fields.

API keys stay in Studio's provider settings. The engine automatically uses the Studio URL inherited when Studio launches it. For a standalone engine, set `studioUrl` explicitly; its fallback is `http://127.0.0.1:8210`. The Studio server must be running for inference.

TypeSafe's published Jev 1.13 API currently supports **text and JSON only**, not image, audio, or video inputs. Use an upstream perception/OCR node to produce text or structured observations for this provider.

Implementation references: [TypeSafe HTTP API](https://docs.typesafe.ai/api), [models and input support](https://docs.typesafe.ai/models), and [probability versus confidence](https://docs.typesafe.ai/confidence), checked September 28, 2026.
