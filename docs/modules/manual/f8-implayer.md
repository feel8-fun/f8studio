### Recommended Use Cases

- Feed local media into shared memory for detection/pose services.
- Validate end-to-end pipelines with deterministic playback.

### Minimal Run Example

```bash
services/f8/implayer/linux/f8implayer_service
```

### Common Pitfalls

- Missing codec/runtime libraries on target host.
- Wrong SHM name wiring disconnects downstream consumers.
- For website URLs, guest mode may provide lower quality streams if auth cookies are not configured.

### Cookie Auth For Online URLs

`f8.implayer` supports cookie-based auth for `yt-dlp/mpv` via state fields:

- `authMode`: `none | browser | cookiesFile` (default `none`)
- `authBrowser`: `chrome | chromium | edge | firefox | safari`
- `authBrowserProfile`: optional profile name for browser mode
- `authCookiesFile`: cookies.txt path for `cookiesFile` mode

Examples:

- Browser cookies (Chrome default profile):
  - `authMode=browser`
  - `authBrowser=chrome`
- Browser cookies (explicit profile):
  - `authMode=browser`
  - `authBrowser=chrome`
  - `authBrowserProfile=Default`
- cookies.txt file:
  - `authMode=cookiesFile`
  - `authCookiesFile=/path/to/cookies.txt`

Security and persistence:

- `authBrowserProfile` and `authCookiesFile` are treated as sensitive runtime values.
- They are not written back through the service KV endpoint path and are not applied from rungraph snapshots.

### Troubleshooting

- Verify media playback locally before integrating downstream nodes.
- Confirm `shmName` is identical across producer and consumers.
- Browser mode requires a logged-in browser profile on the same machine.
- cookies file mode requires a valid, existing cookies file path.
