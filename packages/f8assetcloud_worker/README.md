# f8assetcloud_worker

Cloudflare Worker + D1 backend for Feel8 asset management, rebuilt around `Hono + Better Auth`.

## Architecture

- Auth is provided by Better Auth at `/api/auth/*`
- Business APIs stay under `/v1/*`
- The long-term public API contract should be owned under `/v1/*`, not delegated to Better Auth internals
- OpenAPI docs for audited `/v1/*` endpoints are served at `/docs` and `/openapi.json`
- User and management console is the React app in `console_web`
- The web UI is served at `/console`, and `/` redirects to `/console/`
- Browser auth uses Better Auth cookie sessions
- Desktop auth uses worker-issued opaque `accessToken` + `refreshToken` pairs scoped to `/v1/*`
- `/api/auth/*` stays browser-only; desktop Bearer tokens are never accepted on Better Auth internals
- D1 schema starts from a single fresh migration: `migrations/0001_init.sql`

## Database model

Core auth tables:

- `user`
- `session`
- `account`
- `verification`

User contract:

- The public worker API exposes a single human-readable user field: `name`
- `displayName` is not part of the worker contract anymore
- The management console and `/v1/me` payloads should be treated as `name`-only user surfaces

Asset tables:

- `asset_heads`
- `variant_details`
- `asset_versions`
- `asset_subscriptions`

Asset storage contract:

- `asset_heads` stores current queryable metadata for all assets:
  - owner
  - visibility
  - current version number
  - name
  - description
  - tags
- `variant_details` stores current variant-specific metadata:
  - `variant_kind`
  - `base_node_type`
  - `service_class`
  - `operator_class`
- `asset_versions` stores versioned large payload blobs only:
  - revision
  - component versions store canonical session content `{ schemaVersion, layout }`
  - variant versions store canonical `spec`
- `GET .../content` reconstructs a full API `record` from current relational metadata plus the versioned blob payload
- Historical content is versioned; historical metadata is not. `/content` always returns the current canonical API `record`, reconstructed from the current head metadata plus the selected version blob.

## Auth features

- Email + password sign-in
- Email verification
- Password reset
- Google OAuth login
- Cloudflare Turnstile protection for public sign-up and password-reset-request flows
- Better Auth rate limiting backed by D1
- Worker-side rate limiting on desktop auth, mutations, subscriptions, and management routes
- Management permissions are backed by Better Auth role support
- Bootstrap management account creation from environment variables

## Main routes

Auth:

- `POST /api/auth/sign-up/email`
- `POST /api/auth/sign-in/email`
- `POST /api/auth/sign-out`
- `POST /api/auth/request-password-reset`
- `GET /api/auth/get-session`
- `GET /api/auth/callback/google`

App wrappers:

- `GET /v1/auth/providers`
- `POST /v1/auth/desktop/session`
- `POST /v1/auth/desktop/token`
- `POST /v1/auth/desktop/refresh`
- `POST /v1/auth/desktop/revoke`
- `GET /v1/auth/verify-email?token=...`
- `POST /v1/auth/reset-password`
- `GET /v1/me`
- `PUT /v1/me`
- `POST /v1/me/password`

Web UI:

- `GET /console`
- `GET /console/verify-email`
- `GET /console/reset-password`
- `GET /` redirects to `/console/`

OpenAPI:

- `GET /docs`
- `GET /openapi.json`

Assets:

- `GET /v1/variants`
- `POST /v1/variants`
- `GET /v1/variants/:variantId`
- `PUT /v1/variants/:variantId`
- `DELETE /v1/variants/:variantId`
- `GET /v1/variants/:variantId/versions`
- `GET /v1/variants/:variantId/versions/:versionNumber`
- `POST /v1/variants/:variantId/subscribe`
- `DELETE /v1/variants/:variantId/subscribe`
- `POST /v1/variants/:variantId/fork`
- `GET /v1/components`
- `POST /v1/components`
- `GET /v1/components/:componentId`
- `PUT /v1/components/:componentId`
- `DELETE /v1/components/:componentId`
- `GET /v1/components/:componentId/versions`
- `GET /v1/components/:componentId/versions/:versionNumber`
- `POST /v1/components/:componentId/subscribe`
- `DELETE /v1/components/:componentId/subscribe`
- `POST /v1/components/:componentId/fork`
- `GET /v1/components/:componentId/content`
- `GET /v1/components/:componentId/versions/:versionNumber/content`
- `GET /v1/variants/:variantId/content`
- `GET /v1/variants/:variantId/versions/:versionNumber/content`

Management:

- `GET /v1/management/users`
- `POST /v1/management/users`
- `GET /v1/management/users/:userId`
- `PUT /v1/management/users/:userId`
- `DELETE /v1/management/users/:userId`
- `GET /v1/management/site-settings`
- `PUT /v1/management/site-settings`
- `GET /v1/management/components`
- `GET /v1/management/components/:componentId`
- `PUT /v1/management/components/:componentId`
- `DELETE /v1/management/components/:componentId`
- `GET /v1/management/variants`
- `GET /v1/management/variants/:variantId`
- `PUT /v1/management/variants/:variantId`
- `DELETE /v1/management/variants/:variantId`

OpenAPI contract:

- `/openapi.json` and `/docs` should be treated as the canonical audited API contract for `/v1/*`
- When routes or payloads change, update `src/openapi.js` in the same change so the runtime docs stay in sync

## Environment

Required:

- `BETTER_AUTH_SECRET`
- `AUTH_BASE_URL`
- `BOOTSTRAP_ADMIN_NAME`
- `BOOTSTRAP_ADMIN_EMAIL`
- `BOOTSTRAP_ADMIN_PASSWORD`

Recommended:

- `AUTH_VERIFY_EMAIL_BASE_URL`
- `AUTH_RESET_PASSWORD_BASE_URL`
- `EMAIL_VERIFY_TOKEN_TTL_SECONDS`
- `PASSWORD_RESET_TOKEN_TTL_SECONDS`
- `ENABLE_ASSET_JSON_GZIP`
- `ENABLE_API_JSON_GZIP`

CORS:

- `CORS_ALLOWED_ORIGINS` — comma-separated list of extra allowed origins (e.g. `http://localhost:5173`)
- In production, only `AUTH_BASE_URL` and explicit `CORS_ALLOWED_ORIGINS` are trusted; the worker no longer trusts arbitrary request origins

Google login:

- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`

Cloudflare Turnstile:

- `TURNSTILE_SITE_KEY` (or `CLOUDFLARE_TURNSTILE_SITE_KEY`)
- `TURNSTILE_SECRET_KEY` (or `CLOUDFLARE_TURNSTILE_SECRET_KEY`)

Email delivery via Resend:

- `AUTH_EMAIL_FROM`
- `RESEND_API_KEY`

Local debug only:

- `EXPOSE_DEBUG_AUTH_LINKS=true` — prints verification/reset links to console (only effective when email delivery is not configured)
- `ENABLE_ASSET_JSON_GZIP=true` — targeted default; compresses large asset content responses such as `/v1/components/:componentId/content` without touching auth routes
- `ENABLE_API_JSON_GZIP=false` — broad `/v1/*` compression; keep this off unless you intentionally want to gzip nearly every app JSON response

Variable precedence:

- `wrangler.toml` `[vars]` provides checked-in defaults for deploys and local dev.
- `.dev.vars` overrides those values during `wrangler dev`, so local debugging should usually be adjusted there.

## Security hard requirements

- Set `AUTH_BASE_URL` in every deployed environment. The worker now fails closed when it is missing.
- Deploy Cloudflare WAF / edge rate limiting for anonymous public traffic. Anonymous browse endpoints stay open by design and should not rely on D1-backed application rate limits alone.
- Configure Turnstile when public registration or password reset is enabled.
- Store `TURNSTILE_SECRET_KEY` as a Wrangler secret, not in `wrangler.toml`.
- Keep `CORS_ALLOWED_ORIGINS` minimal. Do not mirror arbitrary request origins.
- Treat every open-source client as untrusted. The server must enforce authorization, quotas, and input limits even when a client knows the protocol.
- Browser-cookie `POST/PUT/PATCH/DELETE` requests to `/v1/*` require allowed `Origin`; desktop Bearer tokens bypass browser CSRF rules and must use `Authorization: Bearer ...`.
- Owner-facing subscriber list endpoints no longer expose subscriber email addresses. Email is admin-only data.
- Large request bodies are size-checked both before and after gzip decompression; over-limit uploads are rejected with `413`.

## Native client browser sign-in notes

The native-client authorization contract uses the system browser plus a temporary loopback callback (`http://127.0.0.1:<port>/callback`). The registered Studio client id is `f8studio`.

Desktop flow summary:

- `GET /v1/auth/desktop/authorize` always renders a confirmation page
- `POST /v1/auth/desktop/authorize` confirms the request with CSRF + allowed-origin checks
- `POST /v1/auth/desktop/token` exchanges a one-time code for a desktop token pair
- `POST /v1/auth/desktop/refresh` rotates the short-lived access token
- `POST /v1/auth/desktop/revoke` invalidates the saved refresh token

Clients must store only the desktop `refreshToken` in the OS credential store. Browser session cookies are not persisted locally.

Key worker variables are `AUTH_BASE_URL` and, for cross-origin local development, `CORS_ALLOWED_ORIGINS`.

Friendly callback pages now live at:

- `/auth-complete`
- `/auth-error`


## Local development

```bash
cd packages/f8assetcloud_worker
cp .dev.vars.example .dev.vars
npm install
npm --prefix console_web install
npm run web:dev
npm run d1:migrate:local
npx wrangler dev
```

Single-origin mode:

```bash
cd packages/f8assetcloud_worker
npm run dev:single
```

## Deployment

```bash
cd packages/f8assetcloud_worker
npm run web:build
npm run d1:migrate
npx wrangler deploy
```

## Rebuild D1 databases

Production reset:

```bash
cd packages/f8assetcloud_worker
npx wrangler d1 delete feel8-assets -y
npx wrangler d1 create feel8-assets --location enam
```

After `wrangler d1 create`, copy the new UUID into `[[d1_databases]].database_id` in `wrangler.toml`, then run:

```bash
cd packages/f8assetcloud_worker
npm run d1:migrate
npm run web:build
npm run deploy
```

Preview reset:

```bash
cd packages/f8assetcloud_worker
npx wrangler d1 delete feel8-assets-preview -y
npx wrangler d1 create feel8-assets-preview --location enam
```

After `wrangler d1 create`, copy the new UUID into `[[d1_databases]].preview_database_id` in `wrangler.toml`, then run:

```bash
cd packages/f8assetcloud_worker
npm run d1:migrate:preview
```

If you recreate both databases, update both IDs in `wrangler.toml` before deploying.

Reset an existing D1 database in place without creating a new database ID:

Production:

```bash
cd packages/f8assetcloud_worker
npm run d1:reset:remote
```

Preview:

```bash
cd packages/f8assetcloud_worker
npm run d1:reset:preview
```

This in-place reset approach uses an explicit ordered table drop list that is compatible with remote D1 execution, then reapplies the current baseline migration.
The reset SQL lives in [scripts/reset_d1.sh](/home/sxs/SS/Feel8/f8studio/packages/f8assetcloud_worker/scripts/reset_d1.sh) so the drop order stays centralized instead of being duplicated inside `package.json`.
By default the script reads `database_name` from [wrangler.toml](/home/sxs/SS/Feel8/f8studio/packages/f8assetcloud_worker/wrangler.toml). You can override it with `D1_DATABASE_NAME=...` or point to a different config with `WRANGLER_TOML_PATH=...`.

## Notes

- This package now assumes a brand-new database baseline.
- Old JWT auth tables and compatibility migrations have been removed.
- To fully reset locally or in a disposable environment, recreate the D1 database and apply `0001_init.sql`.
- A cron trigger runs daily at 03:00 UTC to clean up expired sessions.
- Asset version `content` is stored as a GZIP-compressed BLOB in D1.
- HTTP compression is negotiated only at the transport layer with standard `Content-Encoding` / `Accept-Encoding`.
- The worker does not use field-level compression contracts. The canonical stored payload itself is already the minimal versioned blob:
  - component: `{ schemaVersion, layout }`
  - variant: `spec`
- Stored version blobs are validated strictly on read:
  - component blobs must be the canonical `{ schemaVersion, layout }` object
  - variant blobs must be the raw `spec` object
  - legacy record wrappers and response envelopes are treated as invalid stored content
- Limit: 10 MB per version before storage compression.
- New application-owned endpoints should be added to the OpenAPI contract in `src/openapi.js` as part of the route change, so docs and clients do not drift from implementation.
