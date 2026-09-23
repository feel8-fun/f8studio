import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { gunzipSync, gzipSync } from 'node:zlib';
import { hashPassword as hashLegacyBetterAuthPassword } from 'better-auth/crypto';

import { createApp, resetWorkerCachesForTesting } from '../src/app.js';
import worker from '../src/index.js';
import { authPasswordHashVersion, hashAuthPassword, verifyAuthPassword } from '../src/password.js';
import { createSqliteD1Database } from '../test_support/sqlite_d1_adapter.js';

const TEST_AUTH_SECRET = '0123456789abcdef0123456789abcdef';
const TEST_PASSWORD = 'password123';
const TEST_PASSWORD_2 = 'password456';
const TEST_PASSWORD_3 = 'password789';
const MANAGEMENT_API_BASE_PATH = '/v1/management';

const migrationsDir = path.join(import.meta.dirname, '..', 'migrations');
const migrationsSql = fs.readdirSync(migrationsDir)
  .filter((filename) => filename.endsWith('.sql'))
  .sort()
  .map((filename) => fs.readFileSync(path.join(migrationsDir, filename), 'utf8'))
  .join('\n\n');

function createEnv({ allowUserRegistration = true, ...overrides } = {}) {
  const env = {
    DB: createSqliteD1Database({ migrationsSql }),
    BETTER_AUTH_SECRET: TEST_AUTH_SECRET,
    AUTH_BASE_URL: 'http://worker.test',
    BOOTSTRAP_ADMIN_USERNAME: 'admin',
    BOOTSTRAP_ADMIN_DISPLAY_NAME: 'Administrator',
    BOOTSTRAP_ADMIN_PASSWORD: TEST_PASSWORD,
    BOOTSTRAP_ADMIN_EMAIL: 'admin@example.com',
    EXPOSE_DEBUG_AUTH_LINKS: 'true',
    ...overrides,
  };
  env.DB.prepare(
    'UPDATE site_settings SET allow_user_registration = ?, updated_at = CURRENT_TIMESTAMP WHERE id = 1',
  )
    .bind(allowUserRegistration ? 1 : 0)
    .run();
  return env;
}

async function jsonRequest(app, env, pathname, { method, payload, cookie, origin } = {}) {
  const requestMethod = method || 'GET';
  const headers = {};
  if (payload !== undefined) {
    headers['Content-Type'] = 'application/json';
  }
  if (cookie) {
    headers.cookie = cookie;
  }
  if (origin !== undefined) {
    headers.origin = origin;
  } else if (
    cookie
    && requestMethod !== 'GET'
    && requestMethod !== 'HEAD'
    && String(pathname).startsWith('/v1/')
  ) {
    headers.origin = new URL(String(env.AUTH_BASE_URL || 'http://worker.test')).origin;
  }
  const request = new Request(`http://worker.test${pathname}`, {
    method: requestMethod,
    headers,
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });
  const response = await app.fetch(request, env, {});
  const bodyText = await response.text();
  return {
    status: response.status,
    json: bodyText ? JSON.parse(bodyText) : {},
    text: bodyText,
    headers: response.headers,
  };
}

async function captureConsoleInfo(run) {
  const logs = [];
  const originalInfo = console.info;
  console.info = (...args) => {
    logs.push(args.map((value) => String(value)).join(' '));
  };
  try {
    const result = await run();
    return { result, logs };
  } finally {
    console.info = originalInfo;
  }
}

function extractDebugToken(logs, label) {
  const prefix = `[auth debug] ${label}: `;
  const line = [...logs].reverse().find((entry) => entry.startsWith(prefix));
  if (!line) {
    return '';
  }
  const url = new URL(line.slice(prefix.length));
  const queryToken = String(url.searchParams.get('token') || '');
  if (queryToken) {
    return queryToken;
  }
  const pathParts = url.pathname.split('/').filter((part) => part.length > 0);
  const resetPasswordIndex = pathParts.findIndex((part) => part === 'reset-password');
  if (resetPasswordIndex >= 0 && resetPasswordIndex + 1 < pathParts.length) {
    return String(pathParts[resetPasswordIndex + 1] || '');
  }
  return '';
}

function responseCookie(headers) {
  const setCookie = headers.get('set-cookie');
  return setCookie ? String(setCookie).split(';')[0] : '';
}

function desktopCsrfTokenFromHtml(html) {
  const match = /name="csrf_token" value="([^"]+)"/.exec(String(html || ''));
  return match ? String(match[1] || '') : '';
}

function pkceChallenge(value) {
  return createHash('sha256').update(String(value || ''), 'utf8').digest('base64url');
}

function wrapBlobRowsAsDataArrays(db) {
  const originalPrepare = db.prepare.bind(db);

  function wrapRow(row) {
    if (row === null || row === undefined) {
      return row;
    }
    if (!Object.hasOwn(row, 'content')) {
      return row;
    }
    const content = row.content;
    if (content instanceof Uint8Array) {
      return {
        ...row,
        content: {
          data: Array.from(content),
        },
      };
    }
    if (ArrayBuffer.isView(content)) {
      return {
        ...row,
        content: {
          data: Array.from(new Uint8Array(content.buffer, content.byteOffset, content.byteLength)),
        },
      };
    }
    return row;
  }

  function wrapPrepared(prepared) {
    return {
      bind(...values) {
        return wrapPrepared(prepared.bind(...values));
      },
      async first() {
        return wrapRow(await prepared.first());
      },
      async all() {
        const result = await prepared.all();
        return {
          ...result,
          results: Array.isArray(result.results) ? result.results.map((row) => wrapRow(row)) : result.results,
        };
      },
      async run() {
        return prepared.run();
      },
      async raw() {
        return prepared.raw();
      },
    };
  }

  db.prepare = function prepare(sql) {
    return wrapPrepared(originalPrepare(sql));
  };
}

async function signUpUser(app, env, { name, email, password = TEST_PASSWORD }) {
  const { result, logs } = await captureConsoleInfo(() => jsonRequest(app, env, '/api/auth/sign-up/email', {
    method: 'POST',
    payload: {
      email,
      password,
      name,
    },
  }));
  return {
    ...result,
    verifyToken: extractDebugToken(logs, 'verify email'),
  };
}

async function verifyUserEmail(app, env, token) {
  return jsonRequest(app, env, `/v1/auth/verify-email?token=${encodeURIComponent(token)}`);
}

async function signInUser(app, env, { email, password = TEST_PASSWORD }) {
  const result = await jsonRequest(app, env, '/api/auth/sign-in/email', {
    method: 'POST',
    payload: {
      email,
      password,
    },
  });
  return {
    ...result,
    cookie: responseCookie(result.headers),
  };
}

async function requestPasswordReset(app, env, email) {
  const { result, logs } = await captureConsoleInfo(() => jsonRequest(app, env, '/api/auth/request-password-reset', {
    method: 'POST',
    payload: { email },
  }));
  return {
    ...result,
    resetToken: extractDebugToken(logs, 'reset password'),
  };
}

async function createVerifiedSession(app, env, { name, email, password = TEST_PASSWORD }) {
  const signedUp = await signUpUser(app, env, { name, email, password });
  assert.equal(signedUp.status, 200);
  assert.ok(signedUp.verifyToken);

  const verified = await verifyUserEmail(app, env, signedUp.verifyToken);
  assert.equal(verified.status, 200);

  const signedIn = await signInUser(app, env, { email, password });
  assert.equal(signedIn.status, 200);
  assert.ok(signedIn.cookie);

  return {
    userId: String(signedUp.json.user.id),
    cookie: signedIn.cookie,
    signUp: signedUp,
    signIn: signedIn,
  };
}

function variantPayload({ variantId, name, visibility = 'private', versionNumber } = {}) {
  return {
    record: {
      variantId: variantId || 'variant-1',
      kind: 'operator',
      baseNodeType: 'svc.base.op',
      serviceClass: 'svc.test',
      operatorClass: 'op.test',
      name: name || 'Variant 1',
      description: 'desc',
      tags: ['vision'],
      spec: { label: name || 'Variant 1' },
      createdAt: '2026-01-01T00:00:00.000Z',
      updatedAt: '2026-01-01T00:00:00.000Z',
    },
    visibility,
    versionNumber,
    changeSummary: 'save',
  };
}

function componentPayload({ componentId, name, visibility = 'private', versionNumber } = {}) {
  return {
    record: {
      componentId: componentId || 'component-1',
      name: name || 'Component 1',
      description: 'published session',
      tags: ['session'],
      content: {
        schemaVersion: 'f8studio-session/1',
        layout: {
          nodes: {
            n1: {
              type_: 'svc.f8.implayer',
              custom: {
                authCookiesFile: '',
              },
            },
          },
          connections: [],
        },
      },
      createdAt: '2026-01-01T00:00:00.000Z',
      updatedAt: '2026-01-01T00:00:00.000Z',
    },
    visibility,
    versionNumber,
    changeSummary: 'save',
  };
}

function moddingRecipePayload({ recipeId, name, visibility = 'private', versionNumber, contentOverrides = {} } = {}) {
  return {
    record: {
      recipeId: recipeId || 'recipe-1',
      name: name || 'Unity Recipe',
      description: 'Shareable Unity skeleton stream setup',
      tags: ['modding', 'unity'],
      content: {
        schemaVersion: 'f8moddingrecipe/1',
        engine: 'unity',
        backend: 'mono',
        gameProfile: {
          profileId: 'CUSTOM',
          name: 'Example Unity Game',
          processAliases: ['ExampleGame'],
        },
        installer: {
          requiredToolVersion: 'f8unitymods/0.1',
          selectedExporter: 'F8SkeletonStreamer',
          optionalUtilities: {
            installRuntimeUnityEditor: false,
            installCinematicUnityExplorer: true,
            installConfigurationManager: false,
            installUniversalUnityDemosaics: false,
          },
          releases: {
            bepinex: {
              version: '6.0.0-be.735',
              assetName: 'BepInEx-Unity.Mono-win-x64.zip',
            },
          },
        },
        payloads: {
          exporterConfig: {
            udpHost: '127.0.0.1',
            udpPort: 39540,
          },
          profileJson: {
            exporterKey: 'F8SkeletonStreamer',
          },
        },
        pyStudio: {
          graphFragment: {
            nodes: ['udp-in', 'skeleton-decoder', 'viz-3d'],
          },
        },
        verification: {
          udpPort: 39540,
          sampleKeys: ['hips', 'head'],
          timestamp: '2026-01-01T00:00:00.000Z',
        },
        notes: 'Install BepInEx, launch the game, then verify UDP 39540.',
        ...contentOverrides,
      },
      createdAt: '2026-01-01T00:00:00.000Z',
      updatedAt: '2026-01-01T00:00:00.000Z',
    },
    visibility,
    versionNumber,
    changeSummary: 'save',
  };
}

test('auth flows use Better Auth cookie sessions and email actions', async (t) => {
  const env = createEnv();
  t.after(() => {
    resetWorkerCachesForTesting();
    env.DB.close();
  });
  const app = createApp();

  const providers = await jsonRequest(app, env, '/v1/auth/providers');
  assert.equal(providers.status, 200);
  assert.equal(providers.json.google, false);

  const siteSettings = await jsonRequest(app, env, '/v1/site-settings');
  assert.equal(siteSettings.status, 200);
  assert.equal(siteSettings.json.allowUserRegistration, true);

  const signedUp = await signUpUser(app, env, {
    name: 'Alice',
    email: 'alice@example.com',
  });
  assert.equal(signedUp.status, 200);
  assert.equal(signedUp.json.user.name, 'Alice');
  assert.equal(signedUp.json.user.emailVerified, false);
  assert.ok(signedUp.verifyToken);

  const loginBeforeVerify = await signInUser(app, env, {
    email: 'alice@example.com',
  });
  assert.equal(loginBeforeVerify.status, 403);
  assert.equal(loginBeforeVerify.json.code, 'EMAIL_NOT_VERIFIED');

  const verified = await verifyUserEmail(app, env, signedUp.verifyToken);
  assert.equal(verified.status, 200);
  assert.equal(verified.json.verified, true);

  const duplicate = await jsonRequest(app, env, '/api/auth/sign-up/email', {
    method: 'POST',
    payload: {
      email: 'alice2@example.com',
      password: TEST_PASSWORD,
      name: 'Alice',
    },
  });
  assert.equal(duplicate.status, 422);

  const loginFail = await signInUser(app, env, {
    email: 'alice@example.com',
    password: 'wrong-password',
  });
  assert.equal(loginFail.status, 401);

  const signedIn = await signInUser(app, env, {
    email: 'alice@example.com',
  });
  assert.equal(signedIn.status, 200);
  assert.ok(signedIn.cookie);

  const me = await jsonRequest(app, env, '/v1/me', { cookie: signedIn.cookie });
  assert.equal(me.status, 200);
  assert.equal(Object.hasOwn(me.json, 'displayName'), false);
  assert.equal(me.json.email, 'alice@example.com');
  assert.equal(me.json.emailVerified, true);

  const renamed = await jsonRequest(app, env, '/v1/me', {
    method: 'PUT',
    cookie: signedIn.cookie,
    payload: {
      name: 'Alice Updated',
    },
  });
  assert.equal(renamed.status, 200);
  assert.equal(renamed.json.name, 'Alice Updated');
  assert.equal(Object.hasOwn(renamed.json, 'displayName'), false);

  const meAfterRename = await jsonRequest(app, env, '/v1/me', { cookie: signedIn.cookie });
  assert.equal(meAfterRename.status, 200);
  assert.equal(meAfterRename.json.name, 'Alice Updated');

  const changedPassword = await jsonRequest(app, env, '/v1/me/password', {
    method: 'POST',
    cookie: signedIn.cookie,
    payload: {
      currentPassword: TEST_PASSWORD,
      newPassword: TEST_PASSWORD_2,
    },
  });
  assert.equal(changedPassword.status, 200);

  const oldLogin = await signInUser(app, env, {
    email: 'alice@example.com',
    password: TEST_PASSWORD,
  });
  assert.equal(oldLogin.status, 401);

  const newLogin = await signInUser(app, env, {
    email: 'alice@example.com',
    password: TEST_PASSWORD_2,
  });
  assert.equal(newLogin.status, 200);

  const resetRequest = await requestPasswordReset(app, env, 'alice@example.com');
  assert.equal(resetRequest.status, 200);
  assert.equal(resetRequest.json.status, true);
  assert.ok(resetRequest.resetToken);

  const secondResetRequest = await requestPasswordReset(app, env, 'alice@example.com');
  assert.equal(secondResetRequest.status, 200);
  assert.equal(secondResetRequest.json.status, true);
  assert.ok(secondResetRequest.resetToken);

  const reset = await jsonRequest(app, env, '/v1/auth/reset-password', {
    method: 'POST',
    payload: {
      token: resetRequest.resetToken,
      newPassword: TEST_PASSWORD_3,
    },
  });
  assert.equal(reset.status, 200);
  assert.equal(reset.json.reset, true);

  const resetLogin = await signInUser(app, env, {
    email: 'alice@example.com',
    password: TEST_PASSWORD_3,
  });
  assert.equal(resetLogin.status, 200);
});

test('Better Auth database rate limit writes use the current rateLimit schema', async (t) => {
  const env = createEnv();
  t.after(() => {
    resetWorkerCachesForTesting();
    env.DB.close();
  });
  const app = createApp();

  const response = await app.fetch(new Request('http://worker.test/api/auth/get-session', {
    headers: {
      'x-forwarded-for': '203.0.113.10',
    },
  }), env, {});

  assert.equal(response.status, 200);
  const rateLimitRow = await env.DB
    .prepare('SELECT id, key, count, lastRequest FROM rateLimit LIMIT 1')
    .first();
  assert.equal(typeof rateLimitRow.id, 'string');
  assert.ok(rateLimitRow.id.length > 0);
  assert.match(String(rateLimitRow.key), /203\.0\.113\.10/);
  assert.equal(rateLimitRow.count, 1);
  assert.equal(typeof rateLimitRow.lastRequest, 'number');
});

test('loopback dev origin is trusted for same-origin auth requests', async (t) => {
  const env = createEnv({
    AUTH_BASE_URL: 'https://assetcloud.feel8.fun',
    CORS_ALLOWED_ORIGINS: '',
  });
  t.after(() => {
    resetWorkerCachesForTesting();
    env.DB.close();
  });
  const app = createApp();

  const response = await app.fetch(new Request('http://localhost:8787/api/auth/sign-in/email', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      origin: 'http://localhost:8787',
      cookie: 'test=1',
    },
    body: JSON.stringify({
      email: 'missing@example.com',
      password: TEST_PASSWORD,
    }),
  }), env, {});

  assert.notEqual(response.status, 403);
  assert.equal(response.headers.get('access-control-allow-origin'), 'http://localhost:8787');
});

test('bootstrap admin sync avoids rotating credentials after a cold-cache login', async (t) => {
  const env = createEnv({ allowUserRegistration: false });
  t.after(() => {
    resetWorkerCachesForTesting();
    env.DB.close();
  });

  const firstApp = createApp();
  const firstLogin = await signInUser(firstApp, env, {
    email: 'admin@example.com',
  });
  assert.equal(firstLogin.status, 200);
  assert.ok(firstLogin.cookie);

  const firstAccount = await env.DB.prepare(
    `SELECT
       a.id,
       a.password,
       a.updatedAt AS updated_at
     FROM account a
     JOIN user u ON u.id = a.userId
     WHERE u.email = ? AND a.providerId = 'credential'
     LIMIT 1`,
  )
    .bind('admin@example.com')
    .first();
  assert.notEqual(firstAccount, null);

  const bootstrapState = await env.DB.prepare(
    `SELECT config_fingerprint, user_id
     FROM bootstrap_admin_state
     WHERE id = 1`,
  ).first();
  assert.notEqual(bootstrapState, null);

  resetWorkerCachesForTesting();

  const secondApp = createApp();
  const secondLogin = await signInUser(secondApp, env, {
    email: 'admin@example.com',
  });
  assert.equal(secondLogin.status, 200);
  assert.ok(secondLogin.cookie);

  const secondAccount = await env.DB.prepare(
    `SELECT
       a.id,
       a.password,
       a.updatedAt AS updated_at
     FROM account a
     JOIN user u ON u.id = a.userId
     WHERE u.email = ? AND a.providerId = 'credential'
     LIMIT 1`,
  )
    .bind('admin@example.com')
    .first();
  assert.notEqual(secondAccount, null);
  assert.equal(secondAccount.id, firstAccount.id);
  assert.equal(secondAccount.password, firstAccount.password);
  assert.equal(secondAccount.updated_at, firstAccount.updated_at);
});

test('worker password hash verifies credentials without Better Auth scrypt', async () => {
  const hash = await hashAuthPassword(TEST_PASSWORD);
  assert.match(hash, /^f8pbkdf2-sha256-v1\$50000\$/);
  assert.equal(hash.includes(':'), false);
  assert.equal(await verifyAuthPassword({ hash, password: TEST_PASSWORD }), true);
  assert.equal(await verifyAuthPassword({ hash, password: TEST_PASSWORD_2 }), false);
});

test('bootstrap admin sync replaces legacy Better Auth password hashes', async (t) => {
  const env = createEnv({ allowUserRegistration: false });
  t.after(() => {
    resetWorkerCachesForTesting();
    env.DB.close();
  });

  const app = createApp();
  const firstLogin = await signInUser(app, env, {
    email: 'admin@example.com',
  });
  assert.equal(firstLogin.status, 200);

  const legacyHash = await hashLegacyBetterAuthPassword(TEST_PASSWORD_2);
  await env.DB.prepare(
    `UPDATE account
     SET password = ?
     WHERE providerId = 'credential'
       AND userId = (
         SELECT id
         FROM user
         WHERE email = ?
         LIMIT 1
       )`,
  )
    .bind(legacyHash, 'admin@example.com')
    .run();
  await env.DB.prepare('DELETE FROM bootstrap_admin_state WHERE id = 1').run();

  resetWorkerCachesForTesting();

  const syncedApp = createApp();
  const syncedLogin = await signInUser(syncedApp, env, {
    email: 'admin@example.com',
  });
  assert.equal(syncedLogin.status, 200);
  assert.ok(syncedLogin.cookie);

  const account = await env.DB.prepare(
    `SELECT a.password
     FROM account a
     JOIN user u ON u.id = a.userId
     WHERE u.email = ? AND a.providerId = 'credential'
     LIMIT 1`,
  )
    .bind('admin@example.com')
    .first();
  assert.notEqual(account, null);
  assert.match(String(account.password), /^f8pbkdf2-sha256-v1\$50000\$/);
  assert.equal(await verifyAuthPassword({ hash: account.password, password: TEST_PASSWORD }), true);

  const bootstrapState = await env.DB.prepare(
    `SELECT config_fingerprint
     FROM bootstrap_admin_state
     WHERE id = 1`,
  ).first();
  assert.notEqual(bootstrapState, null);
  assert.ok(String(bootstrapState.config_fingerprint || '').length > 0);
  assert.ok(authPasswordHashVersion().startsWith('f8pbkdf2-sha256-v1:'));
});

test('sign-out requires Origin header and deletes the current session when provided', async (t) => {
  const env = createEnv({ allowUserRegistration: false });
  t.after(() => {
    resetWorkerCachesForTesting();
    env.DB.close();
  });
  const app = createApp();

  const signedIn = await signInUser(app, env, {
    email: 'admin@example.com',
  });
  assert.equal(signedIn.status, 200);
  assert.ok(signedIn.cookie);

  const beforeSignOut = await env.DB.prepare(
    'SELECT COUNT(*) AS count FROM session',
  ).first();
  assert.equal(Number(beforeSignOut?.count ?? 0), 1);

  const missingOrigin = await jsonRequest(app, env, '/api/auth/sign-out', {
    method: 'POST',
    payload: {},
    cookie: signedIn.cookie,
  });
  assert.equal(missingOrigin.status, 403);
  assert.equal(missingOrigin.json.code, 'MISSING_OR_NULL_ORIGIN');

  const afterRejectedSignOut = await env.DB.prepare(
    'SELECT COUNT(*) AS count FROM session',
  ).first();
  assert.equal(Number(afterRejectedSignOut?.count ?? 0), 1);

  const acceptedSignOut = await jsonRequest(app, env, '/api/auth/sign-out', {
    method: 'POST',
    payload: {},
    cookie: signedIn.cookie,
    origin: 'http://worker.test',
  });
  assert.equal(acceptedSignOut.status, 200);
  assert.equal(acceptedSignOut.json.success, true);

  const afterAcceptedSignOut = await env.DB.prepare(
    'SELECT COUNT(*) AS count FROM session',
  ).first();
  assert.equal(Number(afterAcceptedSignOut?.count ?? 0), 0);
});

test('providers endpoint reflects Google auth configuration', async (t) => {
  const env = createEnv({
    GOOGLE_CLIENT_ID: 'google-client-id',
    GOOGLE_CLIENT_SECRET: 'google-client-secret',
  });
  t.after(() => env.DB.close());
  const app = createApp();

  const providers = await jsonRequest(app, env, '/v1/auth/providers');
  assert.equal(providers.status, 200);
  assert.equal(providers.json.google, true);
});

test('desktop browser auth renders a confirmation page and exchanges an authorization code for desktop tokens', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const signedUp = await signUpUser(app, env, {
    name: 'Desktop User',
    email: 'desktop@example.com',
  });
  assert.equal(signedUp.status, 200);
  assert.ok(signedUp.verifyToken);
  const verified = await verifyUserEmail(app, env, signedUp.verifyToken);
  assert.equal(verified.status, 200);

  const redirectUri = 'http://127.0.0.1:41811/callback';
  const state = 'desktop-state-1';
  const codeVerifier = 'desktop-code-verifier-1';
  const codeChallenge = pkceChallenge(codeVerifier);
  const authorizePath = `/v1/auth/desktop/authorize?client_id=f8studio&redirect_uri=${encodeURIComponent(redirectUri)}&state=${encodeURIComponent(state)}&code_challenge=${encodeURIComponent(codeChallenge)}&code_challenge_method=S256`;

  const pageResponse = await app.fetch(new Request(`http://worker.test${authorizePath}`), env, {});
  assert.equal(pageResponse.status, 200);
  assert.match(pageResponse.headers.get('content-type') || '', /text\/html/);
  const pageHtml = await pageResponse.text();
  assert.match(pageHtml, /Continue to Feel8 Studio/);
  assert.match(pageHtml, /name="email"/);
  const pageCsrfCookie = responseCookie(pageResponse.headers);
  const pageCsrfToken = desktopCsrfTokenFromHtml(pageHtml);
  assert.ok(pageCsrfCookie);
  assert.ok(pageCsrfToken);

  const authorizeBody = new URLSearchParams({
    client_id: 'f8studio',
    redirect_uri: redirectUri,
    state,
    code_challenge: codeChallenge,
    code_challenge_method: 'S256',
    csrf_token: pageCsrfToken,
    email: 'desktop@example.com',
    password: TEST_PASSWORD,
  });
  const authorizeResponse = await app.fetch(new Request('http://worker.test/v1/auth/desktop/authorize', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      cookie: pageCsrfCookie,
      origin: 'http://worker.test',
    },
    body: authorizeBody.toString(),
  }), env, {});
  assert.equal(authorizeResponse.status, 302);
  const callbackLocation = authorizeResponse.headers.get('location') || '';
  const callbackUrl = new URL(callbackLocation);
  assert.equal(callbackUrl.origin + callbackUrl.pathname, redirectUri);
  assert.equal(callbackUrl.searchParams.get('state'), state);
  const code = String(callbackUrl.searchParams.get('code') || '');
  assert.ok(code);

  const tokenResponse = await jsonRequest(app, env, '/v1/auth/desktop/token', {
    method: 'POST',
    payload: {
      clientId: 'f8studio',
      redirectUri,
      code,
      codeVerifier,
    },
  });
  assert.equal(tokenResponse.status, 200);
  assert.equal(tokenResponse.json.user.email, 'desktop@example.com');
  assert.ok(String(tokenResponse.json.accessToken || '').trim());
  assert.ok(String(tokenResponse.json.refreshToken || '').trim());

  const meResponseRaw = await app.fetch(new Request('http://worker.test/v1/me', {
    headers: {
      Authorization: `Bearer ${String(tokenResponse.json.accessToken || '')}`,
    },
  }), env, {});
  assert.equal(meResponseRaw.status, 200);
  assert.equal((await meResponseRaw.json()).email, 'desktop@example.com');

  const secondExchange = await jsonRequest(app, env, '/v1/auth/desktop/token', {
    method: 'POST',
    payload: {
      clientId: 'f8studio',
      redirectUri,
      code,
      codeVerifier,
    },
  });
  assert.equal(secondExchange.status, 400);
  assert.match(String(secondExchange.json.message || ''), /already been used/i);
});

test('desktop browser auth can authorize directly from an existing browser session', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const desktopUser = await createVerifiedSession(app, env, {
    name: 'Desktop Existing Session',
    email: 'desktop-session@example.com',
  });

  const redirectUri = 'http://127.0.0.1:41999/callback';
  const state = 'desktop-state-existing';
  const codeVerifier = 'desktop-code-verifier-existing';
  const codeChallenge = pkceChallenge(codeVerifier);
  const authorizePageResponse = await app.fetch(new Request(
    `http://worker.test/v1/auth/desktop/authorize?client_id=f8studio&redirect_uri=${encodeURIComponent(redirectUri)}&state=${encodeURIComponent(state)}&code_challenge=${encodeURIComponent(codeChallenge)}&code_challenge_method=S256`,
    {
      headers: {
        cookie: desktopUser.cookie,
      },
    },
  ), env, {});
  assert.equal(authorizePageResponse.status, 200);
  const authorizePageHtml = await authorizePageResponse.text();
  assert.match(authorizePageHtml, /Continue as/);
  const authorizeBody = new URLSearchParams({
    client_id: 'f8studio',
    redirect_uri: redirectUri,
    state,
    code_challenge: codeChallenge,
    code_challenge_method: 'S256',
    csrf_token: desktopCsrfTokenFromHtml(authorizePageHtml),
  });
  const authorizeResponse = await app.fetch(new Request('http://worker.test/v1/auth/desktop/authorize', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      cookie: `${desktopUser.cookie}; ${responseCookie(authorizePageResponse.headers)}`,
      origin: 'http://worker.test',
    },
    body: authorizeBody.toString(),
  }), env, {});
  assert.equal(authorizeResponse.status, 302);
  const callbackUrl = new URL(String(authorizeResponse.headers.get('location') || ''));
  assert.equal(callbackUrl.searchParams.get('state'), state);
  const code = String(callbackUrl.searchParams.get('code') || '');
  assert.ok(code);

  const tokenResponse = await jsonRequest(app, env, '/v1/auth/desktop/token', {
    method: 'POST',
    payload: {
      clientId: 'f8studio',
      redirectUri,
      code,
      codeVerifier,
    },
  });
  assert.equal(tokenResponse.status, 200);
  assert.equal(tokenResponse.json.user.email, 'desktop-session@example.com');
});

test('desktop refresh rotates tokens and revoke invalidates the refresh token', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const signedUp = await signUpUser(app, env, {
    name: 'Desktop Refresh User',
    email: 'desktop-refresh@example.com',
  });
  assert.equal(signedUp.status, 200);
  assert.ok(signedUp.verifyToken);
  const verified = await verifyUserEmail(app, env, signedUp.verifyToken);
  assert.equal(verified.status, 200);

  const redirectUri = 'http://127.0.0.1:42005/callback';
  const state = 'desktop-refresh-state';
  const codeVerifier = 'desktop-refresh-verifier';
  const codeChallenge = pkceChallenge(codeVerifier);
  const authorizePath = `/v1/auth/desktop/authorize?client_id=f8studio&redirect_uri=${encodeURIComponent(redirectUri)}&state=${encodeURIComponent(state)}&code_challenge=${encodeURIComponent(codeChallenge)}&code_challenge_method=S256`;
  const authorizePage = await app.fetch(new Request(`http://worker.test${authorizePath}`), env, {});
  assert.equal(authorizePage.status, 200);
  const authorizeBody = new URLSearchParams({
    client_id: 'f8studio',
    redirect_uri: redirectUri,
    state,
    code_challenge: codeChallenge,
    code_challenge_method: 'S256',
    csrf_token: desktopCsrfTokenFromHtml(await authorizePage.text()),
    email: 'desktop-refresh@example.com',
    password: TEST_PASSWORD,
  });
  const authorizeResponse = await app.fetch(new Request('http://worker.test/v1/auth/desktop/authorize', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      cookie: responseCookie(authorizePage.headers),
      origin: 'http://worker.test',
    },
    body: authorizeBody.toString(),
  }), env, {});
  assert.equal(authorizeResponse.status, 302);
  const code = String(new URL(String(authorizeResponse.headers.get('location') || '')).searchParams.get('code') || '');
  assert.ok(code);

  const tokenResponse = await jsonRequest(app, env, '/v1/auth/desktop/token', {
    method: 'POST',
    payload: {
      clientId: 'f8studio',
      redirectUri,
      code,
      codeVerifier,
    },
  });
  assert.equal(tokenResponse.status, 200);

  const refreshResponse = await jsonRequest(app, env, '/v1/auth/desktop/refresh', {
    method: 'POST',
    payload: {
      refreshToken: tokenResponse.json.refreshToken,
    },
  });
  assert.equal(refreshResponse.status, 200);
  assert.notEqual(refreshResponse.json.accessToken, tokenResponse.json.accessToken);
  assert.notEqual(refreshResponse.json.refreshToken, tokenResponse.json.refreshToken);

  const oldAccessMe = await app.fetch(new Request('http://worker.test/v1/me', {
    headers: {
      Authorization: `Bearer ${String(tokenResponse.json.accessToken || '')}`,
    },
  }), env, {});
  assert.equal(oldAccessMe.status, 401);

  const revokeResponse = await jsonRequest(app, env, '/v1/auth/desktop/revoke', {
    method: 'POST',
    payload: {
      refreshToken: refreshResponse.json.refreshToken,
    },
  });
  assert.equal(revokeResponse.status, 200);
  assert.equal(revokeResponse.json.revoked, true);

  const refreshAfterRevoke = await jsonRequest(app, env, '/v1/auth/desktop/refresh', {
    method: 'POST',
    payload: {
      refreshToken: refreshResponse.json.refreshToken,
    },
  });
  assert.equal(refreshAfterRevoke.status, 401);
});

test('desktop browser auth page shows a Google button when Google auth is configured', async (t) => {
  const env = createEnv({
    GOOGLE_CLIENT_ID: 'google-client-id',
    GOOGLE_CLIENT_SECRET: 'google-client-secret',
  });
  t.after(() => env.DB.close());
  const app = createApp();

  const redirectUri = 'http://127.0.0.1:42001/callback';
  const state = 'desktop-google-page-state';
  const codeVerifier = 'desktop-google-page-verifier';
  const codeChallenge = pkceChallenge(codeVerifier);
  const pageResponse = await app.fetch(new Request(
    `http://worker.test/v1/auth/desktop/authorize?client_id=f8studio&redirect_uri=${encodeURIComponent(redirectUri)}&state=${encodeURIComponent(state)}&code_challenge=${encodeURIComponent(codeChallenge)}&code_challenge_method=S256`,
  ), env, {});
  assert.equal(pageResponse.status, 200);
  const pageHtml = await pageResponse.text();
  assert.match(pageHtml, /Continue with Google/);
  assert.match(pageHtml, /name="social_provider" value="google"/);
  assert.doesNotMatch(pageHtml, /Desktop callback:/);
});

test('desktop browser auth page hides Google sign-in when registration is disabled', async (t) => {
  const env = createEnv({
    allowUserRegistration: false,
    GOOGLE_CLIENT_ID: 'google-client-id',
    GOOGLE_CLIENT_SECRET: 'google-client-secret',
  });
  t.after(() => env.DB.close());
  const app = createApp();

  const redirectUri = 'http://127.0.0.1:42003/callback';
  const state = 'desktop-google-disabled-state';
  const codeVerifier = 'desktop-google-disabled-verifier';
  const codeChallenge = pkceChallenge(codeVerifier);
  const pageResponse = await app.fetch(new Request(
    `http://worker.test/v1/auth/desktop/authorize?client_id=f8studio&redirect_uri=${encodeURIComponent(redirectUri)}&state=${encodeURIComponent(state)}&code_challenge=${encodeURIComponent(codeChallenge)}&code_challenge_method=S256`,
  ), env, {});
  assert.equal(pageResponse.status, 200);
  const pageHtml = await pageResponse.text();
  assert.doesNotMatch(pageHtml, /Continue with Google/);
  assert.doesNotMatch(pageHtml, /Desktop callback:/);
});

test('desktop browser Google sign-in start is rejected when registration is disabled', async (t) => {
  const env = createEnv({
    allowUserRegistration: false,
    GOOGLE_CLIENT_ID: 'google-client-id',
    GOOGLE_CLIENT_SECRET: 'google-client-secret',
  });
  t.after(() => env.DB.close());
  const app = createApp();

  const redirectUri = 'http://127.0.0.1:42004/callback';
  const state = 'desktop-google-disabled-start-state';
  const codeVerifier = 'desktop-google-disabled-start-verifier';
  const codeChallenge = pkceChallenge(codeVerifier);
  const authorizeResponse = await app.fetch(new Request(
    `http://worker.test/v1/auth/desktop/authorize?client_id=f8studio&redirect_uri=${encodeURIComponent(redirectUri)}&state=${encodeURIComponent(state)}&code_challenge=${encodeURIComponent(codeChallenge)}&code_challenge_method=S256&social_provider=google&social_start=1`,
  ), env, {});
  assert.equal(authorizeResponse.status, 400);
  const pageHtml = await authorizeResponse.text();
  assert.match(pageHtml, /Google sign-in is unavailable while registration is disabled\./);
});

test('desktop browser Google sign-in start redirects to the provider and preserves desktop resume state', async (t) => {
  const env = createEnv({
    GOOGLE_CLIENT_ID: 'google-client-id',
    GOOGLE_CLIENT_SECRET: 'google-client-secret',
  });
  t.after(() => env.DB.close());
  const app = createApp();

  const redirectUri = 'http://127.0.0.1:42002/callback';
  const state = 'desktop-google-start-state';
  const codeVerifier = 'desktop-google-start-verifier';
  const codeChallenge = pkceChallenge(codeVerifier);
  const authorizeResponse = await app.fetch(new Request(
    `http://worker.test/v1/auth/desktop/authorize?client_id=f8studio&redirect_uri=${encodeURIComponent(redirectUri)}&state=${encodeURIComponent(state)}&code_challenge=${encodeURIComponent(codeChallenge)}&code_challenge_method=S256&social_provider=google&social_start=1`,
  ), env, {});
  assert.equal(authorizeResponse.status, 302);
  const location = String(authorizeResponse.headers.get('location') || '');
  assert.ok(location);
  const providerUrl = new URL(location);
  assert.match(providerUrl.hostname, /google/i);
  assert.equal(providerUrl.searchParams.get('redirect_uri'), 'http://worker.test/api/auth/callback/google');
  assert.ok(String(authorizeResponse.headers.get('set-cookie') || '').trim());
});

test('openapi endpoints expose the audited worker contract', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const openapi = await jsonRequest(app, env, '/openapi.json');
  assert.equal(openapi.status, 200);
  assert.equal(openapi.json.info.title, 'Feel8 Asset Cloud API');
  assert.ok(openapi.json.paths['/v1/auth/providers']);
  assert.ok(openapi.json.paths['/v1/auth/desktop/token']);
  assert.ok(openapi.json.paths['/v1/auth/desktop/session']);
  assert.ok(openapi.json.paths['/v1/auth/desktop/refresh']);
  assert.ok(openapi.json.paths['/v1/auth/desktop/revoke']);
  assert.ok(openapi.json.paths['/v1/site-settings']);
  assert.ok(openapi.json.paths['/v1/me']);
  assert.ok(openapi.json.paths['/v1/me'].put);
  assert.equal(openapi.json.paths['/v1/search'], undefined);
  assert.ok(openapi.json.paths['/v1/components']);
  assert.ok(openapi.json.paths['/v1/components/{componentId}']);
  assert.ok(openapi.json.paths['/v1/components/{componentId}/content']);
  assert.ok(openapi.json.paths['/v1/components/{componentId}/meta']);
  assert.ok(openapi.json.paths['/v1/components/{componentId}/versions']);
  assert.ok(openapi.json.paths['/v1/components/{componentId}/versions/{versionNumber}'].patch);
  assert.ok(openapi.json.paths['/v1/components/{componentId}/subscribers']);
  assert.ok(openapi.json.paths['/v1/components/{componentId}/subscribe']);
  assert.ok(openapi.json.paths['/v1/modding-recipes']);
  assert.ok(openapi.json.paths['/v1/modding-recipes/{recipeId}']);
  assert.ok(openapi.json.paths['/v1/modding-recipes/{recipeId}/content']);
  assert.ok(openapi.json.paths['/v1/modding-recipes/{recipeId}/meta']);
  assert.ok(openapi.json.paths['/v1/modding-recipes/{recipeId}/versions']);
  assert.ok(openapi.json.paths['/v1/modding-recipes/{recipeId}/versions/{versionNumber}'].patch);
  assert.ok(openapi.json.paths['/v1/modding-recipes/{recipeId}/subscribers']);
  assert.ok(openapi.json.paths['/v1/modding-recipes/{recipeId}/subscribe']);
  assert.ok(openapi.json.paths['/v1/variants']);
  assert.ok(openapi.json.paths['/v1/variants/{variantId}']);
  assert.ok(openapi.json.paths['/v1/variants/{variantId}/content']);
  assert.ok(openapi.json.paths['/v1/variants/{variantId}/meta']);
  assert.ok(openapi.json.paths['/v1/variants/{variantId}/versions']);
  assert.ok(openapi.json.paths['/v1/variants/{variantId}/versions/{versionNumber}'].patch);
  assert.ok(openapi.json.paths['/v1/variants/{variantId}/subscribers']);
  assert.ok(openapi.json.paths['/v1/variants/{variantId}/subscribe']);
  assert.ok(openapi.json.paths['/v1/management/users']);
  assert.ok(openapi.json.paths['/v1/management/users/{userId}']);
  assert.ok(openapi.json.paths['/v1/management/site-settings']);
  assert.ok(openapi.json.paths['/v1/management/assets/purge-all']);
  assert.ok(openapi.json.paths['/v1/management/components']);
  assert.ok(openapi.json.paths['/v1/management/components/{componentId}']);
  assert.ok(openapi.json.paths['/v1/management/modding-recipes']);
  assert.ok(openapi.json.paths['/v1/management/modding-recipes/{recipeId}']);
  assert.ok(openapi.json.paths['/v1/management/variants']);
  assert.ok(openapi.json.paths['/v1/management/variants/{variantId}']);
  assert.equal(openapi.json.paths['/v1/management/users/{userId}/assets'], undefined);
  assert.equal(openapi.json.paths['/v1/management/assets'], undefined);
  assert.equal(openapi.json.paths['/v1/management/assets/{assetId}'], undefined);

  const docsRequest = new Request('http://worker.test/docs');
  const docsResponse = await app.fetch(docsRequest, env, {});
  const docsHtml = await docsResponse.text();
  assert.equal(docsResponse.status, 200);
  assert.match(docsHtml, /openapi\.json/);
});

test('current user rename rejects duplicate names', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  await createVerifiedSession(app, env, {
    name: 'Alice',
    email: 'alice@example.com',
  });
  const bob = await createVerifiedSession(app, env, {
    name: 'Bob',
    email: 'bob@example.com',
  });

  const duplicateRename = await jsonRequest(app, env, '/v1/me', {
    method: 'PUT',
    cookie: bob.cookie,
    payload: {
      name: 'Alice',
    },
  });
  assert.equal(duplicateRename.status, 409);
  assert.equal(duplicateRename.json.message, 'name already in use');
});

test('current user rename blocks reserved names for non-admin users and allows them for admins', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const alice = await createVerifiedSession(app, env, {
    name: 'Alice',
    email: 'alice@example.com',
  });
  const admin = await signInUser(app, env, {
    email: 'admin@example.com',
  });
  assert.equal(admin.status, 200);
  assert.ok(admin.cookie);

  const nonAdminReservedRename = await jsonRequest(app, env, '/v1/me', {
    method: 'PUT',
    cookie: alice.cookie,
    payload: {
      name: 'root',
    },
  });
  assert.equal(nonAdminReservedRename.status, 400);
  assert.equal(nonAdminReservedRename.json.message, 'name must be 2-64 visible characters and not reserved');

  const adminReservedRename = await jsonRequest(app, env, '/v1/me', {
    method: 'PUT',
    cookie: admin.cookie,
    payload: {
      name: 'root',
    },
  });
  assert.equal(adminReservedRename.status, 200);
  assert.equal(adminReservedRename.json.name, 'root');
  assert.equal(adminReservedRename.json.role, 'admin');
});

test('public sign-up blocks reserved names while admin user creation can still provision them', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const reservedSignUp = await jsonRequest(app, env, '/api/auth/sign-up/email', {
    method: 'POST',
    payload: {
      name: 'root',
      email: 'root@example.com',
      password: TEST_PASSWORD,
    },
  });
  assert.equal(reservedSignUp.status, 400);
  assert.equal(reservedSignUp.json.message, 'reserved names are only available to admins');

  const admin = await signInUser(app, env, {
    email: 'admin@example.com',
  });
  assert.equal(admin.status, 200);
  assert.ok(admin.cookie);

  const reservedAdminCreate = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/users`, {
    method: 'POST',
    cookie: admin.cookie,
    payload: {
      name: 'root',
      email: 'root-admin@example.com',
      password: TEST_PASSWORD,
      role: 'admin',
    },
  });
  assert.equal(reservedAdminCreate.status, 200);
  assert.equal(reservedAdminCreate.json.name, 'root');
  assert.equal(reservedAdminCreate.json.role, 'admin');
});

test('authenticated user can request an email change and verify the new email', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const alice = await createVerifiedSession(app, env, {
    name: 'Alice',
    email: 'alice@example.com',
  });

  const { result: changeEmail, logs } = await captureConsoleInfo(() => jsonRequest(app, env, '/api/auth/change-email', {
    method: 'POST',
    cookie: alice.cookie,
    origin: 'http://worker.test',
    payload: {
      newEmail: 'alice.updated@example.com',
      callbackURL: 'http://worker.test/verify-email?verified=1',
    },
  }));
  assert.equal(changeEmail.status, 200);
  const changeEmailToken = extractDebugToken(logs, 'verify email');
  assert.ok(changeEmailToken);

  const verified = await verifyUserEmail(app, env, changeEmailToken);
  assert.equal(verified.status, 200);

  const me = await jsonRequest(app, env, '/v1/me', { cookie: alice.cookie });
  assert.equal(me.status, 200);
  assert.equal(me.json.email, 'alice.updated@example.com');
  assert.equal(me.json.emailVerified, true);
});

test('hot asset list queries use composite indexes without temp sorting', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());

  const componentPlan = await env.DB.prepare(
    `EXPLAIN QUERY PLAN
     SELECT
       h.*,
       u.name AS owner_display_name,
       s.subscribed_at,
       s.last_seen_version_number,
       v.created_by_user_id,
       v.change_summary,
       v.version_number
     FROM asset_heads h
     JOIN asset_versions v
       ON v.asset_id = h.asset_id AND v.version_number = h.current_version_number
     LEFT JOIN user u ON u.id = h.owner_user_id
     LEFT JOIN asset_subscriptions s
       ON s.asset_id = h.asset_id AND s.subscriber_user_id = ?
     WHERE h.asset_type = ? AND h.visibility = 'public'
     ORDER BY LOWER(h.name), h.asset_id
     LIMIT ? OFFSET ?`,
  )
    .bind('', 'component', 101, 0)
    .all();
  const componentPlanDetails = (componentPlan.results || []).map((row) => String(row.detail || ''));
  assert.ok(componentPlanDetails.some((detail) => detail.includes('idx_asset_heads_type_visibility_name')));
  assert.equal(componentPlanDetails.some((detail) => detail.includes('USE TEMP B-TREE FOR ORDER BY')), false);

  const managementPlan = await env.DB.prepare(
    `EXPLAIN QUERY PLAN
     SELECT
       h.*,
       u.name AS owner_display_name,
       v.created_at AS version_created_at,
       v.created_by_user_id,
       v.change_summary,
       v.version_number
     FROM asset_heads h
     JOIN asset_versions v
       ON v.asset_id = h.asset_id AND v.version_number = h.current_version_number
     LEFT JOIN user u ON u.id = h.owner_user_id
     WHERE 1 = 1
     ORDER BY h.updated_at DESC, h.asset_id
     LIMIT ? OFFSET ?`,
  )
    .bind(101, 0)
    .all();
  const managementPlanDetails = (managementPlan.results || []).map((row) => String(row.detail || ''));
  assert.ok(managementPlanDetails.some((detail) => detail.includes('idx_asset_heads_updated')));
  assert.equal(managementPlanDetails.some((detail) => detail.includes('USE TEMP B-TREE FOR ORDER BY')), false);
});

test('variant asset lifecycle works with Better Auth cookie sessions', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const alice = await createVerifiedSession(app, env, {
    name: 'Alice',
    email: 'alice@example.com',
  });
  const bob = await createVerifiedSession(app, env, {
    name: 'Bob',
    email: 'bob@example.com',
  });

  const created = await jsonRequest(app, env, '/v1/variants', {
    method: 'POST',
    cookie: alice.cookie,
    payload: variantPayload({ variantId: 'alice-variant', name: 'Alice Private', visibility: 'private' }),
  });
  assert.equal(created.status, 200);
  assert.equal(created.json.variantId, 'alice-variant');
  assert.equal(created.json.assetType, 'variant');
  assert.equal(created.json.versionNumber, 1);
  assert.equal(created.json.editable, true);
  const assetHeadColumns = await env.DB.prepare("PRAGMA table_info(asset_heads)").all();
  assert.equal(
    Array.isArray(assetHeadColumns.results) && assetHeadColumns.results.some((column) => String(column.name) === 'content'),
    false,
  );
  assert.equal(
    Array.isArray(assetHeadColumns.results) && assetHeadColumns.results.some((column) => String(column.name) === 'current_version_number'),
    true,
  );
  assert.equal(
    Array.isArray(assetHeadColumns.results) && assetHeadColumns.results.some((column) => String(column.name) === 'latest_revision'),
    false,
  );
  assert.equal(
    Array.isArray(assetHeadColumns.results) && assetHeadColumns.results.some((column) => String(column.name) === 'schema_version'),
    false,
  );
  const storedVariantVersion = await env.DB.prepare(
    `SELECT content
     FROM asset_versions
     WHERE asset_id = ? AND version_number = 1`,
  )
    .bind('alice-variant')
    .first();
  const storedVariantSpec = JSON.parse(gunzipSync(Buffer.from(storedVariantVersion.content)).toString('utf-8'));
  assert.equal(storedVariantSpec.label, 'Alice Private');
  assert.equal(storedVariantSpec.record, undefined);
  assert.equal(storedVariantSpec.name, undefined);
  const variantDetails = await env.DB.prepare(
    'SELECT variant_kind, base_node_type, service_class, operator_class FROM variant_details WHERE asset_id = ?',
  )
    .bind('alice-variant')
    .first();
  assert.equal(String(variantDetails?.variant_kind || ''), 'operator');
  assert.equal(String(variantDetails?.base_node_type || ''), 'svc.base.op');
  assert.equal(String(variantDetails?.service_class || ''), 'svc.test');
  assert.equal(String(variantDetails?.operator_class || ''), 'op.test');

  const publicListBefore = await jsonRequest(app, env, '/v1/variants?owner=public');
  assert.equal(publicListBefore.status, 200);
  assert.equal(publicListBefore.json.entries.length, 0);

  const privateByBob = await jsonRequest(app, env, '/v1/variants/alice-variant', { cookie: bob.cookie });
  assert.equal(privateByBob.status, 403);

  const updated = await jsonRequest(app, env, '/v1/variants/alice-variant', {
    method: 'PUT',
    cookie: alice.cookie,
    payload: variantPayload({
      variantId: 'alice-variant',
      name: 'Alice Public',
      visibility: 'public',
      versionNumber: created.json.versionNumber,
    }),
  });
  assert.equal(updated.status, 200);
  assert.equal(updated.json.versionNumber, 2);
  assert.equal(updated.json.createdAt, created.json.createdAt);
  assert.equal(updated.json.visibility, 'public');

  const publicListAfter = await jsonRequest(app, env, '/v1/variants?owner=public&q=alice');
  assert.equal(publicListAfter.status, 200);
  assert.equal(publicListAfter.json.entries.length, 1);

  const subscribed = await jsonRequest(app, env, '/v1/variants/alice-variant/subscribe', {
    method: 'POST',
    cookie: bob.cookie,
  });
  assert.equal(subscribed.status, 200);
  assert.equal(subscribed.json.subscribed, true);
  assert.equal(subscribed.json.editable, false);

  const ownerSubscribe = await jsonRequest(app, env, '/v1/variants/alice-variant/subscribe', {
    method: 'POST',
    cookie: alice.cookie,
  });
  assert.equal(ownerSubscribe.status, 200);
  assert.equal(ownerSubscribe.json.subscribed, true);
  assert.equal(ownerSubscribe.json.editable, true);

  const subscribedList = await jsonRequest(app, env, '/v1/variants?owner=subscribed', {
    cookie: bob.cookie,
  });
  assert.equal(subscribedList.status, 200);
  assert.equal(subscribedList.json.entries.length, 1);
  assert.equal(subscribedList.json.entries[0].variantId, 'alice-variant');

  const ownerSubscribedList = await jsonRequest(app, env, '/v1/variants?owner=subscribed', {
    cookie: alice.cookie,
  });
  assert.equal(ownerSubscribedList.status, 200);
  assert.equal(ownerSubscribedList.json.entries.length, 1);
  assert.equal(ownerSubscribedList.json.entries[0].variantId, 'alice-variant');

  const subscribers = await jsonRequest(app, env, '/v1/variants/alice-variant/subscribers', {
    cookie: alice.cookie,
  });
  assert.equal(subscribers.status, 200);
  assert.equal(subscribers.json.entries.length, 2);
  assert.deepEqual(
    subscribers.json.entries.map((entry) => entry.userId),
    [alice.userId, bob.userId],
  );
  assert.deepEqual(
    subscribers.json.entries.map((entry) => entry.name),
    ['Alice', 'Bob'],
  );
  assert.ok(subscribers.json.entries.every((entry) => Object.hasOwn(entry, 'email') === false));

  const forbiddenSubscribers = await jsonRequest(app, env, '/v1/variants/alice-variant/subscribers', {
    cookie: bob.cookie,
  });
  assert.equal(forbiddenSubscribers.status, 403);

  const forbiddenEdit = await jsonRequest(app, env, '/v1/variants/alice-variant', {
    method: 'PUT',
    cookie: bob.cookie,
    payload: variantPayload({
      variantId: 'alice-variant',
      name: 'Bob Edit',
      visibility: 'public',
      versionNumber: updated.json.versionNumber,
    }),
  });
  assert.equal(forbiddenEdit.status, 403);

  const history = await jsonRequest(app, env, '/v1/variants/alice-variant/versions', { cookie: alice.cookie });
  assert.equal(history.status, 200);
  assert.equal(history.json.versions.length, 2);
  assert.equal(history.json.versions[0].variantId, 'alice-variant');
  assert.equal(history.json.versions[0].versionNumber, 2);

  const oldVersion = await jsonRequest(app, env, '/v1/variants/alice-variant/versions/1', { cookie: alice.cookie });
  assert.equal(oldVersion.status, 200);
  assert.equal(oldVersion.json.variantId, 'alice-variant');
  assert.equal(oldVersion.json.hasContent, true);
  assert.equal(oldVersion.json.createdAt, created.json.createdAt);

  const versionNoteUpdated = await jsonRequest(app, env, '/v1/variants/alice-variant/versions/1', {
    method: 'PATCH',
    cookie: alice.cookie,
    payload: {
      changeSummary: 'Initial release note',
    },
  });
  assert.equal(versionNoteUpdated.status, 200);
  assert.equal(versionNoteUpdated.json.changeSummary, 'Initial release note');

  const forbiddenVersionNote = await jsonRequest(app, env, '/v1/variants/alice-variant/versions/1', {
    method: 'PATCH',
    cookie: bob.cookie,
    payload: {
      changeSummary: 'Bob note',
    },
  });
  assert.equal(forbiddenVersionNote.status, 403);

  const oldVersionContent = await jsonRequest(app, env, '/v1/variants/alice-variant/versions/1/content', { cookie: alice.cookie });
  assert.equal(oldVersionContent.status, 200);
  assert.equal(oldVersionContent.json.record.name, 'Alice Public');
  assert.equal(oldVersionContent.json.record.spec.label, 'Alice Private');
  assert.equal(oldVersionContent.json.record.createdAt, created.json.createdAt);
  assert.equal(oldVersionContent.json.record.updatedAt, history.json.versions[1].createdAt);

  const conflict = await jsonRequest(app, env, '/v1/variants/alice-variant', {
    method: 'PUT',
    cookie: alice.cookie,
    payload: variantPayload({
      variantId: 'alice-variant',
      name: 'Stale Update',
      visibility: 'public',
      versionNumber: 1,
    }),
  });
  assert.equal(conflict.status, 409);
  assert.equal(conflict.json.versionNumber, 2);

  const variantHeadBeforeMetaPatch = await env.DB.prepare(
    `SELECT current_version_number, updated_at
     FROM asset_heads
     WHERE asset_id = ?`,
  )
    .bind('alice-variant')
    .first();

  const metadataPatched = await jsonRequest(app, env, '/v1/variants/alice-variant/meta', {
    method: 'PATCH',
    cookie: alice.cookie,
    payload: {
      name: 'Alice Public Metadata',
      description: 'Metadata only update',
      tags: ['meta', 'variant'],
    },
  });
  assert.equal(metadataPatched.status, 200);
  assert.equal(metadataPatched.json.name, 'Alice Public Metadata');
  assert.equal(metadataPatched.json.description, 'Metadata only update');
  assert.deepEqual(metadataPatched.json.tags, ['meta', 'variant']);
  assert.equal(metadataPatched.json.versionNumber, 2);

  const variantHeadAfterMetaPatch = await env.DB.prepare(
    `SELECT current_version_number, updated_at
     FROM asset_heads
     WHERE asset_id = ?`,
  )
    .bind('alice-variant')
    .first();
  assert.equal(Number(variantHeadAfterMetaPatch?.current_version_number ?? 0), 2);
  assert.notEqual(
    String(variantHeadAfterMetaPatch?.updated_at || ''),
    String(variantHeadBeforeMetaPatch?.updated_at || ''),
  );

  const variantVersionsAfterMetaPatch = await jsonRequest(app, env, '/v1/variants/alice-variant/versions', { cookie: alice.cookie });
  assert.equal(variantVersionsAfterMetaPatch.status, 200);
  assert.equal(variantVersionsAfterMetaPatch.json.versions.length, 2);
  assert.equal(variantVersionsAfterMetaPatch.json.versions[1].changeSummary, 'Initial release note');

  const forbiddenVariantMetadataPatch = await jsonRequest(app, env, '/v1/variants/alice-variant/meta', {
    method: 'PATCH',
    cookie: bob.cookie,
    payload: {
      name: 'Bob Variant Edit',
      description: 'forbidden',
      tags: ['forbidden'],
    },
  });
  assert.equal(forbiddenVariantMetadataPatch.status, 403);

  const forked = await jsonRequest(app, env, '/v1/variants/alice-variant/fork', {
    method: 'POST',
    cookie: bob.cookie,
    payload: { variantId: 'bob-fork', name: 'Bob Fork' },
  });
  assert.equal(forked.status, 200);
  assert.equal(forked.json.variantId, 'bob-fork');
  assert.equal(forked.json.visibility, 'private');
  assert.equal(forked.json.editable, true);

  const bobMine = await jsonRequest(app, env, '/v1/variants?owner=me', { cookie: bob.cookie });
  assert.equal(bobMine.status, 200);
  assert.equal(bobMine.json.entries.length, 1);
  assert.equal(bobMine.json.entries[0].variantId, 'bob-fork');

  const bobPublicOnly = await jsonRequest(app, env, '/v1/variants?owner=public', { cookie: bob.cookie });
  assert.equal(bobPublicOnly.status, 200);
  assert.equal(bobPublicOnly.json.entries.length, 1);
  assert.equal(bobPublicOnly.json.entries[0].variantId, 'alice-variant');

  const unsubscribed = await jsonRequest(app, env, '/v1/variants/alice-variant/subscribe', {
    method: 'DELETE',
    cookie: bob.cookie,
  });
  assert.equal(unsubscribed.status, 200);

  const removed = await jsonRequest(app, env, '/v1/variants/alice-variant', {
    method: 'DELETE',
    cookie: alice.cookie,
  });
  assert.equal(removed.status, 200);

  const publicAfterDelete = await jsonRequest(app, env, '/v1/variants?owner=public');
  assert.equal(publicAfterDelete.status, 200);
  assert.equal(publicAfterDelete.json.entries.length, 0);
});

test('component asset lifecycle validates session envelope and visibility rules', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const alice = await createVerifiedSession(app, env, {
    name: 'Alice',
    email: 'alice@example.com',
  });
  const bob = await createVerifiedSession(app, env, {
    name: 'Bob',
    email: 'bob@example.com',
  });

  const invalid = await jsonRequest(app, env, '/v1/components', {
    method: 'POST',
    cookie: alice.cookie,
    payload: {
      record: {
        componentId: 'bad-component',
        name: 'Bad',
        description: '',
        tags: [],
        schemaVersion: 'f8studio-session/1',
        content: { schemaVersion: 'bad', layout: {} },
      },
    },
  });
  assert.equal(invalid.status, 400);
  assert.equal(invalid.json.message, 'record.schemaVersion is not allowed; use record.content.schemaVersion');

  const created = await jsonRequest(app, env, '/v1/components', {
    method: 'POST',
    cookie: alice.cookie,
    payload: componentPayload({ componentId: 'component-a', name: 'Published Session', visibility: 'public' }),
  });
  assert.equal(created.status, 200);
  assert.equal(created.json.componentId, 'component-a');
  assert.equal(Object.hasOwn(created.json, 'schemaVersion'), false);
  const componentVariantDetails = await env.DB.prepare(
    'SELECT asset_id FROM variant_details WHERE asset_id = ?',
  )
    .bind('component-a')
    .first();
  assert.equal(componentVariantDetails, null);

  const publicList = await jsonRequest(app, env, '/v1/components?owner=public');
  assert.equal(publicList.status, 200);
  assert.equal(publicList.json.entries.length, 1);
  assert.equal(publicList.json.entries[0].componentId, 'component-a');
  assert.equal(publicList.json.entries[0].name, 'Published Session');
  assert.equal(publicList.json.entries[0].hasContent, true);

  const componentListSearch = await jsonRequest(app, env, '/v1/components?owner=public');
  assert.equal(componentListSearch.status, 200);
  assert.equal(componentListSearch.json.entries[0].componentId, 'component-a');
  assert.equal(Object.hasOwn(componentListSearch.json.entries[0], 'schemaVersion'), false);
  assert.equal(Object.hasOwn(componentListSearch.json.entries[0], 'variantKind'), false);

  const subscribed = await jsonRequest(app, env, '/v1/components/component-a/subscribe', {
    method: 'POST',
    cookie: bob.cookie,
  });
  assert.equal(subscribed.status, 200);
  assert.equal(subscribed.json.subscribed, true);
  assert.equal(subscribed.json.editable, false);

  const ownerSubscribe = await jsonRequest(app, env, '/v1/components/component-a/subscribe', {
    method: 'POST',
    cookie: alice.cookie,
  });
  assert.equal(ownerSubscribe.status, 200);
  assert.equal(ownerSubscribe.json.subscribed, true);
  assert.equal(ownerSubscribe.json.editable, true);

  const ownerSubscribedList = await jsonRequest(app, env, '/v1/components?owner=subscribed', {
    cookie: alice.cookie,
  });
  assert.equal(ownerSubscribedList.status, 200);
  assert.equal(ownerSubscribedList.json.entries.length, 1);
  assert.equal(ownerSubscribedList.json.entries[0].componentId, 'component-a');

  const subscribers = await jsonRequest(app, env, '/v1/components/component-a/subscribers', {
    cookie: alice.cookie,
  });
  assert.equal(subscribers.status, 200);
  assert.equal(subscribers.json.entries.length, 2);
  assert.deepEqual(
    subscribers.json.entries.map((entry) => entry.userId),
    [alice.userId, bob.userId],
  );
  assert.deepEqual(
    subscribers.json.entries.map((entry) => entry.name),
    ['Alice', 'Bob'],
  );
  assert.ok(subscribers.json.entries.every((entry) => Object.hasOwn(entry, 'email') === false));

  const history1 = await jsonRequest(app, env, '/v1/components/component-a/versions', { cookie: alice.cookie });
  assert.equal(history1.status, 200);
  assert.equal(history1.json.versions.length, 1);
  assert.equal(history1.json.versions[0].componentId, 'component-a');

  const versionNoteUpdated = await jsonRequest(app, env, '/v1/components/component-a/versions/1', {
    method: 'PATCH',
    cookie: alice.cookie,
    payload: {
      changeSummary: 'Initial component note',
    },
  });
  assert.equal(versionNoteUpdated.status, 200);
  assert.equal(versionNoteUpdated.json.changeSummary, 'Initial component note');

  const updated = await jsonRequest(app, env, '/v1/components/component-a', {
    method: 'PUT',
    cookie: alice.cookie,
    payload: componentPayload({
      componentId: 'component-a',
      name: 'Published Session v2',
      visibility: 'public',
      versionNumber: created.json.versionNumber,
    }),
  });
  assert.equal(updated.status, 200);
  assert.equal(updated.json.versionNumber, 2);

  const componentHeadBeforeMetaPatch = await env.DB.prepare(
    `SELECT current_version_number, updated_at
     FROM asset_heads
     WHERE asset_id = ?`,
  )
    .bind('component-a')
    .first();

  const componentMetadataPatched = await jsonRequest(app, env, '/v1/components/component-a/meta', {
    method: 'PATCH',
    cookie: alice.cookie,
    payload: {
      name: 'Published Session Metadata',
      description: 'Metadata only component update',
      tags: ['meta', 'component'],
    },
  });
  assert.equal(componentMetadataPatched.status, 200);
  assert.equal(componentMetadataPatched.json.name, 'Published Session Metadata');
  assert.equal(componentMetadataPatched.json.description, 'Metadata only component update');
  assert.deepEqual(componentMetadataPatched.json.tags, ['meta', 'component']);
  assert.equal(componentMetadataPatched.json.versionNumber, 2);

  const componentHeadAfterMetaPatch = await env.DB.prepare(
    `SELECT current_version_number, updated_at
     FROM asset_heads
     WHERE asset_id = ?`,
  )
    .bind('component-a')
    .first();
  assert.equal(Number(componentHeadAfterMetaPatch?.current_version_number ?? 0), 2);
  assert.notEqual(
    String(componentHeadAfterMetaPatch?.updated_at || ''),
    String(componentHeadBeforeMetaPatch?.updated_at || ''),
  );

  const componentVersionsAfterMetaPatch = await jsonRequest(app, env, '/v1/components/component-a/versions', { cookie: alice.cookie });
  assert.equal(componentVersionsAfterMetaPatch.status, 200);
  assert.equal(componentVersionsAfterMetaPatch.json.versions.length, 2);
  assert.equal(componentVersionsAfterMetaPatch.json.versions[1].changeSummary, 'Initial component note');

  const forbiddenComponentMetadataPatch = await jsonRequest(app, env, '/v1/components/component-a/meta', {
    method: 'PATCH',
    cookie: bob.cookie,
    payload: {
      name: 'Bob Component Edit',
      description: 'forbidden',
      tags: ['forbidden'],
    },
  });
  assert.equal(forbiddenComponentMetadataPatch.status, 403);

  const oldVersion = await jsonRequest(app, env, '/v1/components/component-a/versions/1', { cookie: bob.cookie });
  assert.equal(oldVersion.status, 200);
  assert.equal(oldVersion.json.componentId, 'component-a');
  assert.equal(oldVersion.json.hasContent, true);
  assert.equal(oldVersion.json.createdAt, created.json.createdAt);
  const oldVersionContent = await jsonRequest(app, env, '/v1/components/component-a/versions/1/content', { cookie: bob.cookie });
  assert.equal(oldVersionContent.status, 200);
  assert.equal(oldVersionContent.json.record.name, 'Published Session Metadata');
  assert.equal(oldVersionContent.json.record.createdAt, created.json.createdAt);
  assert.equal(oldVersionContent.json.record.updatedAt, history1.json.versions[0].createdAt);

  const forbidden = await jsonRequest(app, env, '/v1/components/component-a', {
    method: 'PUT',
    cookie: bob.cookie,
    payload: componentPayload({
      componentId: 'component-a',
      name: 'Bob Edit',
      visibility: 'public',
      versionNumber: updated.json.versionNumber,
    }),
  });
  assert.equal(forbidden.status, 403);

  const forked = await jsonRequest(app, env, '/v1/components/component-a/fork', {
    method: 'POST',
    cookie: bob.cookie,
    payload: { componentId: 'component-b', name: 'Bob Session Copy' },
  });
  assert.equal(forked.status, 200);
  assert.equal(forked.json.componentId, 'component-b');
  assert.equal(forked.json.visibility, 'private');
});

test('modding recipe asset lifecycle validates shareable recipe payloads', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const alice = await createVerifiedSession(app, env, {
    name: 'Alice',
    email: 'alice@example.com',
  });
  const bob = await createVerifiedSession(app, env, {
    name: 'Bob',
    email: 'bob@example.com',
  });

  const invalidSchema = await jsonRequest(app, env, '/v1/modding-recipes', {
    method: 'POST',
    cookie: alice.cookie,
    payload: moddingRecipePayload({
      recipeId: 'bad-recipe-schema',
      contentOverrides: {
        schemaVersion: 'bad',
      },
    }),
  });
  assert.equal(invalidSchema.status, 400);
  assert.equal(invalidSchema.json.message, 'modding recipe schemaVersion must be f8moddingrecipe/1');

  const localPathField = await jsonRequest(app, env, '/v1/modding-recipes', {
    method: 'POST',
    cookie: alice.cookie,
    payload: moddingRecipePayload({
      recipeId: 'bad-recipe-path-field',
      contentOverrides: {
        gameProfile: {
          targetPath: 'H:\\Feel8\\Games\\ExampleGame',
        },
      },
    }),
  });
  assert.equal(localPathField.status, 400);
  assert.equal(localPathField.json.message, 'modding recipe published content must not contain local path field $.gameProfile.targetPath');

  const absolutePathValue = await jsonRequest(app, env, '/v1/modding-recipes', {
    method: 'POST',
    cookie: alice.cookie,
    payload: moddingRecipePayload({
      recipeId: 'bad-recipe-path-value',
      contentOverrides: {
        notes: 'C:\\Games\\ExampleGame',
      },
    }),
  });
  assert.equal(absolutePathValue.status, 400);
  assert.equal(absolutePathValue.json.message, 'modding recipe published content must not contain absolute local paths at $.notes');

  const created = await jsonRequest(app, env, '/v1/modding-recipes', {
    method: 'POST',
    cookie: alice.cookie,
    payload: moddingRecipePayload({ recipeId: 'unity-recipe', name: 'Unity Skeleton Stream', visibility: 'public' }),
  });
  assert.equal(created.status, 200);
  assert.equal(created.json.recipeId, 'unity-recipe');
  assert.equal(created.json.assetType, 'modding_recipe');
  assert.equal(created.json.hasContent, true);
  assert.equal(created.json.versionNumber, 1);

  const storedVersion = await env.DB.prepare(
    `SELECT content
     FROM asset_versions
     WHERE asset_id = ? AND version_number = 1`,
  )
    .bind('unity-recipe')
    .first();
  const storedContent = JSON.parse(gunzipSync(Buffer.from(storedVersion.content)).toString('utf-8'));
  assert.equal(storedContent.schemaVersion, 'f8moddingrecipe/1');
  assert.equal(storedContent.engine, 'unity');
  assert.equal(storedContent.record, undefined);
  assert.equal(storedContent.recipeId, undefined);
  assert.equal(storedContent.gameProfile.targetPath, undefined);

  const publicList = await jsonRequest(app, env, '/v1/modding-recipes?owner=public&q=skeleton');
  assert.equal(publicList.status, 200);
  assert.equal(publicList.json.entries.length, 1);
  assert.equal(publicList.json.entries[0].recipeId, 'unity-recipe');
  assert.equal(publicList.json.entries[0].assetType, 'modding_recipe');

  const content = await jsonRequest(app, env, '/v1/modding-recipes/unity-recipe/content');
  assert.equal(content.status, 200);
  assert.equal(content.json.recipeId, 'unity-recipe');
  assert.equal(content.json.assetType, 'modding_recipe');
  assert.equal(content.json.record.recipeId, 'unity-recipe');
  assert.equal(content.json.record.content.verification.udpPort, 39540);

  const resolved = await jsonRequest(app, env, '/v1/assets/unity-recipe');
  assert.equal(resolved.status, 200);
  assert.equal(resolved.json.assetType, 'modding_recipe');
  assert.equal(resolved.json.asset.recipeId, 'unity-recipe');

  const subscribed = await jsonRequest(app, env, '/v1/modding-recipes/unity-recipe/subscribe', {
    method: 'POST',
    cookie: bob.cookie,
  });
  assert.equal(subscribed.status, 200);
  assert.equal(subscribed.json.subscribed, true);
  assert.equal(subscribed.json.editable, false);

  const updated = await jsonRequest(app, env, '/v1/modding-recipes/unity-recipe', {
    method: 'PUT',
    cookie: alice.cookie,
    payload: moddingRecipePayload({
      recipeId: 'unity-recipe',
      name: 'Unity Skeleton Stream v2',
      visibility: 'public',
      versionNumber: created.json.versionNumber,
      contentOverrides: {
        notes: 'Updated notes',
      },
    }),
  });
  assert.equal(updated.status, 200);
  assert.equal(updated.json.versionNumber, 2);

  const history = await jsonRequest(app, env, '/v1/modding-recipes/unity-recipe/versions', { cookie: alice.cookie });
  assert.equal(history.status, 200);
  assert.equal(history.json.versions.length, 2);
  assert.equal(history.json.versions[0].recipeId, 'unity-recipe');
  assert.equal(history.json.versions[0].assetType, 'modding_recipe');

  const versionNoteUpdated = await jsonRequest(app, env, '/v1/modding-recipes/unity-recipe/versions/1', {
    method: 'PATCH',
    cookie: alice.cookie,
    payload: {
      changeSummary: 'Initial modding recipe note',
    },
  });
  assert.equal(versionNoteUpdated.status, 200);
  assert.equal(versionNoteUpdated.json.recipeId, 'unity-recipe');
  assert.equal(versionNoteUpdated.json.changeSummary, 'Initial modding recipe note');

  const conflict = await jsonRequest(app, env, '/v1/modding-recipes/unity-recipe', {
    method: 'PUT',
    cookie: alice.cookie,
    payload: moddingRecipePayload({
      recipeId: 'unity-recipe',
      name: 'Stale Recipe Update',
      visibility: 'public',
      versionNumber: 1,
    }),
  });
  assert.equal(conflict.status, 409);
  assert.equal(conflict.json.recipeId, 'unity-recipe');
  assert.equal(conflict.json.versionNumber, 2);

  const metadataPatched = await jsonRequest(app, env, '/v1/modding-recipes/unity-recipe/meta', {
    method: 'PATCH',
    cookie: alice.cookie,
    payload: {
      name: 'Unity Recipe Metadata',
      description: 'Metadata only recipe update',
      tags: ['meta', 'recipe'],
    },
  });
  assert.equal(metadataPatched.status, 200);
  assert.equal(metadataPatched.json.name, 'Unity Recipe Metadata');
  assert.equal(metadataPatched.json.versionNumber, 2);

  const forked = await jsonRequest(app, env, '/v1/modding-recipes/unity-recipe/fork', {
    method: 'POST',
    cookie: bob.cookie,
    payload: { recipeId: 'bob-recipe-fork', name: 'Bob Recipe Fork' },
  });
  assert.equal(forked.status, 200);
  assert.equal(forked.json.recipeId, 'bob-recipe-fork');
  assert.equal(forked.json.assetType, 'modding_recipe');
  assert.equal(forked.json.visibility, 'private');

  const admin = await signInUser(app, env, {
    email: 'admin@example.com',
  });
  assert.equal(admin.status, 200);
  assert.ok(admin.cookie);

  const managedList = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/modding-recipes`, {
    cookie: admin.cookie,
  });
  assert.equal(managedList.status, 200);
  assert.equal(managedList.json.entries.some((entry) => entry.assetId === 'unity-recipe' && entry.assetType === 'modding_recipe'), true);

  const managedDetail = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/modding-recipes/unity-recipe`, {
    cookie: admin.cookie,
  });
  assert.equal(managedDetail.status, 200);
  assert.equal(managedDetail.json.assetId, 'unity-recipe');
  assert.equal(managedDetail.json.record.content.schemaVersion, 'f8moddingrecipe/1');
});

test('component list and search do not depend on variant details table', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const alice = await createVerifiedSession(app, env, {
    name: 'Alice',
    email: 'alice@example.com',
  });

  const created = await jsonRequest(app, env, '/v1/components', {
    method: 'POST',
    cookie: alice.cookie,
    payload: componentPayload({ componentId: 'component-no-vd', name: 'Standalone Component', visibility: 'public' }),
  });
  assert.equal(created.status, 200);

  const admin = await signInUser(app, env, {
    email: 'admin@example.com',
  });
  assert.equal(admin.status, 200);

  await env.DB.prepare('DROP TABLE variant_details').run();

  const listed = await jsonRequest(app, env, '/v1/components?visibility=public&owner=public');
  assert.equal(listed.status, 200);
  assert.equal(listed.json.entries.length, 1);
  assert.equal(listed.json.entries[0].componentId, 'component-no-vd');

  const detail = await jsonRequest(app, env, '/v1/components/component-no-vd');
  assert.equal(detail.status, 200);
  assert.equal(detail.json.componentId, 'component-no-vd');

  const content = await jsonRequest(app, env, '/v1/components/component-no-vd/content');
  assert.equal(content.status, 200);
  assert.equal(content.json.componentId, 'component-no-vd');
  assert.equal(content.json.record.name, 'Standalone Component');

  const searched = await jsonRequest(app, env, '/v1/components?visibility=public&owner=public&q=standalone');
  assert.equal(searched.status, 200);
  assert.equal(searched.json.entries.length, 1);
  assert.equal(searched.json.entries[0].componentId, 'component-no-vd');

  const managedList = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/components`, {
    cookie: admin.cookie,
  });
  assert.equal(managedList.status, 200);
  assert.equal(managedList.json.entries.length, 1);
  assert.equal(managedList.json.entries[0].assetId, 'component-no-vd');

  const managedDetail = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/components/component-no-vd`, {
    cookie: admin.cookie,
  });
  assert.equal(managedDetail.status, 200);
  assert.equal(managedDetail.json.assetId, 'component-no-vd');
});

test('component content endpoint reads canonical stored session payload and rejects legacy wrapped blobs', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const alice = await createVerifiedSession(app, env, {
    name: 'Alice',
    email: 'alice@example.com',
  });

  const created = await jsonRequest(app, env, '/v1/components', {
    method: 'POST',
    cookie: alice.cookie,
    payload: componentPayload({ componentId: 'component-canonical', name: 'Canonical Component', visibility: 'public' }),
  });
  assert.equal(created.status, 200);

  const storedVersion = await env.DB.prepare(
    `SELECT content
     FROM asset_versions
     WHERE asset_id = ? AND version_number = 1`,
  )
    .bind('component-canonical')
    .first();
  const storedContent = JSON.parse(gunzipSync(Buffer.from(storedVersion.content)).toString('utf-8'));
  assert.equal(storedContent.schemaVersion, 'f8studio-session/1');
  assert.ok(storedContent.layout);
  assert.equal(storedContent.record, undefined);
  assert.equal(storedContent.name, undefined);

  const canonicalSessionPayload = JSON.stringify({
    schemaVersion: 'f8studio-session/1',
    layout: {
      nodes: {
        canonicalNode: {
          id: 'canonicalNode',
          name: 'Canonical Node',
          pos: [0, 0],
        },
      },
      connections: [],
    },
  });
  await env.DB.prepare(
    `UPDATE asset_versions
     SET content = ?
     WHERE asset_id = ? AND version_number = 1`,
  )
    .bind(gzipSync(Buffer.from(canonicalSessionPayload)), 'component-canonical')
    .run();

  const contentResponse = await jsonRequest(app, env, '/v1/components/component-canonical/content');
  assert.equal(contentResponse.status, 200);
  assert.equal(contentResponse.json.record.componentId, 'component-canonical');
  assert.equal(contentResponse.json.record.name, 'Canonical Component');
  assert.equal(contentResponse.json.record.description, 'published session');
  assert.equal(contentResponse.json.record.content.layout.nodes.canonicalNode.name, 'Canonical Node');

  const directRecordBlob = JSON.stringify({
    componentId: 'component-canonical',
    name: 'Direct Record Blob',
    description: 'should be rejected',
    tags: ['invalid'],
    schemaVersion: 'f8studio-session/1',
    content: {
      schemaVersion: 'f8studio-session/1',
      layout: {
        nodes: {},
        connections: [],
      },
    },
    createdAt: '2026-04-01T00:00:00.000Z',
    updatedAt: '2026-04-02T00:00:00.000Z',
  });
  await env.DB.prepare(
    `UPDATE asset_versions
     SET content = ?
     WHERE asset_id = ? AND version_number = 1`,
  )
    .bind(gzipSync(Buffer.from(directRecordBlob)), 'component-canonical')
    .run();

  const directRecordContent = await jsonRequest(app, env, '/v1/components/component-canonical/content');
  assert.equal(directRecordContent.status, 400);
  assert.equal(directRecordContent.json.message, 'stored component content must be the canonical session payload { schemaVersion, layout }');

  const legacyEnvelopeBlob = JSON.stringify({
    componentId: 'component-canonical',
    assetType: 'component',
    versionNumber: 1,
    record: {
      componentId: 'component-canonical',
      name: 'Envelope Record Blob',
      description: 'legacy envelope',
      tags: ['legacy'],
      schemaVersion: 'f8studio-session/1',
      content: {
        schemaVersion: 'f8studio-session/1',
        layout: {
          nodes: {
            fromEnvelope: {
              id: 'fromEnvelope',
              name: 'Envelope Node',
              pos: [10, 20],
            },
          },
          connections: [],
        },
      },
      createdAt: '2026-04-01T00:00:00.000Z',
      updatedAt: '2026-04-02T00:00:00.000Z',
    },
  });
  await env.DB.prepare(
    `UPDATE asset_versions
     SET content = ?
     WHERE asset_id = ? AND version_number = 1`,
  )
    .bind(gzipSync(Buffer.from(legacyEnvelopeBlob)), 'component-canonical')
    .run();

  const envelopeContent = await jsonRequest(app, env, '/v1/components/component-canonical/content');
  assert.equal(envelopeContent.status, 400);
  assert.equal(envelopeContent.json.message, 'stored component content must be the canonical session payload { schemaVersion, layout }');
});

test('component content endpoint decodes canonical gzip blobs from buffer-like D1 rows', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const alice = await createVerifiedSession(app, env, {
    name: 'Alice',
    email: 'alice@example.com',
  });

  const created = await jsonRequest(app, env, '/v1/components', {
    method: 'POST',
    cookie: alice.cookie,
    payload: componentPayload({ componentId: 'component-buffer-shape', name: 'Buffer Shape', visibility: 'public' }),
  });
  assert.equal(created.status, 200);

  wrapBlobRowsAsDataArrays(env.DB);

  const contentResponse = await jsonRequest(app, env, '/v1/components/component-buffer-shape/content');
  assert.equal(contentResponse.status, 200);
  assert.equal(contentResponse.json.record.componentId, 'component-buffer-shape');
  assert.equal(contentResponse.json.record.content.schemaVersion, 'f8studio-session/1');
  assert.ok(contentResponse.json.record.content.layout);
});

test('variant content endpoint reads canonical raw spec and rejects wrapped blobs', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const alice = await createVerifiedSession(app, env, {
    name: 'Alice',
    email: 'alice@example.com',
  });

  const created = await jsonRequest(app, env, '/v1/variants', {
    method: 'POST',
    cookie: alice.cookie,
    payload: variantPayload({ variantId: 'variant-canonical', name: 'Canonical Variant', visibility: 'public' }),
  });
  assert.equal(created.status, 200);

  const rawSpecContent = await jsonRequest(app, env, '/v1/variants/variant-canonical/content');
  assert.equal(rawSpecContent.status, 200);
  assert.equal(rawSpecContent.json.record.variantId, 'variant-canonical');
  assert.equal(rawSpecContent.json.record.spec.label, 'Canonical Variant');

  const fullRecordBlob = JSON.stringify({
    variantId: 'variant-canonical',
    kind: 'operator',
    baseNodeType: 'svc.base.op',
    serviceClass: 'svc.test',
    operatorClass: 'op.test',
    name: 'Legacy Variant Record',
    description: 'legacy record blob',
    tags: ['legacy'],
    spec: {
      label: 'Legacy Variant Spec',
      fields: ['a', 'b'],
    },
    createdAt: '2026-04-01T00:00:00.000Z',
    updatedAt: '2026-04-02T00:00:00.000Z',
  });
  await env.DB.prepare(
    `UPDATE asset_versions
     SET content = ?
     WHERE asset_id = ? AND version_number = 1`,
  )
    .bind(gzipSync(Buffer.from(fullRecordBlob)), 'variant-canonical')
    .run();

  const fullRecordContent = await jsonRequest(app, env, '/v1/variants/variant-canonical/content');
  assert.equal(fullRecordContent.status, 400);
  assert.equal(fullRecordContent.json.message, 'stored variant content must be the raw spec JSON object without record or envelope metadata');

  const envelopeBlob = JSON.stringify({
    variantId: 'variant-canonical',
    assetType: 'variant',
    versionNumber: 1,
    record: {
      variantId: 'variant-canonical',
      kind: 'operator',
      baseNodeType: 'svc.base.op',
      serviceClass: 'svc.test',
      operatorClass: 'op.test',
      name: 'Envelope Variant Record',
      description: 'legacy envelope',
      tags: ['envelope'],
      spec: {
        label: 'Envelope Variant Spec',
        knobs: 4,
      },
      createdAt: '2026-04-01T00:00:00.000Z',
      updatedAt: '2026-04-02T00:00:00.000Z',
    },
  });
  await env.DB.prepare(
    `UPDATE asset_versions
     SET content = ?
     WHERE asset_id = ? AND version_number = 1`,
  )
    .bind(gzipSync(Buffer.from(envelopeBlob)), 'variant-canonical')
    .run();

  const envelopeContent = await jsonRequest(app, env, '/v1/variants/variant-canonical/content');
  assert.equal(envelopeContent.status, 400);
  assert.equal(envelopeContent.json.message, 'stored variant content must be the raw spec JSON object without record or envelope metadata');
});

test('management APIs support Better Auth backed user and asset management', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const managementLogin = await signInUser(app, env, {
    email: 'admin@example.com',
  });
  assert.equal(managementLogin.status, 200);
  assert.ok(managementLogin.cookie);

  const alice = await createVerifiedSession(app, env, {
    name: 'Alice',
    email: 'alice@example.com',
  });

  const createdByAlice = await jsonRequest(app, env, '/v1/variants', {
    method: 'POST',
    cookie: alice.cookie,
    payload: variantPayload({ variantId: 'alice-private-asset', name: 'Alice Private Asset', visibility: 'private' }),
  });
  assert.equal(createdByAlice.status, 200);

  const nonAdminDenied = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/users`, { cookie: alice.cookie });
  assert.equal(nonAdminDenied.status, 403);

  const managementUsers = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/users`, { cookie: managementLogin.cookie });
  assert.equal(managementUsers.status, 200);
  assert.equal(managementUsers.json.entries.length >= 2, true);

  const managementCreatesUser = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/users`, {
    method: 'POST',
    cookie: managementLogin.cookie,
    payload: {
      name: 'Ops',
      email: 'ops@example.com',
      password: TEST_PASSWORD,
      role: 'readonly',
    },
  });
  assert.equal(managementCreatesUser.status, 200);
  const opsUserId = String(managementCreatesUser.json.userId);
  assert.equal(managementCreatesUser.json.role, 'readonly');

  const managementUpdatesUser = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/users/${opsUserId}`, {
    method: 'PUT',
    cookie: managementLogin.cookie,
    payload: {
      name: 'Ops Team',
      role: 'user',
      password: TEST_PASSWORD_2,
    },
  });
  assert.equal(managementUpdatesUser.status, 200);
  assert.equal(managementUpdatesUser.json.name, 'Ops Team');
  assert.equal(Object.hasOwn(managementUpdatesUser.json, 'displayName'), false);
  assert.equal(managementUpdatesUser.json.role, 'user');

  const managementUsersAfterUpdate = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/users`, {
    cookie: managementLogin.cookie,
  });
  assert.equal(managementUsersAfterUpdate.status, 200);
  assert.equal(
    managementUsersAfterUpdate.json.entries.some((entry) => entry.userId === opsUserId && entry.role === 'user'),
    true,
  );

  const nameConflict = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/users/${opsUserId}`, {
    method: 'PUT',
    cookie: managementLogin.cookie,
    payload: {
      name: 'Alice',
    },
  });
  assert.equal(nameConflict.status, 409);

  const managementViewsAliceAssets = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/variants?ownerUserId=${encodeURIComponent(alice.userId)}`, {
    cookie: managementLogin.cookie,
  });
  assert.equal(managementViewsAliceAssets.status, 200);
  assert.equal(managementViewsAliceAssets.json.entries.length, 1);
  assert.equal(managementViewsAliceAssets.json.entries[0].assetId, 'alice-private-asset');

  const managementListsAssets = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/variants`, {
    cookie: managementLogin.cookie,
  });
  assert.equal(managementListsAssets.status, 200);
  assert.equal(managementListsAssets.json.entries.length >= 1, true);
  assert.equal(managementListsAssets.json.entries[0].variantKind, 'operator');

  const managementVariantDetail = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/variants/alice-private-asset`, {
    cookie: managementLogin.cookie,
  });
  assert.equal(managementVariantDetail.status, 200);
  assert.equal(managementVariantDetail.json.assetId, 'alice-private-asset');
  assert.equal(managementVariantDetail.json.variantKind, 'operator');
  assert.equal(managementVariantDetail.json.baseNodeType, 'svc.base.op');

  const managementChangesVisibility = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/variants/alice-private-asset`, {
    method: 'PUT',
    cookie: managementLogin.cookie,
    payload: { visibility: 'public' },
  });
  assert.equal(managementChangesVisibility.status, 200);
  assert.equal(managementChangesVisibility.json.visibility, 'public');

  const managementDeletesAsset = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/variants/alice-private-asset`, {
    method: 'DELETE',
    cookie: managementLogin.cookie,
  });
  assert.equal(managementDeletesAsset.status, 200);

  const hiddenFromDefaultManagementList = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/variants`, {
    cookie: managementLogin.cookie,
  });
  assert.equal(hiddenFromDefaultManagementList.status, 200);
  assert.equal(
    hiddenFromDefaultManagementList.json.entries.some((entry) => entry.assetId === 'alice-private-asset'),
    false,
  );

  const deletedManagementDetail = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/variants/alice-private-asset`, {
    cookie: managementLogin.cookie,
  });
  assert.equal(deletedManagementDetail.status, 404);

  const managementLocksAliceUploads = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/users/${alice.userId}`, {
    method: 'PUT',
    cookie: managementLogin.cookie,
    payload: {
      role: 'readonly',
    },
  });
  assert.equal(managementLocksAliceUploads.status, 200);
  assert.equal(managementLocksAliceUploads.json.role, 'readonly');

  const aliceBlockedUpload = await jsonRequest(app, env, '/v1/variants', {
    method: 'POST',
    cookie: alice.cookie,
    payload: variantPayload({ variantId: 'alice-blocked-upload', name: 'Alice Blocked Upload', visibility: 'private' }),
  });
  assert.equal(aliceBlockedUpload.status, 403);
  assert.equal(aliceBlockedUpload.json.message, 'upload permission required');

  const managementUnlocksAliceUploads = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/users/${alice.userId}`, {
    method: 'PUT',
    cookie: managementLogin.cookie,
    payload: {
      role: 'user',
    },
  });
  assert.equal(managementUnlocksAliceUploads.status, 200);
  assert.equal(managementUnlocksAliceUploads.json.role, 'user');

  const aliceAllowedUpload = await jsonRequest(app, env, '/v1/variants', {
    method: 'POST',
    cookie: alice.cookie,
    payload: variantPayload({ variantId: 'alice-allowed-upload', name: 'Alice Allowed Upload', visibility: 'private' }),
  });
  assert.equal(aliceAllowedUpload.status, 200);

  const deleteAliceBlocked = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/users/${alice.userId}`, {
    method: 'DELETE',
    cookie: managementLogin.cookie,
  });
  assert.equal(deleteAliceBlocked.status, 409);

  const managementSelf = String(managementLogin.json.user.id);
  const selfDemotion = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/users/${managementSelf}`, {
    method: 'PUT',
    cookie: managementLogin.cookie,
    payload: {
      role: 'user',
    },
  });
  assert.equal(selfDemotion.status, 400);

  const selfDelete = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/users/${managementSelf}`, {
    method: 'DELETE',
    cookie: managementLogin.cookie,
  });
  assert.equal(selfDelete.status, 400);

  const deleteOps = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/users/${opsUserId}`, {
    method: 'DELETE',
    cookie: managementLogin.cookie,
  });
  assert.equal(deleteOps.status, 200);
});

test('site settings default to registration disabled and management can enable registration', async (t) => {
  const env = createEnv({ allowUserRegistration: false });
  t.after(() => env.DB.close());
  const app = createApp();

  const initialSettings = await jsonRequest(app, env, '/v1/site-settings');
  assert.equal(initialSettings.status, 200);
  assert.equal(initialSettings.json.allowUserRegistration, false);

  const blockedSignUp = await signUpUser(app, env, {
    name: 'Blocked User',
    email: 'blocked@example.com',
  });
  assert.equal(blockedSignUp.status, 403);
  assert.equal(blockedSignUp.json.message, 'new user registration is disabled');

  const managementLogin = await signInUser(app, env, {
    email: 'admin@example.com',
  });
  assert.equal(managementLogin.status, 200);

  const managementSettings = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/site-settings`, {
    cookie: managementLogin.cookie,
  });
  assert.equal(managementSettings.status, 200);
  assert.equal(managementSettings.json.allowUserRegistration, false);

  const enabledSettings = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/site-settings`, {
    method: 'PUT',
    cookie: managementLogin.cookie,
    payload: {
      allowUserRegistration: true,
    },
  });
  assert.equal(enabledSettings.status, 200);
  assert.equal(enabledSettings.json.allowUserRegistration, true);

  const allowedSignUp = await signUpUser(app, env, {
    name: 'Allowed User',
    email: 'allowed@example.com',
  });
  assert.equal(allowedSignUp.status, 200);
  assert.equal(allowedSignUp.json.user.name, 'Allowed User');
});

test('management can permanently purge all assets', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const managementLogin = await signInUser(app, env, {
    email: 'admin@example.com',
  });
  assert.equal(managementLogin.status, 200);

  const alice = await createVerifiedSession(app, env, {
    name: 'Alice',
    email: 'alice@example.com',
  });

  const createdVariant = await jsonRequest(app, env, '/v1/variants', {
    method: 'POST',
    cookie: alice.cookie,
    payload: variantPayload({ variantId: 'alice-variant', name: 'Alice Variant', visibility: 'public' }),
  });
  assert.equal(createdVariant.status, 200);

  const createdComponent = await jsonRequest(app, env, '/v1/components', {
    method: 'POST',
    cookie: alice.cookie,
    payload: componentPayload({ componentId: 'alice-component', name: 'Alice Component', visibility: 'public' }),
  });
  assert.equal(createdComponent.status, 200);

  const variantUpdate = await jsonRequest(app, env, '/v1/variants/alice-variant', {
    method: 'PUT',
    cookie: alice.cookie,
    payload: {
      ...variantPayload({ variantId: 'alice-variant', name: 'Alice Variant v2', visibility: 'public' }),
      versionNumber: 1,
    },
  });
  assert.equal(variantUpdate.status, 200);

  const subscribeComponent = await jsonRequest(app, env, '/v1/components/alice-component/subscribe', {
    method: 'POST',
    cookie: managementLogin.cookie,
  });
  assert.equal(subscribeComponent.status, 200);

  const subscribeVariant = await jsonRequest(app, env, '/v1/variants/alice-variant/subscribe', {
    method: 'POST',
    cookie: managementLogin.cookie,
  });
  assert.equal(subscribeVariant.status, 200);

  const rejectedPurge = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/assets/purge-all`, {
    method: 'POST',
    cookie: managementLogin.cookie,
    payload: {
      confirmationText: 'nope',
    },
  });
  assert.equal(rejectedPurge.status, 400);

  const purge = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/assets/purge-all`, {
    method: 'POST',
    cookie: managementLogin.cookie,
    payload: {
      confirmationText: 'DELETE ALL ASSETS',
    },
  });
  assert.equal(purge.status, 200);
  assert.equal(purge.json.deletedAssets, 2);
  assert.equal(purge.json.deletedAssetVersions, 3);
  assert.equal(purge.json.deletedAssetSubscriptions, 2);
  assert.equal(purge.json.deletedVariantDetails, 1);

  const managedVariants = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/variants`, {
    cookie: managementLogin.cookie,
  });
  assert.equal(managedVariants.status, 200);
  assert.deepEqual(managedVariants.json.entries, []);

  const managedComponents = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/components`, {
    cookie: managementLogin.cookie,
  });
  assert.equal(managedComponents.status, 200);
  assert.deepEqual(managedComponents.json.entries, []);

  const publicVariants = await jsonRequest(app, env, '/v1/variants?owner=public');
  assert.equal(publicVariants.status, 200);
  assert.deepEqual(publicVariants.json.entries, []);

  const publicComponents = await jsonRequest(app, env, '/v1/components?owner=public');
  assert.equal(publicComponents.status, 200);
  assert.deepEqual(publicComponents.json.entries, []);

  const userDirectory = await jsonRequest(app, env, `${MANAGEMENT_API_BASE_PATH}/users`, {
    cookie: managementLogin.cookie,
  });
  assert.equal(userDirectory.status, 200);
  const aliceEntry = userDirectory.json.entries.find((entry) => entry.email === 'alice@example.com');
  assert.equal(aliceEntry.assetCount, 0);
});

test('root portal entry page is served as html', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const rootResponse = await app.fetch(new Request('http://worker.test/'), env, {});
  assert.equal(rootResponse.status, 200);
  assert.match(rootResponse.headers.get('Content-Type') || '', /text\/html/);

  const html = await rootResponse.text();
  assert.match(html, /Feel8 Asset Cloud/);
});

test('/console routes are no longer served by the worker portal', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const consoleRootResponse = await app.fetch(new Request('http://worker.test/console'), env, {});
  assert.equal(consoleRootResponse.status, 404);

  const consoleLoginResponse = await app.fetch(new Request('http://worker.test/console/login'), env, {});
  assert.equal(consoleLoginResponse.status, 404);

  const consoleAssetResponse = await app.fetch(new Request('http://worker.test/console/assets/mine'), env, {});
  assert.equal(consoleAssetResponse.status, 404);
});

test('auth helper pages are served as html', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const verifyResponse = await app.fetch(new Request('http://worker.test/verify-email?token=test-token'), env, {});
  assert.equal(verifyResponse.status, 200);
  assert.match(verifyResponse.headers.get('Content-Type') || '', /text\/html/);
  const verifyHtml = await verifyResponse.text();
  assert.match(verifyHtml, /Feel8 Asset Cloud/);

  const resetResponse = await app.fetch(new Request('http://worker.test/reset-password?token=test-token'), env, {});
  assert.equal(resetResponse.status, 200);
  assert.match(resetResponse.headers.get('Content-Type') || '', /text\/html/);
  const resetHtml = await resetResponse.text();
  assert.match(resetHtml, /Feel8 Asset Cloud/);
});

test('portal client routes under root are served as html', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const myAssetsResponse = await app.fetch(new Request('http://worker.test/assets/mine'), env, {});
  assert.equal(myAssetsResponse.status, 200);
  assert.match(myAssetsResponse.headers.get('Content-Type') || '', /text\/html/);
  assert.match(await myAssetsResponse.text(), /Feel8 Asset Cloud/);

  const browseResponse = await app.fetch(new Request('http://worker.test/browse'), env, {});
  assert.equal(browseResponse.status, 200);
  assert.match(browseResponse.headers.get('Content-Type') || '', /text\/html/);
  assert.match(await browseResponse.text(), /Feel8 Asset Cloud/);

  const publicAssetResponse = await app.fetch(new Request('http://worker.test/assets/public-demo-asset'), env, {});
  assert.equal(publicAssetResponse.status, 200);
  assert.match(publicAssetResponse.headers.get('Content-Type') || '', /text\/html/);
  assert.match(await publicAssetResponse.text(), /Feel8 Asset Cloud/);
});

test('scheduled cleanup removes stale rate limit rows', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const now = Date.now();
  await env.DB.prepare('INSERT INTO rateLimit (id, key, count, lastRequest) VALUES (?, ?, ?, ?)')
    .bind('old-rate-limit', 'worker:desktop_refresh:old', 3, now - (25 * 60 * 60 * 1000))
    .run();
  await env.DB.prepare('INSERT INTO rateLimit (id, key, count, lastRequest) VALUES (?, ?, ?, ?)')
    .bind('fresh-rate-limit', 'worker:desktop_refresh:fresh', 1, now)
    .run();

  await worker.scheduled({}, env, {
    waitUntil(promise) {
      return promise;
    },
  });

  const oldRow = await env.DB.prepare('SELECT key FROM rateLimit WHERE key = ?')
    .bind('worker:desktop_refresh:old')
    .first();
  const freshRow = await env.DB.prepare('SELECT key FROM rateLimit WHERE key = ?')
    .bind('worker:desktop_refresh:fresh')
    .first();
  assert.equal(oldRow, null);
  assert.equal(freshRow.key, 'worker:desktop_refresh:fresh');
});

test('worker leaves typed list responses uncompressed by default', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());

  const listRequest = new Request('http://worker.test/v1/components?owner=public', {
    headers: {
      'Accept-Encoding': 'gzip',
    },
  });
  const listResponse = await worker.fetch(listRequest, env, {});
  assert.equal(listResponse.status, 200);
  assert.equal(listResponse.headers.get('Content-Encoding'), null);
});

test('worker gzips large asset payload responses by default and leaves auth/list uncompressed', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const alice = await createVerifiedSession(app, env, {
    name: 'Alice',
    email: 'alice@example.com',
  });

  const created = await jsonRequest(app, env, '/v1/components', {
    method: 'POST',
    cookie: alice.cookie,
    payload: componentPayload({ componentId: 'component-gzip', name: 'Compressed Component', visibility: 'public' }),
  });
  assert.equal(created.status, 200);

  const listRequest = new Request('http://worker.test/v1/components?owner=public', {
    headers: {
      'Accept-Encoding': 'gzip',
    },
  });
  const listResponse = await worker.fetch(listRequest, env, {});
  assert.equal(listResponse.status, 200);
  assert.equal(listResponse.headers.get('Content-Encoding'), null);

  const detailRequest = new Request('http://worker.test/v1/components/component-gzip/content', {
    headers: {
      'Accept-Encoding': 'gzip',
      cookie: alice.cookie,
    },
  });
  const detailResponse = await worker.fetch(detailRequest, env, {});
  assert.equal(detailResponse.status, 200);
  assert.equal(detailResponse.headers.get('Content-Encoding'), 'gzip');
  assert.match(detailResponse.headers.get('Cache-Control') || '', /(?:^|,\s*)no-transform(?:,|$)/);
  assert.match(detailResponse.headers.get('Content-Type') || '', /application\/json/);
  const compressedBody = Buffer.from(await detailResponse.arrayBuffer());
  const detailPayload = JSON.parse(gunzipSync(compressedBody).toString('utf-8'));
  assert.equal(detailPayload.componentId, 'component-gzip');

  const sessionRequest = new Request('http://worker.test/api/auth/get-session', {
    headers: {
      'Accept-Encoding': 'gzip',
      cookie: alice.cookie,
    },
  });
  const sessionResponse = await worker.fetch(sessionRequest, env, {});
  assert.equal(sessionResponse.status, 200);
  assert.equal(sessionResponse.headers.get('Content-Encoding'), null);
});

test('worker can accept gzip-compressed asset JSON request bodies', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const alice = await createVerifiedSession(app, env, {
    name: 'Alice',
    email: 'alice@example.com',
  });

  const compressedPayload = gzipSync(Buffer.from(JSON.stringify(
    componentPayload({ componentId: 'component-gzip-upload', name: 'Compressed Upload', visibility: 'private' }),
  )));

  const createRequest = new Request('http://worker.test/v1/components', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Encoding': 'gzip',
      cookie: alice.cookie,
      origin: 'http://worker.test',
    },
    body: compressedPayload,
  });
  const createResponse = await worker.fetch(createRequest, env, {});
  assert.equal(createResponse.status, 200);
  const createJson = JSON.parse(await createResponse.text());
  assert.equal(createJson.componentId, 'component-gzip-upload');
});

test('worker rejects mismatched gzip request body headers', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());

  const alice = await createVerifiedSession(createApp(), env, {
    name: 'Alice',
    email: 'alice@example.com',
  });

  const payloadJson = JSON.stringify(
    componentPayload({ componentId: 'component-invalid-gzip', name: 'Invalid Gzip', visibility: 'private' }),
  );
  const gzippedPayload = gzipSync(Buffer.from(payloadJson));

  const missingHeaderResponse = await worker.fetch(new Request('http://worker.test/v1/components', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      cookie: alice.cookie,
      origin: 'http://worker.test',
    },
    body: gzippedPayload,
  }), env, {});
  assert.equal(missingHeaderResponse.status, 400);
  assert.equal((await missingHeaderResponse.json()).message, 'request body must be a JSON object');

  const invalidGzipHeaderResponse = await worker.fetch(new Request('http://worker.test/v1/components', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Encoding': 'gzip',
      cookie: alice.cookie,
      origin: 'http://worker.test',
    },
    body: payloadJson,
  }), env, {});
  assert.equal(invalidGzipHeaderResponse.status, 400);
  assert.equal((await invalidGzipHeaderResponse.json()).message, 'request body gzip decompression failed');
});

test('worker only gzips all /v1 json responses when explicitly enabled and still leaves /api/auth uncompressed', async (t) => {
  const env = createEnv({
    ENABLE_API_JSON_GZIP: 'true',
  });
  t.after(() => env.DB.close());

  const listRequest = new Request('http://worker.test/v1/components?owner=public', {
    headers: {
      'Accept-Encoding': 'gzip',
    },
  });
  const listResponse = await worker.fetch(listRequest, env, {});
  assert.equal(listResponse.status, 200);
  assert.equal(listResponse.headers.get('Content-Encoding'), 'gzip');
  assert.match(listResponse.headers.get('Cache-Control') || '', /(?:^|,\s*)no-transform(?:,|$)/);
  assert.match(listResponse.headers.get('Content-Type') || '', /application\/json/);
  const compressedBody = Buffer.from(await listResponse.arrayBuffer());
  const listPayload = JSON.parse(gunzipSync(compressedBody).toString('utf-8'));
  assert.ok(Array.isArray(listPayload.entries));

  const sessionRequest = new Request('http://worker.test/api/auth/get-session', {
    headers: {
      'Accept-Encoding': 'gzip',
    },
  });
  const sessionResponse = await worker.fetch(sessionRequest, env, {});
  assert.equal(sessionResponse.status, 200);
  assert.equal(sessionResponse.headers.get('Content-Encoding'), null);
});

test('type-agnostic asset resolve returns component and variant details', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const alice = await createVerifiedSession(app, env, { name: 'Alice', email: 'alice@example.com' });

  const createdComponent = await jsonRequest(app, env, '/v1/components', {
    method: 'POST',
    cookie: alice.cookie,
    payload: componentPayload({ componentId: 'resolve-component', name: 'Resolve Component', visibility: 'public' }),
  });
  assert.equal(createdComponent.status, 200);

  const createdVariant = await jsonRequest(app, env, '/v1/variants', {
    method: 'POST',
    cookie: alice.cookie,
    payload: variantPayload({ variantId: 'resolve-variant', name: 'Resolve Variant', visibility: 'public' }),
  });
  assert.equal(createdVariant.status, 200);

  const resolvedComponent = await jsonRequest(app, env, '/v1/assets/resolve-component');
  assert.equal(resolvedComponent.status, 200);
  assert.equal(resolvedComponent.json.assetType, 'component');
  assert.equal(resolvedComponent.json.asset.componentId, 'resolve-component');
  assert.equal(resolvedComponent.json.asset.name, 'Resolve Component');

  const resolvedVariant = await jsonRequest(app, env, '/v1/assets/resolve-variant');
  assert.equal(resolvedVariant.status, 200);
  assert.equal(resolvedVariant.json.assetType, 'variant');
  assert.equal(resolvedVariant.json.asset.variantId, 'resolve-variant');

  const missing = await jsonRequest(app, env, '/v1/assets/does-not-exist');
  assert.equal(missing.status, 404);
});

test('asset resolve hides private assets from anonymous and other users', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const alice = await createVerifiedSession(app, env, { name: 'Alice', email: 'alice@example.com' });
  const bob = await createVerifiedSession(app, env, { name: 'Bob', email: 'bob@example.com' });

  const created = await jsonRequest(app, env, '/v1/components', {
    method: 'POST',
    cookie: alice.cookie,
    payload: componentPayload({ componentId: 'private-component', name: 'Hidden', visibility: 'private' }),
  });
  assert.equal(created.status, 200);

  const anonymous = await jsonRequest(app, env, '/v1/assets/private-component');
  assert.equal(anonymous.status, 404);

  const stranger = await jsonRequest(app, env, '/v1/assets/private-component', { cookie: bob.cookie });
  assert.equal(stranger.status, 404);

  const owner = await jsonRequest(app, env, '/v1/assets/private-component', { cookie: alice.cookie });
  assert.equal(owner.status, 200);
  assert.equal(owner.json.assetType, 'component');
});

test('asset download endpoints return attachment responses for public assets', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const alice = await createVerifiedSession(app, env, { name: 'Alice', email: 'alice@example.com' });

  const created = await jsonRequest(app, env, '/v1/components', {
    method: 'POST',
    cookie: alice.cookie,
    payload: componentPayload({ componentId: 'download-component', name: 'Download Me!', visibility: 'public' }),
  });
  assert.equal(created.status, 200);

  const downloadRequest = new Request('http://worker.test/v1/components/download-component/download');
  const downloadResponse = await app.fetch(downloadRequest, env, {});
  assert.equal(downloadResponse.status, 200);
  assert.match(downloadResponse.headers.get('Content-Type') || '', /application\/json/);
  const disposition = downloadResponse.headers.get('Content-Disposition') || '';
  assert.match(disposition, /attachment/);
  assert.match(disposition, /filename="download-me-1\.json"/);

  const body = JSON.parse(await downloadResponse.text());
  assert.equal(body.componentId, 'download-component');
  assert.equal(body.versionNumber, 1);
  assert.ok(body.record);

  const versionDownloadRequest = new Request('http://worker.test/v1/components/download-component/versions/1/download');
  const versionDownloadResponse = await app.fetch(versionDownloadRequest, env, {});
  assert.equal(versionDownloadResponse.status, 200);
  assert.match(versionDownloadResponse.headers.get('Content-Disposition') || '', /filename="download-me-1\.json"/);
});

test('asset download endpoints deny anonymous access to private assets', async (t) => {
  const env = createEnv();
  t.after(() => env.DB.close());
  const app = createApp();

  const alice = await createVerifiedSession(app, env, { name: 'Alice', email: 'alice@example.com' });

  const created = await jsonRequest(app, env, '/v1/variants', {
    method: 'POST',
    cookie: alice.cookie,
    payload: variantPayload({ variantId: 'private-variant', name: 'Private Variant', visibility: 'private' }),
  });
  assert.equal(created.status, 200);

  const anonymousRequest = new Request('http://worker.test/v1/variants/private-variant/download');
  const anonymousResponse = await app.fetch(anonymousRequest, env, {});
  assert.equal(anonymousResponse.status, 404);

  const ownerRequest = new Request('http://worker.test/v1/variants/private-variant/download', {
    headers: { cookie: alice.cookie },
  });
  const ownerResponse = await app.fetch(ownerRequest, env, {});
  assert.equal(ownerResponse.status, 200);
  assert.match(ownerResponse.headers.get('Content-Disposition') || '', /filename="private-variant-1\.json"/);
});
