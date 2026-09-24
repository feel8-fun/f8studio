import { betterAuth, generateId } from 'better-auth';
import { APIError } from '@better-auth/core/error';
import { drizzleAdapter } from 'better-auth/adapters/drizzle';
import { admin, captcha } from 'better-auth/plugins';
import { ApiException } from 'chanfana';
import { drizzle } from 'drizzle-orm/d1';
import { Hono } from 'hono';
import { cors } from 'hono/cors';

import { authSchema } from './auth_schema.js';
import { registerOpenApiRoutes } from './openapi.js';
import { authPasswordHashVersion, hashAuthPassword, verifyAuthPassword } from './password.js';
import { AssetConflictError, AssetNotFoundError, AssetPermissionError, AssetValidationError, AssetRepository } from './repository.js';
import { decompressGzip, isPlainObject, stringOrDefault, toBoolean } from './utils.js';

const AUTH_BASE_PATH = '/api/auth';
const DESKTOP_AUTH_BASE_PATH = '/v1/auth/desktop';
const MANAGEMENT_API_BASE_PATH = '/v1/management';
const PORTAL_STATIC_DIR = '/_portal';
const PURGE_ALL_ASSETS_CONFIRMATION_TEXT = 'DELETE ALL ASSETS';
const DESKTOP_AUTHORIZATION_CODE_TTL_SECONDS = 300;
const DESKTOP_ACCESS_TOKEN_TTL_SECONDS = 3600;
const DESKTOP_REFRESH_TOKEN_TTL_SECONDS = 30 * 24 * 3600;
const DESKTOP_REQUEST_PURGE_INTERVAL_MS = 60 * 1000;
const DESKTOP_AUTH_CONFIRM_CSRF_COOKIE = 'f8assetcloud_desktop_csrf';
const DESKTOP_AUTH_ALLOWED_CLIENT_IDS = new Set(['f8studio']);
const MAX_REQUEST_COMPRESSED_BYTES = 12 * 1024 * 1024;
const MAX_REQUEST_JSON_BYTES = 12 * 1024 * 1024;
const PUBLIC_CACHE_CONTROL_HEADER = 'public, max-age=120, stale-while-revalidate=300';
const USER_ROLE_ADMIN = 'admin';
const USER_ROLE_USER = 'user';
const USER_ROLE_READONLY = 'readonly';
const BOOTSTRAP_ADMIN_STATE_ROW_ID = 1;
const RESERVED_IDENTITY_NAMES = new Set([
  'admin',
  'administrator',
  'owner',
  'root',
  'system',
  'feel8',
  'feel8fun',
  'f8',
  'f8studio',
  'support',
  'staff',
  'moderator',
  'official',
]);
const textEncoder = new TextEncoder();
let bootstrapAdminInitByDb = new WeakMap();
let authCacheByDb = new WeakMap();
let lastDesktopAuthorizationCodePurgeByDb = new WeakMap();
let lastDesktopSessionPurgeByDb = new WeakMap();

export function createApp() {
  const app = new Hono();

  app.use('*', async (c, next) => {
    validateEnv(c.env);
    await next();
  });

  app.use('*', cors({
    origin: (origin, c) => resolveAllowedOrigin(c.env, origin, new URL(c.req.url)),
    credentials: true,
    allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'HEAD'],
    allowHeaders: ['Authorization', 'Content-Type', 'X-Captcha-Response'],
  }));

  app.all(`${AUTH_BASE_PATH}/*`, async (c) => {
    const siteSettings = await readSiteSettings(c.env.DB);
    const auth = await getOrCreateAuth(c.env, c.req.raw, siteSettings);
    await ensureBootstrapAdmin({ env: c.env });
    if (shouldBlockPublicRegistration({ siteSettings, request: c.req.raw })) {
      return jsonResponse(403, { message: 'new user registration is disabled' });
    }
    return auth.handler(c.req.raw);
  });

  app.use('/v1/*', async (c, next) => {
    const auth = await getOrCreateAuth(c.env, c.req.raw);
    await ensureBootstrapAdmin({ env: c.env });
    c.set('auth', auth);
    c.set('repo', new AssetRepository(c.env.DB));
    c.set('allowedOrigins', calculateAllowedOrigins(c.env));
    await next();
  });

  registerOpenApiRoutes(app, {
    getAuthProviders: async (c) => ({
      google: hasGoogleProvider(c.env),
      turnstileSiteKey: turnstileSiteKey(c.env),
    }),
    getSiteSettings: async (c) => c.get('repo').getSiteSettings(),
    routeDesktopTokenPost: async (c) => routeDesktopTokenPost({
      db: c.env.DB,
      request: c.req.raw,
    }),
    routeDesktopSessionPost: async (c) => routeDesktopSessionPost({
      auth: c.get('auth'),
      db: c.env.DB,
      request: c.req.raw,
      allowedOrigins: c.get('allowedOrigins'),
    }),
    routeDesktopRefreshPost: async (c) => routeDesktopRefreshPost({
      db: c.env.DB,
      request: c.req.raw,
    }),
    routeDesktopRevokePost: async (c) => routeDesktopRevokePost({
      db: c.env.DB,
      request: c.req.raw,
    }),
    resolveAsset: async (c) => {
      const auth = c.get('auth');
      const repo = c.get('repo');
      const viewer = await optionalAuthenticatedUser({ auth, db: c.env.DB, request: c.req.raw });
      const viewerId = viewer === null ? null : viewer.userId;
      const assetId = c.req.param('assetId');
      const head = await repo.getAssetById(assetId);
      if (head === null) {
        throw new HttpError(404, 'not found');
      }
      if (String(head.visibility) !== 'public' && String(head.owner_user_id) !== String(viewerId || '')) {
        throw new HttpError(404, 'not found');
      }
      const assetType = String(head.asset_type);
      const asset = await getTypedAsset({ repo, assetType, assetId, userId: viewerId, head });
      applyAnonymousPublicCacheHeaders(c, c.req.raw, String(head.visibility) === 'public' && viewer === null);
      return { assetType, asset };
    },
    getMe: async (c) => {
      const user = await requireAuthenticatedUser({
        auth: c.get('auth'),
        db: c.env.DB,
        request: c.req.raw,
        allowedOrigins: c.get('allowedOrigins'),
      });
      return toApiUser(user);
    },
    updateMe: async (c) => {
      const repo = c.get('repo');
      const user = await requireAuthenticatedUser({
        auth: c.get('auth'),
        db: c.env.DB,
        request: c.req.raw,
        allowedOrigins: c.get('allowedOrigins'),
      });
      const payload = await readJsonBody(c.req.raw);
      const name = requireUserProfileName(payload.name, {
        currentName: user.name,
        allowReserved: user.isAdmin,
      });
      if (name !== user.name) {
        if (await repo.isUserNameTaken({ name, excludeUserId: user.userId })) {
          throw new HttpError(409, 'name already in use');
        }
        await updateAppUserName(c.env.DB, {
          userId: user.userId,
          name,
        });
      }
      const updated = await repo.getUserByIdWithStats(user.userId);
      if (updated === null) {
        throw new HttpError(404, 'user not found');
      }
      return toApiUser(updated);
    },
    listComponents: async (c) => {
      const viewer = await optionalAuthenticatedUser({ auth: c.get('auth'), db: c.env.DB, request: c.req.raw });
      applyAnonymousPublicCacheHeaders(c, c.req.raw, viewer === null);
      return c.get('repo').listComponents({
        userId: viewer === null ? null : viewer.userId,
        query: c.req.query('q') || '',
        visibility: c.req.query('visibility') || '',
        owner: c.req.query('owner') || '',
        cursor: c.req.query('cursor') || '',
      });
    },
    getComponentContent: async (c) => {
      const viewer = await optionalAuthenticatedUser({ auth: c.get('auth'), db: c.env.DB, request: c.req.raw });
      const result = await c.get('repo').getComponentContent({
        componentId: c.req.param('componentId'),
        userId: viewer === null ? null : viewer.userId,
      });
      applyAnonymousPublicCacheHeaders(c, c.req.raw, viewer === null);
      return result;
    },
    listVariants: async (c) => {
      const viewer = await optionalAuthenticatedUser({ auth: c.get('auth'), db: c.env.DB, request: c.req.raw });
      applyAnonymousPublicCacheHeaders(c, c.req.raw, viewer === null);
      return c.get('repo').listVariants({
        userId: viewer === null ? null : viewer.userId,
        kind: c.req.query('kind') || '',
        baseNodeType: c.req.query('baseNodeType') || '',
        query: c.req.query('q') || '',
        visibility: c.req.query('visibility') || '',
        owner: c.req.query('owner') || '',
        cursor: c.req.query('cursor') || '',
      });
    },
    createVariant: async (c) => {
      const repo = c.get('repo');
      const user = await requireAssetWriteUser({
        auth: c.get('auth'),
        db: c.env.DB,
        repo,
        request: c.req.raw,
        allowedOrigins: c.get('allowedOrigins'),
      });
      const payload = await readJsonBody(c.req.raw);
      return repo.createVariant({ payload, user });
    },
    createComponent: async (c) => {
      const repo = c.get('repo');
      const user = await requireAssetWriteUser({
        auth: c.get('auth'),
        db: c.env.DB,
        repo,
        request: c.req.raw,
        allowedOrigins: c.get('allowedOrigins'),
      });
      const payload = await readJsonBody(c.req.raw);
      return repo.createComponent({ payload, user });
    },
    listModdingRecipes: async (c) => {
      const viewer = await optionalAuthenticatedUser({ auth: c.get('auth'), db: c.env.DB, request: c.req.raw });
      applyAnonymousPublicCacheHeaders(c, c.req.raw, viewer === null);
      return c.get('repo').listModdingRecipes({
        userId: viewer === null ? null : viewer.userId,
        query: c.req.query('q') || '',
        visibility: c.req.query('visibility') || '',
        owner: c.req.query('owner') || '',
        cursor: c.req.query('cursor') || '',
      });
    },
    createModdingRecipe: async (c) => {
      const repo = c.get('repo');
      const user = await requireAssetWriteUser({
        auth: c.get('auth'),
        db: c.env.DB,
        repo,
        request: c.req.raw,
        allowedOrigins: c.get('allowedOrigins'),
      });
      const payload = await readJsonBody(c.req.raw);
      return repo.createModdingRecipe({ payload, user });
    },
    routeVariantAssetRequest: async (c) => routeAssetRequest({
      auth: c.get('auth'),
      db: c.env.DB,
      repo: c.get('repo'),
      request: c.req.raw,
      url: new URL(c.req.raw.url),
      assetType: 'variant',
      allowedOrigins: c.get('allowedOrigins'),
    }),
    routeComponentAssetRequest: async (c) => routeAssetRequest({
      auth: c.get('auth'),
      db: c.env.DB,
      repo: c.get('repo'),
      request: c.req.raw,
      url: new URL(c.req.raw.url),
      assetType: 'component',
      allowedOrigins: c.get('allowedOrigins'),
    }),
    routeModdingRecipeAssetRequest: async (c) => routeAssetRequest({
      auth: c.get('auth'),
      db: c.env.DB,
      repo: c.get('repo'),
      request: c.req.raw,
      url: new URL(c.req.raw.url),
      assetType: 'modding_recipe',
      allowedOrigins: c.get('allowedOrigins'),
    }),
    routeManagementRequest: async (c) => {
      const managementUser = await requireManagementUser({
        auth: c.get('auth'),
        db: c.env.DB,
        request: c.req.raw,
        allowedOrigins: c.get('allowedOrigins'),
      });
      return routeManagementRequest({
        db: c.env.DB,
        env: c.env,
        managementUser,
        auth: c.get('auth'),
        repo: c.get('repo'),
        request: c.req.raw,
        url: new URL(c.req.raw.url),
        allowedOrigins: c.get('allowedOrigins'),
      });
    },
  });

  app.get('/v1/auth/verify-email', async (c) => {
    const auth = c.get('auth');
    const token = requireQueryString(c.req.query('token'), 'token is required');
    await auth.api.verifyEmail({
      query: { token },
      headers: c.req.raw.headers,
    });
    return c.json({ verified: true });
  });

  app.post('/v1/auth/reset-password', async (c) => {
    const auth = c.get('auth');
    const payload = await readJsonBody(c.req.raw);
    await auth.api.resetPassword({
      body: {
        token: requireBodyString(payload.token, 'token is required'),
        newPassword: requireBodyString(payload.newPassword, 'newPassword is required'),
      },
      headers: c.req.raw.headers,
    });
    return c.json({ reset: true });
  });

  app.get(`${DESKTOP_AUTH_BASE_PATH}/authorize`, async (c) => {
    const auth = c.get('auth');
    const siteSettings = await c.get('repo').getSiteSettings();
    return routeDesktopAuthorizeGet({
      auth,
      db: c.env.DB,
      env: c.env,
      request: c.req.raw,
      siteSettings,
    });
  });

  app.post(`${DESKTOP_AUTH_BASE_PATH}/authorize`, async (c) => {
    const auth = c.get('auth');
    const siteSettings = await c.get('repo').getSiteSettings();
    return routeDesktopAuthorizePost({
      auth,
      db: c.env.DB,
      env: c.env,
      request: c.req.raw,
      siteSettings,
    });
  });

  app.post(`${DESKTOP_AUTH_BASE_PATH}/token`, async (c) => {
    return routeDesktopTokenPost({
      db: c.env.DB,
      request: c.req.raw,
    });
  });

  app.post(`${DESKTOP_AUTH_BASE_PATH}/session`, async (c) => routeDesktopSessionPost({
    auth: c.get('auth'),
    db: c.env.DB,
    request: c.req.raw,
    allowedOrigins: c.get('allowedOrigins'),
  }));

  app.post(`${DESKTOP_AUTH_BASE_PATH}/refresh`, async (c) => routeDesktopRefreshPost({
    db: c.env.DB,
    request: c.req.raw,
  }));

  app.post(`${DESKTOP_AUTH_BASE_PATH}/revoke`, async (c) => routeDesktopRevokePost({
    db: c.env.DB,
    request: c.req.raw,
  }));

  app.post('/v1/me/password', async (c) => {
    const user = await requireAuthenticatedUser({
      auth: c.get('auth'),
      db: c.env.DB,
      request: c.req.raw,
      allowedOrigins: c.get('allowedOrigins'),
    });
    const payload = await readJsonBody(c.req.raw);
    await changeAppUserPassword(c.env.DB, {
      userId: user.userId,
      currentPassword: requireBodyString(payload.currentPassword, 'currentPassword is required'),
      newPassword: requireBodyString(payload.newPassword, 'newPassword is required'),
    });
    return c.json({});
  });

  app.get('/v1/variants', async (c) => {
    const repo = c.get('repo');
    const viewer = await optionalAuthenticatedUser({ auth: c.get('auth'), db: c.env.DB, request: c.req.raw });
    applyAnonymousPublicCacheHeaders(c, c.req.raw, viewer === null);
    const result = await repo.listVariants({
      userId: viewer === null ? null : viewer.userId,
      kind: c.req.query('kind') || '',
      baseNodeType: c.req.query('baseNodeType') || '',
      query: c.req.query('q') || '',
      visibility: c.req.query('visibility') || '',
      owner: c.req.query('owner') || '',
      cursor: c.req.query('cursor') || '',
    });
    return c.json(result);
  });

  app.post('/v1/variants', async (c) => {
    const repo = c.get('repo');
    const user = await requireAssetWriteUser({
      auth: c.get('auth'),
      db: c.env.DB,
      repo,
      request: c.req.raw,
      allowedOrigins: c.get('allowedOrigins'),
    });
    const payload = await readJsonBody(c.req.raw);
    return c.json(await repo.createVariant({ payload, user }));
  });

  app.post('/v1/components', async (c) => {
    const repo = c.get('repo');
    const user = await requireAssetWriteUser({
      auth: c.get('auth'),
      db: c.env.DB,
      repo,
      request: c.req.raw,
      allowedOrigins: c.get('allowedOrigins'),
    });
    const payload = await readJsonBody(c.req.raw);
    return c.json(await repo.createComponent({ payload, user }));
  });

  app.get('/v1/modding-recipes', async (c) => {
    const repo = c.get('repo');
    const viewer = await optionalAuthenticatedUser({ auth: c.get('auth'), db: c.env.DB, request: c.req.raw });
    applyAnonymousPublicCacheHeaders(c, c.req.raw, viewer === null);
    const result = await repo.listModdingRecipes({
      userId: viewer === null ? null : viewer.userId,
      query: c.req.query('q') || '',
      visibility: c.req.query('visibility') || '',
      owner: c.req.query('owner') || '',
      cursor: c.req.query('cursor') || '',
    });
    return c.json(result);
  });

  app.post('/v1/modding-recipes', async (c) => {
    const repo = c.get('repo');
    const user = await requireAssetWriteUser({
      auth: c.get('auth'),
      db: c.env.DB,
      repo,
      request: c.req.raw,
      allowedOrigins: c.get('allowedOrigins'),
    });
    const payload = await readJsonBody(c.req.raw);
    return c.json(await repo.createModdingRecipe({ payload, user }));
  });

  app.all('/v1/variants/*', async (c) => routeAssetRequest({
    auth: c.get('auth'),
    db: c.env.DB,
    repo: c.get('repo'),
    request: c.req.raw,
    url: new URL(c.req.raw.url),
    assetType: 'variant',
    allowedOrigins: c.get('allowedOrigins'),
  }));

  app.all('/v1/components/*', async (c) => routeAssetRequest({
    auth: c.get('auth'),
    db: c.env.DB,
    repo: c.get('repo'),
    request: c.req.raw,
    url: new URL(c.req.raw.url),
    assetType: 'component',
    allowedOrigins: c.get('allowedOrigins'),
  }));

  app.all('/v1/modding-recipes/*', async (c) => routeAssetRequest({
    auth: c.get('auth'),
    db: c.env.DB,
    repo: c.get('repo'),
    request: c.req.raw,
    url: new URL(c.req.raw.url),
    assetType: 'modding_recipe',
    allowedOrigins: c.get('allowedOrigins'),
  }));

  app.all(`${MANAGEMENT_API_BASE_PATH}/*`, async (c) => {
    const managementUser = await requireManagementUser({
      auth: c.get('auth'),
      db: c.env.DB,
      request: c.req.raw,
      allowedOrigins: c.get('allowedOrigins'),
    });
    return routeManagementRequest({
      db: c.env.DB,
      env: c.env,
      managementUser,
      auth: c.get('auth'),
      repo: c.get('repo'),
      request: c.req.raw,
      url: new URL(c.req.raw.url),
      allowedOrigins: c.get('allowedOrigins'),
    });
  });

  app.on(['GET', 'HEAD'], '*', async (c) => serveFrontend(c.req.raw, c.env, new URL(c.req.raw.url)));

  app.all('*', () => jsonResponse(404, { message: 'not found' }));
  app.onError((error) => handleError(error));

  return app;
}

export function resetWorkerCachesForTesting() {
  bootstrapAdminInitByDb = new WeakMap();
  authCacheByDb = new WeakMap();
  lastDesktopAuthorizationCodePurgeByDb = new WeakMap();
  lastDesktopSessionPurgeByDb = new WeakMap();
}

function assetRouteConfig(assetType) {
  if (assetType === 'variant') {
    return {
      prefix: '/v1/variants/',
      get: ({ repo, assetId, userId, head = null }) => repo.getVariant({ variantId: assetId, userId, head }),
      update: ({ repo, assetId, payload, user }) => repo.updateVariant({ variantId: assetId, payload, user }),
      delete: ({ repo, assetId, userId }) => repo.deleteVariant({ variantId: assetId, userId }),
      getContent: ({ repo, assetId, userId, head = null }) => repo.getVariantContent({ variantId: assetId, userId, head }),
      updateVisibility: ({ repo, assetId, visibility, versionNumber, userId }) => repo.updateVariantVisibility({ variantId: assetId, visibility, versionNumber, userId }),
      updateMeta: ({ repo, assetId, payload, user }) => repo.updateVariantMeta({ variantId: assetId, payload, user }),
      listVersions: ({ repo, assetId, userId }) => repo.listVariantVersions({ variantId: assetId, userId }),
      getVersion: ({ repo, assetId, versionNumber, userId, head = null }) => repo.getVariantVersion({ variantId: assetId, versionNumber, userId, head }),
      updateVersionNote: ({ repo, assetId, versionNumber, changeSummary, userId }) => repo.updateVariantVersionNote({ variantId: assetId, versionNumber, changeSummary, userId }),
      getVersionContent: ({ repo, assetId, versionNumber, userId, head = null }) => repo.getVariantVersionContent({ variantId: assetId, versionNumber, userId, head }),
      listSubscribers: ({ repo, assetId, userId }) => repo.listVariantSubscribers({ variantId: assetId, userId }),
      subscribe: ({ repo, assetId, userId }) => repo.subscribeVariant({ variantId: assetId, userId }),
      unsubscribe: ({ repo, assetId, userId }) => repo.unsubscribeVariant({ variantId: assetId, userId }),
      fork: ({ repo, assetId, payload, user }) => repo.forkVariant({ variantId: assetId, payload, user }),
    };
  }
  if (assetType === 'component') {
    return {
      prefix: '/v1/components/',
      get: ({ repo, assetId, userId, head = null }) => repo.getComponent({ componentId: assetId, userId, head }),
      update: ({ repo, assetId, payload, user }) => repo.updateComponent({ componentId: assetId, payload, user }),
      delete: ({ repo, assetId, userId }) => repo.deleteComponent({ componentId: assetId, userId }),
      getContent: ({ repo, assetId, userId, head = null }) => repo.getComponentContent({ componentId: assetId, userId, head }),
      updateVisibility: ({ repo, assetId, visibility, versionNumber, userId }) => repo.updateComponentVisibility({ componentId: assetId, visibility, versionNumber, userId }),
      updateMeta: ({ repo, assetId, payload, user }) => repo.updateComponentMeta({ componentId: assetId, payload, user }),
      listVersions: ({ repo, assetId, userId }) => repo.listComponentVersions({ componentId: assetId, userId }),
      getVersion: ({ repo, assetId, versionNumber, userId, head = null }) => repo.getComponentVersion({ componentId: assetId, versionNumber, userId, head }),
      updateVersionNote: ({ repo, assetId, versionNumber, changeSummary, userId }) => repo.updateComponentVersionNote({ componentId: assetId, versionNumber, changeSummary, userId }),
      getVersionContent: ({ repo, assetId, versionNumber, userId, head = null }) => repo.getComponentVersionContent({ componentId: assetId, versionNumber, userId, head }),
      listSubscribers: ({ repo, assetId, userId }) => repo.listComponentSubscribers({ componentId: assetId, userId }),
      subscribe: ({ repo, assetId, userId }) => repo.subscribeComponent({ componentId: assetId, userId }),
      unsubscribe: ({ repo, assetId, userId }) => repo.unsubscribeComponent({ componentId: assetId, userId }),
      fork: ({ repo, assetId, payload, user }) => repo.forkComponent({ componentId: assetId, payload, user }),
    };
  }
  if (assetType === 'modding_recipe') {
    return {
      prefix: '/v1/modding-recipes/',
      get: ({ repo, assetId, userId, head = null }) => repo.getModdingRecipe({ recipeId: assetId, userId, head }),
      update: ({ repo, assetId, payload, user }) => repo.updateModdingRecipe({ recipeId: assetId, payload, user }),
      delete: ({ repo, assetId, userId }) => repo.deleteModdingRecipe({ recipeId: assetId, userId }),
      getContent: ({ repo, assetId, userId, head = null }) => repo.getModdingRecipeContent({ recipeId: assetId, userId, head }),
      updateVisibility: ({ repo, assetId, visibility, versionNumber, userId }) => repo.updateModdingRecipeVisibility({ recipeId: assetId, visibility, versionNumber, userId }),
      updateMeta: ({ repo, assetId, payload, user }) => repo.updateModdingRecipeMeta({ recipeId: assetId, payload, user }),
      listVersions: ({ repo, assetId, userId }) => repo.listModdingRecipeVersions({ recipeId: assetId, userId }),
      getVersion: ({ repo, assetId, versionNumber, userId, head = null }) => repo.getModdingRecipeVersion({ recipeId: assetId, versionNumber, userId, head }),
      updateVersionNote: ({ repo, assetId, versionNumber, changeSummary, userId }) => repo.updateModdingRecipeVersionNote({ recipeId: assetId, versionNumber, changeSummary, userId }),
      getVersionContent: ({ repo, assetId, versionNumber, userId, head = null }) => repo.getModdingRecipeVersionContent({ recipeId: assetId, versionNumber, userId, head }),
      listSubscribers: ({ repo, assetId, userId }) => repo.listModdingRecipeSubscribers({ recipeId: assetId, userId }),
      subscribe: ({ repo, assetId, userId }) => repo.subscribeModdingRecipe({ recipeId: assetId, userId }),
      unsubscribe: ({ repo, assetId, userId }) => repo.unsubscribeModdingRecipe({ recipeId: assetId, userId }),
      fork: ({ repo, assetId, payload, user }) => repo.forkModdingRecipe({ recipeId: assetId, payload, user }),
    };
  }
  throw new HttpError(404, 'not found');
}

function getTypedAsset({ repo, assetType, assetId, userId, head = null }) {
  return assetRouteConfig(assetType).get({ repo, assetId, userId, head });
}

async function routeAssetRequest({ auth, db, repo, request, url, assetType, allowedOrigins }) {
  const assetConfig = assetRouteConfig(assetType);
  const prefix = assetConfig.prefix;
  const tail = decodeURIComponent(url.pathname.slice(prefix.length));
  const parts = tail.split('/').filter((part) => part.length > 0);
  const assetId = parts[0] || '';
  if (!assetId) {
    return jsonResponse(404, { message: 'not found' });
  }
  if (isStateChangingRequestMethod(request.method)) {
    await enforceWorkerRateLimit({
      db,
      key: buildWorkerRateLimitKey({
        namespace: `${assetType}_${parts[1] || 'root'}_${String(request.method).toLowerCase()}`,
        request,
      }),
      windowSeconds: 60,
      maxRequests: parts[1] === 'subscribe' ? 20 : 30,
    });
  }

  if (parts.length === 1) {
    if (request.method === 'GET') {
      const viewer = await optionalAuthenticatedUser({ auth, db, request });
      const result = await assetConfig.get({ repo, assetId, userId: viewer === null ? null : viewer.userId });
      return jsonResponse(200, result, anonymousPublicResponseHeaders(request, viewer === null && String(result.visibility) === 'public'));
    }
    if (request.method === 'PUT') {
      const user = await requireAssetWriteUser({ auth, db, repo, request, allowedOrigins });
      const payload = await readJsonBody(request);
      const result = await assetConfig.update({ repo, assetId, payload, user });
      return jsonResponse(200, result);
    }
    if (request.method === 'DELETE') {
      const user = await requireAssetWriteUser({ auth, db, repo, request, allowedOrigins });
      await assetConfig.delete({ repo, assetId, userId: user.userId });
      return jsonResponse(200, {});
    }
  }

  if (parts.length === 2 && parts[1] === 'content' && request.method === 'GET') {
    const viewer = await optionalAuthenticatedUser({ auth, db, request });
    const viewerId = viewer === null ? null : viewer.userId;
    const head = await repo.getAssetById(assetId, assetType);
    if (head === null || String(head.asset_type) !== assetType) {
      return jsonResponse(404, { message: 'not found' });
    }
    const result = await assetConfig.getContent({ repo, assetId, userId: viewerId, head });
    return jsonResponse(200, result, anonymousPublicResponseHeaders(request, viewer === null && String(head.visibility) === 'public'));
  }

  if (parts.length === 2 && parts[1] === 'download' && request.method === 'GET') {
    const viewer = await optionalAuthenticatedUser({ auth, db, request });
    const viewerId = viewer === null ? null : viewer.userId;
    const head = await repo.getAssetById(assetId, assetType);
    if (head === null || String(head.asset_type) !== assetType) {
      return jsonResponse(404, { message: 'not found' });
    }
    if (String(head.visibility) !== 'public' && String(head.owner_user_id) !== String(viewerId || '')) {
      return jsonResponse(404, { message: 'not found' });
    }
    const payload = await assetConfig.getContent({ repo, assetId, userId: viewerId, head });
    return assetDownloadResponse(payload, {
      head,
      versionNumber: payload.versionNumber,
      headers: anonymousPublicResponseHeaders(request, viewer === null && String(head.visibility) === 'public'),
    });
  }

  if (parts.length === 2 && parts[1] === 'visibility' && request.method === 'PUT') {
    const user = await requireAssetWriteUser({ auth, db, repo, request, allowedOrigins });
    const payload = await readJsonBody(request);
    const visibility = requireBodyString(payload.visibility, 'visibility is required');
    const result = await assetConfig.updateVisibility({ repo, assetId, visibility, versionNumber: payload.versionNumber, userId: user.userId });
    return jsonResponse(200, result);
  }

  if (parts.length === 2 && parts[1] === 'meta' && request.method === 'PATCH') {
    const user = await requireAssetWriteUser({ auth, db, repo, request, allowedOrigins });
    const payload = await readJsonBody(request);
    const result = await assetConfig.updateMeta({ repo, assetId, payload, user });
    return jsonResponse(200, result);
  }

  if (parts.length === 2 && parts[1] === 'versions' && request.method === 'GET') {
    const viewer = await optionalAuthenticatedUser({ auth, db, request });
    const result = await assetConfig.listVersions({ repo, assetId, userId: viewer === null ? null : viewer.userId });
    return jsonResponse(200, result, anonymousPublicResponseHeaders(request, viewer === null));
  }

  if (parts.length === 3 && parts[1] === 'versions' && request.method === 'GET') {
    const viewer = await optionalAuthenticatedUser({ auth, db, request });
    const versionNumber = parts[2];
    const result = await assetConfig.getVersion({ repo, assetId, versionNumber, userId: viewer === null ? null : viewer.userId });
    return jsonResponse(200, result, anonymousPublicResponseHeaders(request, viewer === null && String(result.visibility) === 'public'));
  }

  if (parts.length === 3 && parts[1] === 'versions' && request.method === 'PATCH') {
    const user = await requireAssetWriteUser({ auth, db, repo, request, allowedOrigins });
    const payload = await readJsonBody(request);
    const versionNumber = parts[2];
    const result = await assetConfig.updateVersionNote({
      repo,
      assetId,
      versionNumber,
      changeSummary: payload.changeSummary,
      userId: user.userId,
    });
    return jsonResponse(200, result);
  }

  if (parts.length === 4 && parts[1] === 'versions' && parts[3] === 'content' && request.method === 'GET') {
    const viewer = await optionalAuthenticatedUser({ auth, db, request });
    const viewerId = viewer === null ? null : viewer.userId;
    const versionNumber = parts[2];
    const head = await repo.getAssetById(assetId, assetType);
    if (head === null || String(head.asset_type) !== assetType) {
      return jsonResponse(404, { message: 'not found' });
    }
    const result = await assetConfig.getVersionContent({ repo, assetId, versionNumber, userId: viewerId, head });
    return jsonResponse(200, result, anonymousPublicResponseHeaders(request, viewer === null && String(head.visibility) === 'public'));
  }

  if (parts.length === 4 && parts[1] === 'versions' && parts[3] === 'download' && request.method === 'GET') {
    const viewer = await optionalAuthenticatedUser({ auth, db, request });
    const viewerId = viewer === null ? null : viewer.userId;
    const versionNumber = parts[2];
    const head = await repo.getAssetById(assetId, assetType);
    if (head === null || String(head.asset_type) !== assetType) {
      return jsonResponse(404, { message: 'not found' });
    }
    if (String(head.visibility) !== 'public' && String(head.owner_user_id) !== String(viewerId || '')) {
      return jsonResponse(404, { message: 'not found' });
    }
    const payload = await assetConfig.getVersionContent({ repo, assetId, versionNumber, userId: viewerId, head });
    return assetDownloadResponse(payload, {
      head,
      versionNumber: payload.versionNumber,
      headers: anonymousPublicResponseHeaders(request, viewer === null && String(head.visibility) === 'public'),
    });
  }

  if (parts.length === 2 && parts[1] === 'subscribers' && request.method === 'GET') {
    const user = await requireAuthenticatedUser({ auth, db, request, allowedOrigins });
    const result = await assetConfig.listSubscribers({ repo, assetId, userId: user.userId });
    return jsonResponse(200, result);
  }

  if (parts.length === 2 && parts[1] === 'subscribe') {
    const user = await requireAuthenticatedUser({ auth, db, request, allowedOrigins });
    if (request.method === 'POST') {
      const result = await assetConfig.subscribe({ repo, assetId, userId: user.userId });
      return jsonResponse(200, result);
    }
    if (request.method === 'DELETE') {
      const result = await assetConfig.unsubscribe({ repo, assetId, userId: user.userId });
      return jsonResponse(200, result);
    }
  }

  if (parts.length === 2 && parts[1] === 'fork' && request.method === 'POST') {
    const user = await requireAssetWriteUser({ auth, db, repo, request, allowedOrigins });
    const payload = await readJsonBody(request);
    const result = await assetConfig.fork({ repo, assetId, payload, user });
    return jsonResponse(200, result);
  }

  return jsonResponse(404, { message: 'not found' });
}

async function routeManagementRequest({ db, env, managementUser, auth, repo, request, url, allowedOrigins }) {
  if (isStateChangingRequestMethod(request.method)) {
    await enforceWorkerRateLimit({
      db,
      key: buildWorkerRateLimitKey({
        namespace: `management_${String(request.method).toLowerCase()}`,
        request,
      }),
      windowSeconds: 60,
      maxRequests: 20,
    });
  }
  if (request.method === 'GET' && url.pathname === `${MANAGEMENT_API_BASE_PATH}/users`) {
    const users = await repo.listUsers({
      query: url.searchParams.get('q') || '',
      cursor: url.searchParams.get('cursor') || '',
    });
    return jsonResponse(200, users);
  }

  if (request.method === 'POST' && url.pathname === `${MANAGEMENT_API_BASE_PATH}/users`) {
    const payload = await readJsonBody(request);
    const role = normalizeManagedUserRolePayload(payload);
    const name = requireUserProfileName(payload.name, {
      allowReserved: role === USER_ROLE_ADMIN,
    });
    const created = await auth.api.createUser({
      body: {
        email: requireBodyString(payload.email, 'email is required'),
        password: requireBodyString(payload.password, 'password is required'),
        name,
        role,
      },
      headers: request.headers,
    });
    const user = await repo.getUserByIdWithStats(String(created.user.id));
    return jsonResponse(200, user ?? toApiUser(toAppUser(created.user)));
  }

  if (request.method === 'GET' && url.pathname === `${MANAGEMENT_API_BASE_PATH}/site-settings`) {
    return jsonResponse(200, await repo.getSiteSettings());
  }

  if (request.method === 'GET' && url.pathname.startsWith(`${MANAGEMENT_API_BASE_PATH}/users/`)) {
    const parts = parseManagementUserPath(url.pathname);
    const userId = parts.userId;
    if (!userId) {
      return jsonResponse(404, { message: 'not found' });
    }
    if (parts.suffix.length === 0) {
      const user = await repo.getUserByIdWithStats(userId);
      if (user === null) {
        return jsonResponse(404, { message: 'user not found' });
      }
      return jsonResponse(200, user);
    }
  }

  if (request.method === 'PUT' && url.pathname.startsWith(`${MANAGEMENT_API_BASE_PATH}/users/`)) {
    const parts = parseManagementUserPath(url.pathname);
    if (!parts.userId || parts.suffix.length !== 0) {
      return jsonResponse(404, { message: 'not found' });
    }
    const currentTargetUser = await repo.getUserByIdWithStats(parts.userId);
    if (currentTargetUser === null) {
      return jsonResponse(404, { message: 'user not found' });
    }
    const payload = await readJsonBody(request);
    const data = {};
    const requestedRole = payload.role !== undefined || payload.isAdmin !== undefined || payload.canUpload !== undefined
      ? normalizeManagedUserRolePayload(payload)
      : currentTargetUser.role;
    if (payload.name !== undefined) {
      data.name = requireUserProfileName(payload.name, {
        currentName: currentTargetUser.name,
        allowReserved: requestedRole === USER_ROLE_ADMIN,
      });
    }
    if (Object.keys(data).length > 0) {
      await auth.api.adminUpdateUser({
        body: {
          userId: parts.userId,
          data,
        },
        headers: request.headers,
      });
    }
    if (payload.role !== undefined || payload.isAdmin !== undefined || payload.canUpload !== undefined) {
      if (parts.userId === managementUser.userId && requestedRole !== USER_ROLE_ADMIN) {
        throw new HttpError(400, 'management user cannot remove own admin role');
      }
      await auth.api.setRole({
        body: {
          userId: parts.userId,
          role: requestedRole,
        },
        headers: request.headers,
      });
    }
    if (payload.password !== undefined) {
      await auth.api.setUserPassword({
        body: {
          userId: parts.userId,
          newPassword: requireBodyString(payload.password, 'password is required'),
        },
        headers: request.headers,
      });
    }
    const updated = await repo.getUserByIdWithStats(parts.userId);
    if (updated === null) {
      return jsonResponse(404, { message: 'user not found' });
    }
    return jsonResponse(200, updated);
  }

  if (request.method === 'PUT' && url.pathname === `${MANAGEMENT_API_BASE_PATH}/site-settings`) {
    const payload = await readJsonBody(request);
    const updated = await repo.updateSiteSettings({
      allowUserRegistration: payload.allowUserRegistration,
      updatedByUserId: managementUser.userId,
    });
    invalidateAuthCache(env?.DB);
    return jsonResponse(200, updated);
  }

  if (request.method === 'POST' && url.pathname === `${MANAGEMENT_API_BASE_PATH}/assets/purge-all`) {
    const payload = await readJsonBody(request);
    const confirmationText = requireBodyString(payload.confirmationText, 'confirmationText is required');
    if (confirmationText !== PURGE_ALL_ASSETS_CONFIRMATION_TEXT) {
      throw new HttpError(400, `confirmationText must equal ${PURGE_ALL_ASSETS_CONFIRMATION_TEXT}`);
    }
    return jsonResponse(200, await repo.adminPurgeAllAssets());
  }

  if (request.method === 'DELETE' && url.pathname.startsWith(`${MANAGEMENT_API_BASE_PATH}/users/`)) {
    const parts = parseManagementUserPath(url.pathname);
    if (!parts.userId || parts.suffix.length !== 0) {
      return jsonResponse(404, { message: 'not found' });
    }
    if (parts.userId === managementUser.userId) {
      throw new HttpError(400, 'management user cannot delete self');
    }
    await assertUserHasNoAssets(repo, parts.userId);
    await auth.api.removeUser({
      body: { userId: parts.userId },
      headers: request.headers,
    });
    return jsonResponse(200, {});
  }

  if (request.method === 'GET' && url.pathname === `${MANAGEMENT_API_BASE_PATH}/components`) {
    const result = await repo.listManagedAssets({
      assetType: 'component',
      ownerUserId: url.searchParams.get('ownerUserId') || '',
      query: url.searchParams.get('q') || '',
      cursor: url.searchParams.get('cursor') || '',
    });
    return jsonResponse(200, result);
  }

  if (request.method === 'GET' && url.pathname === `${MANAGEMENT_API_BASE_PATH}/variants`) {
    const result = await repo.listManagedAssets({
      assetType: 'variant',
      ownerUserId: url.searchParams.get('ownerUserId') || '',
      query: url.searchParams.get('q') || '',
      cursor: url.searchParams.get('cursor') || '',
      kind: url.searchParams.get('kind') || '',
      baseNodeType: url.searchParams.get('baseNodeType') || '',
    });
    return jsonResponse(200, result);
  }

  if (request.method === 'GET' && url.pathname === `${MANAGEMENT_API_BASE_PATH}/modding-recipes`) {
    const result = await repo.listManagedAssets({
      assetType: 'modding_recipe',
      ownerUserId: url.searchParams.get('ownerUserId') || '',
      query: url.searchParams.get('q') || '',
      cursor: url.searchParams.get('cursor') || '',
    });
    return jsonResponse(200, result);
  }

  if (request.method === 'GET' && url.pathname.startsWith(`${MANAGEMENT_API_BASE_PATH}/components/`)) {
    const componentId = decodeSinglePathValue(url.pathname, `${MANAGEMENT_API_BASE_PATH}/components/`);
    if (!componentId) {
      return jsonResponse(404, { message: 'not found' });
    }
    const asset = await repo.getManagedAsset({
      assetId: componentId,
      assetTypeHint: 'component',
    });
    if (asset === null) {
      return jsonResponse(404, { message: 'asset not found' });
    }
    return jsonResponse(200, asset);
  }

  if (request.method === 'GET' && url.pathname.startsWith(`${MANAGEMENT_API_BASE_PATH}/variants/`)) {
    const variantId = decodeSinglePathValue(url.pathname, `${MANAGEMENT_API_BASE_PATH}/variants/`);
    if (!variantId) {
      return jsonResponse(404, { message: 'not found' });
    }
    const asset = await repo.getManagedAsset({
      assetId: variantId,
      assetTypeHint: 'variant',
    });
    if (asset === null) {
      return jsonResponse(404, { message: 'asset not found' });
    }
    return jsonResponse(200, asset);
  }

  if (request.method === 'GET' && url.pathname.startsWith(`${MANAGEMENT_API_BASE_PATH}/modding-recipes/`)) {
    const recipeId = decodeSinglePathValue(url.pathname, `${MANAGEMENT_API_BASE_PATH}/modding-recipes/`);
    if (!recipeId) {
      return jsonResponse(404, { message: 'not found' });
    }
    const asset = await repo.getManagedAsset({
      assetId: recipeId,
      assetTypeHint: 'modding_recipe',
    });
    if (asset === null) {
      return jsonResponse(404, { message: 'asset not found' });
    }
    return jsonResponse(200, asset);
  }

  if (request.method === 'PUT' && url.pathname.startsWith(`${MANAGEMENT_API_BASE_PATH}/components/`)) {
    const componentId = decodeSinglePathValue(url.pathname, `${MANAGEMENT_API_BASE_PATH}/components/`);
    if (!componentId) {
      return jsonResponse(404, { message: 'not found' });
    }
    const payload = await readJsonBody(request);
    if (payload.visibility !== undefined) {
      const updated = await repo.adminUpdateAssetVisibility({
        assetId: componentId,
        visibility: payload.visibility,
        assetTypeHint: 'component',
      });
      if (updated === null) {
        return jsonResponse(404, { message: 'asset not found' });
      }
      return jsonResponse(200, updated);
    }
    const current = await repo.getManagedAsset({ assetId: componentId, assetTypeHint: 'component' });
    if (current === null) {
      return jsonResponse(404, { message: 'asset not found' });
    }
    return jsonResponse(200, current);
  }

  if (request.method === 'PUT' && url.pathname.startsWith(`${MANAGEMENT_API_BASE_PATH}/variants/`)) {
    const variantId = decodeSinglePathValue(url.pathname, `${MANAGEMENT_API_BASE_PATH}/variants/`);
    if (!variantId) {
      return jsonResponse(404, { message: 'not found' });
    }
    const payload = await readJsonBody(request);
    if (payload.visibility !== undefined) {
      const updated = await repo.adminUpdateAssetVisibility({
        assetId: variantId,
        visibility: payload.visibility,
        assetTypeHint: 'variant',
      });
      if (updated === null) {
        return jsonResponse(404, { message: 'asset not found' });
      }
      return jsonResponse(200, updated);
    }
    const current = await repo.getManagedAsset({ assetId: variantId, assetTypeHint: 'variant' });
    if (current === null) {
      return jsonResponse(404, { message: 'asset not found' });
    }
    return jsonResponse(200, current);
  }

  if (request.method === 'PUT' && url.pathname.startsWith(`${MANAGEMENT_API_BASE_PATH}/modding-recipes/`)) {
    const recipeId = decodeSinglePathValue(url.pathname, `${MANAGEMENT_API_BASE_PATH}/modding-recipes/`);
    if (!recipeId) {
      return jsonResponse(404, { message: 'not found' });
    }
    const payload = await readJsonBody(request);
    if (payload.visibility !== undefined) {
      const updated = await repo.adminUpdateAssetVisibility({
        assetId: recipeId,
        visibility: payload.visibility,
        assetTypeHint: 'modding_recipe',
      });
      if (updated === null) {
        return jsonResponse(404, { message: 'asset not found' });
      }
      return jsonResponse(200, updated);
    }
    const current = await repo.getManagedAsset({ assetId: recipeId, assetTypeHint: 'modding_recipe' });
    if (current === null) {
      return jsonResponse(404, { message: 'asset not found' });
    }
    return jsonResponse(200, current);
  }

  if (request.method === 'DELETE' && url.pathname.startsWith(`${MANAGEMENT_API_BASE_PATH}/components/`)) {
    const componentId = decodeSinglePathValue(url.pathname, `${MANAGEMENT_API_BASE_PATH}/components/`);
    if (!componentId) {
      return jsonResponse(404, { message: 'not found' });
    }
    const deleted = await repo.adminDeleteAsset({ assetId: componentId, assetTypeHint: 'component' });
    if (!deleted) {
      return jsonResponse(404, { message: 'asset not found' });
    }
    return jsonResponse(200, {});
  }

  if (request.method === 'DELETE' && url.pathname.startsWith(`${MANAGEMENT_API_BASE_PATH}/variants/`)) {
    const variantId = decodeSinglePathValue(url.pathname, `${MANAGEMENT_API_BASE_PATH}/variants/`);
    if (!variantId) {
      return jsonResponse(404, { message: 'not found' });
    }
    const deleted = await repo.adminDeleteAsset({ assetId: variantId, assetTypeHint: 'variant' });
    if (!deleted) {
      return jsonResponse(404, { message: 'asset not found' });
    }
    return jsonResponse(200, {});
  }

  if (request.method === 'DELETE' && url.pathname.startsWith(`${MANAGEMENT_API_BASE_PATH}/modding-recipes/`)) {
    const recipeId = decodeSinglePathValue(url.pathname, `${MANAGEMENT_API_BASE_PATH}/modding-recipes/`);
    if (!recipeId) {
      return jsonResponse(404, { message: 'not found' });
    }
    const deleted = await repo.adminDeleteAsset({ assetId: recipeId, assetTypeHint: 'modding_recipe' });
    if (!deleted) {
      return jsonResponse(404, { message: 'asset not found' });
    }
    return jsonResponse(200, {});
  }

  return jsonResponse(404, { message: 'not found' });
}

async function routeDesktopAuthorizeGet({ auth, db, env, request, siteSettings }) {
  const requestUrl = new URL(request.url);
  const authorizeRequest = parseDesktopAuthorizeRequestFromUrl(requestUrl);
  assertAllowedDesktopClientId(authorizeRequest.clientId);
  await enforceWorkerRateLimit({
    db,
    key: buildWorkerRateLimitKey({ namespace: 'desktop_authorize_get', request }),
    windowSeconds: 60,
    maxRequests: 30,
  });
  const socialProvider = desktopAuthorizeSocialProvider(requestUrl);
  const errorMessage = desktopAuthorizeErrorMessage(requestUrl);
  const allowRegistration = Boolean(siteSettings?.allowUserRegistration);
  const allowDesktopGoogle = allowRegistration && hasGoogleProvider(env);
  const session = await auth.api.getSession({
    headers: request.headers,
  });
  const currentUser = session === null || !session.user ? null : toAppUser(session.user);
  const pageCsrfToken = generateId(32);
  if (socialProvider) {
    if (!allowRegistration) {
      return desktopAuthorizePageResponse(
        400,
        buildDesktopAuthorizeHtml({
          request: authorizeRequest,
          allowGoogle: false,
          allowRegistration: false,
          currentUser,
          csrfToken: pageCsrfToken,
          errorMessage: 'Google sign-in is unavailable while registration is disabled.',
          email: '',
        }),
        request,
        pageCsrfToken,
      );
    }
    if (socialProvider !== 'google') {
      throw new HttpError(400, `unsupported social provider: ${socialProvider}`);
    }
    if (!allowDesktopGoogle) {
      throw new HttpError(400, 'Google sign-in is not configured');
    }
    return startDesktopSocialSignIn({
      auth,
      request,
      providerId: socialProvider,
      callbackURL: desktopAuthorizeResumeUrl(requestUrl),
      errorCallbackURL: desktopAuthorizeErrorCallbackUrl(requestUrl),
    });
  }
  return desktopAuthorizePageResponse(
    200,
    buildDesktopAuthorizeHtml({
      request: authorizeRequest,
      allowGoogle: allowDesktopGoogle,
      allowRegistration,
      currentUser,
      csrfToken: pageCsrfToken,
      errorMessage,
      email: '',
    }),
    request,
    pageCsrfToken,
  );
}

async function routeDesktopAuthorizePost({ auth, db, env, request, siteSettings }) {
  assertAllowedCookieStateChange(calculateAllowedOrigins(env), request);
  await enforceWorkerRateLimit({
    db,
    key: buildWorkerRateLimitKey({ namespace: 'desktop_authorize_post', request }),
    windowSeconds: 60,
    maxRequests: 20,
  });
  const form = await request.formData();
  const authorizeRequest = parseDesktopAuthorizeRequestFromForm(form);
  assertAllowedDesktopClientId(authorizeRequest.clientId);
  validateDesktopAuthorizeCsrf({ request, form });
  const email = String(form.get('email') || '').trim();
  const password = String(form.get('password') || '');
  const allowRegistration = Boolean(siteSettings?.allowUserRegistration);
  const allowGoogle = allowRegistration && hasGoogleProvider(env);
  const existingSession = await auth.api.getSession({
    headers: request.headers,
  });
  try {
    let userId = '';
    if (existingSession !== null && existingSession.user && !email && !password) {
      userId = String(existingSession.user.id || '').trim();
    } else {
      const normalizedEmail = requireFormString(email, 'email is required');
      const normalizedPassword = requireFormString(password, 'password is required');
      const sessionCookie = await signInDesktopBrowserUser({
        auth,
        authorizeUrl: request.url,
        request,
        email: normalizedEmail,
        password: normalizedPassword,
      });
      const sessionHeaders = new Headers();
      sessionHeaders.set('cookie', sessionCookie);
      const signedInSession = await auth.api.getSession({
        headers: sessionHeaders,
      });
      if (signedInSession === null || !signedInSession.user) {
        throw new HttpError(401, 'browser session is no longer valid');
      }
      userId = String(signedInSession.user.id || '').trim();
    }
    if (!userId) {
      throw new HttpError(401, 'browser session is no longer valid');
    }
    return redirectWithDesktopAuthorizationCode({
      db,
      desktopRequest: authorizeRequest,
      userId,
    });
  } catch (error) {
    if (error instanceof HttpError) {
      const pageCsrfToken = generateId(32);
      return desktopAuthorizePageResponse(
        error.status,
        buildDesktopAuthorizeHtml({
          request: authorizeRequest,
          allowGoogle,
          allowRegistration,
          currentUser: existingSession === null || !existingSession.user ? null : toAppUser(existingSession.user),
          csrfToken: pageCsrfToken,
          errorMessage: error.message,
          email,
        }),
        request,
        pageCsrfToken,
      );
    }
    throw error;
  }
}

async function routeDesktopTokenPost({ db, request }) {
  await enforceWorkerRateLimit({
    db,
    key: buildWorkerRateLimitKey({ namespace: 'desktop_token', request }),
    windowSeconds: 60,
    maxRequests: 20,
  });
  const payload = await readJsonBody(request);
  const code = requireBodyString(payload.code, 'code is required');
  const clientId = requireBodyString(payload.clientId ?? payload.client_id, 'clientId is required');
  const redirectUri = requireBodyString(payload.redirectUri ?? payload.redirect_uri, 'redirectUri is required');
  const codeVerifier = requireBodyString(payload.codeVerifier ?? payload.code_verifier, 'codeVerifier is required');
  assertAllowedDesktopClientId(clientId);
  requireLoopbackRedirectUri(redirectUri);
  const record = await loadDesktopAuthorizationCodeRecord(db, code);
  if (record === null) {
    throw new HttpError(400, 'authorization code is invalid or has expired');
  }
  if (record.usedAt !== null) {
    throw new HttpError(400, 'authorization code has already been used');
  }
  if (record.expiresAt <= Date.now()) {
    throw new HttpError(400, 'authorization code has expired');
  }
  if (record.clientId !== clientId) {
    throw new HttpError(400, 'clientId does not match the authorization request');
  }
  if (record.redirectUri !== redirectUri) {
    throw new HttpError(400, 'redirectUri does not match the authorization request');
  }
  if (record.codeChallengeMethod !== 'S256') {
    throw new HttpError(400, 'unsupported code_challenge_method');
  }
  const computedChallenge = await computePkceCodeChallenge(codeVerifier);
  if (computedChallenge !== record.codeChallenge) {
    throw new HttpError(400, 'codeVerifier is invalid');
  }
  if (record.user === null) {
    throw new HttpError(401, 'browser session is no longer valid');
  }
  const usedAt = Date.now();
  const updateResult = await db.prepare(
    'UPDATE desktop_authorization_codes SET used_at = ? WHERE code = ? AND used_at IS NULL',
  )
    .bind(usedAt, code)
    .run();
  if (Number(updateResult.meta?.changes || 0) < 1) {
    throw new HttpError(400, 'authorization code has already been used');
  }
  const desktopAuth = await issueDesktopTokenPair({
    db,
    user: record.user,
  });
  return jsonResponse(200, desktopAuthResponsePayload(desktopAuth));
}

async function routeDesktopSessionPost({ auth, db, request, allowedOrigins }) {
  assertAllowedCookieStateChange(allowedOrigins, request);
  await enforceWorkerRateLimit({
    db,
    key: buildWorkerRateLimitKey({ namespace: 'desktop_session_exchange', request }),
    windowSeconds: 60,
    maxRequests: 12,
  });
  const session = await auth.api.getSession({
    headers: request.headers,
  });
  if (session === null || !session.user) {
    throw new HttpError(401, 'authentication required');
  }
  const user = await readAppUserById(db, String(session.user.id || ''));
  if (user === null) {
    throw new HttpError(401, 'authentication required');
  }
  return jsonResponse(200, desktopAuthResponsePayload(await issueDesktopTokenPair({ db, user })));
}

async function routeDesktopRefreshPost({ db, request }) {
  const payload = await readJsonBody(request);
  const refreshToken = requireBodyString(payload.refreshToken ?? payload.refresh_token, 'refreshToken is required');
  await enforceWorkerRateLimit({
    db,
    key: `worker:desktop_refresh:${await hashOpaqueToken(refreshToken)}`,
    windowSeconds: 60,
    maxRequests: 20,
  });
  return jsonResponse(200, desktopAuthResponsePayload(await refreshDesktopTokenPair({
    db,
    refreshToken,
  })));
}

async function routeDesktopRevokePost({ db, request }) {
  const payload = await readJsonBody(request);
  const refreshToken = requireBodyString(payload.refreshToken ?? payload.refresh_token, 'refreshToken is required');
  await revokeDesktopRefreshToken({
    db,
    refreshToken,
  });
  return jsonResponse(200, { revoked: true });
}

function createAuth(env, { siteSettings, baseURL, trustedOrigins }) {
  const db = drizzle(env.DB);
  const socialProviders = {};
  const plugins = [
    admin({
      defaultRole: USER_ROLE_USER,
      adminRoles: [USER_ROLE_ADMIN],
    }),
  ];
  if (turnstileSecretKey(env)) {
    plugins.push(captcha({
      provider: 'cloudflare-turnstile',
      secretKey: turnstileSecretKey(env),
      endpoints: ['/sign-up/email', '/request-password-reset'],
    }));
  }
  if (hasGoogleProvider(env)) {
    socialProviders.google = {
      clientId: String(env.GOOGLE_CLIENT_ID || '').trim(),
      clientSecret: String(env.GOOGLE_CLIENT_SECRET || '').trim(),
      disableImplicitSignUp: !siteSettings.allowUserRegistration,
    };
  }

  return betterAuth({
    secret: getAuthSecret(env),
    baseURL,
    basePath: AUTH_BASE_PATH,
    trustedOrigins,
    database: drizzleAdapter(db, {
      provider: 'sqlite',
      schema: authSchema,
      usePlural: false,
      transaction: false,
    }),
    databaseHooks: {
      user: {
        create: {
          async before(user) {
            assertReservedNameAllowedForRole({
              name: user?.name,
              role: user?.role,
            });
          },
        },
      },
    },
    emailAndPassword: {
      enabled: true,
      autoSignIn: false,
      requireEmailVerification: true,
      revokeSessionsOnPasswordReset: true,
      password: {
        hash: hashAuthPassword,
        verify: verifyAuthPassword,
      },
      resetPasswordTokenExpiresIn: secondsFromEnv(env.PASSWORD_RESET_TOKEN_TTL_SECONDS, 1800),
      sendResetPassword: async ({ user, token, url }) => {
        const appUser = toAppUser(user);
        const resetUrl = resolveAuthActionUrl({
          preferredUrl: url,
          fallbackUrl: buildResetPasswordUrl({ env, baseURL, token }),
        });
        await sendResetPasswordMessage({
          env,
          toEmail: appUser.email,
          recipientName: appUser.name || appUser.email,
          resetUrl,
        });
      },
    },
    emailVerification: {
      sendOnSignIn: true,
      sendOnSignUp: true,
      autoSignInAfterVerification: false,
      expiresIn: secondsFromEnv(env.EMAIL_VERIFY_TOKEN_TTL_SECONDS, 1800),
      sendVerificationEmail: async ({ user, token, url }) => {
        const appUser = toAppUser(user);
        const verificationUrl = resolveAuthActionUrl({
          preferredUrl: url,
          fallbackUrl: buildVerifyEmailUrl({ env, baseURL, token }),
        });
        await sendVerifyEmailMessage({
          env,
          toEmail: appUser.email,
          recipientName: appUser.name || appUser.email,
          verificationUrl,
        });
      },
    },
    user: {
      changeEmail: {
        enabled: true,
      },
    },
    socialProviders,
    plugins,
    rateLimit: {
      enabled: true,
      storage: 'database',
      window: 60,
      max: 60,
      customRules: {
        '/sign-in/email': { window: 60, max: 6 },
        '/sign-up/email': { window: 600, max: 5 },
        '/request-password-reset': { window: 900, max: 4 },
        '/send-verification-email': { window: 900, max: 4 },
        '/verify-email': { window: 300, max: 10 },
      },
    },
  });
}

async function getOrCreateAuth(env, request, siteSettings = null) {
  const resolvedSiteSettings = siteSettings ?? await readSiteSettings(env.DB);
  const requestUrl = new URL(request.url);
  const baseURL = resolveAuthBaseUrl(env, requestUrl);
  const trustedOrigins = resolveTrustedOrigins(env, requestUrl, baseURL);
  const cacheKey = buildAuthCacheKey({
    allowUserRegistration: resolvedSiteSettings.allowUserRegistration,
    baseURL,
    trustedOrigins,
  });
  const dbCache = getOrCreateDbCache(authCacheByDb, env.DB);
  const cached = dbCache.get(cacheKey);
  if (cached !== undefined) {
    return cached;
  }
  const auth = createAuth(env, {
    siteSettings: resolvedSiteSettings,
    baseURL,
    trustedOrigins,
  });
  dbCache.set(cacheKey, auth);
  return auth;
}

async function ensureBootstrapAdmin({ env }) {
  const config = readBootstrapAdminConfig(env);
  if (config === null) {
    return;
  }

  const db = env?.DB;
  if (db && typeof db === 'object') {
    const pending = bootstrapAdminInitByDb.get(db);
    if (pending) {
      await pending;
      return;
    }

    const initPromise = ensureBootstrapAdminOnce(env, config);
    bootstrapAdminInitByDb.set(db, initPromise);
    try {
      await initPromise;
    } catch (error) {
      bootstrapAdminInitByDb.delete(db);
      throw error;
    }
    return;
  }

  await ensureBootstrapAdminOnce(env, config);
}

function readBootstrapAdminConfig(env) {
  const legacyUsername = String(env.BOOTSTRAP_ADMIN_USERNAME || '').trim();
  const name = String(
    env.BOOTSTRAP_ADMIN_NAME
      || env.BOOTSTRAP_ADMIN_DISPLAY_NAME
      || legacyUsername
      || 'Administrator',
  ).trim();
  const password = String(env.BOOTSTRAP_ADMIN_PASSWORD || '').trim();
  const email = String(
    env.BOOTSTRAP_ADMIN_EMAIL
      || (legacyUsername ? `${legacyUsername.toLowerCase()}@local.invalid` : ''),
  ).trim().toLowerCase();
  if (!name || !password || !email) {
    return null;
  }
  return {
    name,
    password,
    email,
  };
}

async function ensureBootstrapAdminOnce(env, config) {
  const configFingerprint = await computeBootstrapAdminFingerprint(env, config);
  const syncedState = await readBootstrapAdminState(env.DB);
  if (syncedState !== null && syncedState.configFingerprint === configFingerprint) {
    const syncedAdmin = await readBootstrapAdminCredentialAccount(env.DB, syncedState.userId);
    if (bootstrapAdminMatchesConfig(syncedAdmin, config)) {
      return;
    }
  }

  const existing = await env.DB.prepare(
    `SELECT
       u.id,
       a.id AS credential_account_id
     FROM user u
     LEFT JOIN account a
       ON a.userId = u.id AND a.providerId = 'credential'
     WHERE u.email = ? OR u.name = ?
     LIMIT 1`,
  )
    .bind(config.email, config.name)
    .first();
  const timestamp = Date.now();

  if (existing === null) {
    const passwordHash = await hashAuthPassword(config.password);
    const userId = generateId();
    await env.DB.prepare(
      `INSERT INTO user
         ("id", "name", "email", "emailVerified", "image", "createdAt", "updatedAt", "role", "banned", "banReason", "banExpires")
       VALUES (?, ?, ?, 1, NULL, ?, ?, 'admin', 0, NULL, NULL)`,
    )
      .bind(userId, config.name, config.email, timestamp, timestamp)
      .run();
    await env.DB.prepare(
      `INSERT INTO account
         ("id", "accountId", "providerId", "userId", "accessToken", "refreshToken", "idToken", "accessTokenExpiresAt", "refreshTokenExpiresAt", "scope", "password", "createdAt", "updatedAt")
       VALUES (?, ?, 'credential', ?, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, ?)`,
    )
      .bind(generateId(), userId, userId, passwordHash, timestamp, timestamp)
      .run();
    await upsertBootstrapAdminState(env.DB, {
      configFingerprint,
      userId,
    });
    return;
  }

  const userId = String(existing.id);
  await env.DB.prepare(
    `UPDATE user
     SET email = ?,
         role = 'admin',
         emailVerified = 1,
         name = ?,
         updatedAt = ?
     WHERE id = ?`,
  )
    .bind(config.email, config.name, timestamp, userId)
    .run();

  if (existing.credential_account_id === null || existing.credential_account_id === undefined) {
    const passwordHash = await hashAuthPassword(config.password);
    await env.DB.prepare(
      `INSERT INTO account
         ("id", "accountId", "providerId", "userId", "accessToken", "refreshToken", "idToken", "accessTokenExpiresAt", "refreshTokenExpiresAt", "scope", "password", "createdAt", "updatedAt")
       VALUES (?, ?, 'credential', ?, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, ?)`,
    )
      .bind(generateId(), userId, userId, passwordHash, timestamp, timestamp)
      .run();
    await upsertBootstrapAdminState(env.DB, {
      configFingerprint,
      userId,
    });
    return;
  }

  if (syncedState === null || syncedState.configFingerprint !== configFingerprint) {
    const passwordHash = await hashAuthPassword(config.password);
    await env.DB.prepare(
      `UPDATE account
       SET accountId = ?,
           password = ?,
           updatedAt = ?
       WHERE id = ?`,
    )
      .bind(userId, passwordHash, timestamp, String(existing.credential_account_id))
      .run();
  }

  await upsertBootstrapAdminState(env.DB, {
    configFingerprint,
    userId,
  });
}

async function requireAuthenticatedUser({ auth, db, request, allowedOrigins = null }) {
  const authState = await readAuthenticatedRequestState({ auth, db, request });
  if (authState === null) {
    throw new HttpError(401, 'authentication required');
  }
  if (authState.authType === 'cookie' && isStateChangingRequestMethod(request.method)) {
    assertAllowedCookieStateChange(allowedOrigins, request);
  }
  return authState.user;
}

async function optionalAuthenticatedUser({ auth, db, request }) {
  const authState = await readAuthenticatedRequestState({ auth, db, request });
  return authState === null ? null : authState.user;
}

async function requireManagementUser({ auth, db, request, allowedOrigins = null }) {
  const user = await requireAuthenticatedUser({ auth, db, request, allowedOrigins });
  if (!user.isAdmin) {
    throw new HttpError(403, 'management access required');
  }
  return user;
}

async function requireAssetWriteUser({ auth, db, repo, request, allowedOrigins = null }) {
  const user = await requireAuthenticatedUser({ auth, db, request, allowedOrigins });
  const latestUser = await repo.getUserById(user.userId);
  if (latestUser === null) {
    throw new HttpError(401, 'authentication required');
  }
  if (latestUser.role === USER_ROLE_READONLY) {
    throw new HttpError(403, 'upload permission required');
  }
  return {
    ...user,
    role: latestUser.role,
    canUpload: latestUser.canUpload,
  };
}

async function readAuthenticatedRequestState({ auth, db, request }) {
  const bearerToken = readBearerTokenFromRequest(request);
  if (bearerToken !== null) {
    const user = await readDesktopAccessTokenUser({
      db,
      accessToken: bearerToken,
    });
    if (user === null) {
      throw new HttpError(401, 'authentication required');
    }
    return {
      authType: 'bearer',
      user,
    };
  }
  const cookieHeader = String(request.headers.get('cookie') || '').trim();
  if (!cookieHeader) {
    return null;
  }
  const session = await auth.api.getSession({
    headers: request.headers,
  });
  if (session === null || !session.user) {
    return null;
  }
  return {
    authType: 'cookie',
    user: toAppUser(session.user),
  };
}

function readBearerTokenFromRequest(request) {
  const headerValue = String(request.headers.get('authorization') || '').trim();
  if (!headerValue) {
    return null;
  }
  const match = /^Bearer\s+(.+)$/i.exec(headerValue);
  if (!match) {
    throw new HttpError(401, 'Authorization header must use Bearer token');
  }
  const token = String(match[1] || '').trim();
  if (!token) {
    throw new HttpError(401, 'Authorization header must use Bearer token');
  }
  return token;
}

async function assertUserHasNoAssets(repo, userId) {
  const ownsAssets = await repo.hasAssets(userId);
  if (ownsAssets) {
    throw new HttpError(409, 'cannot delete user with existing assets');
  }
}

function toApiUser(user) {
  return {
    userId: user.userId,
    name: user.name,
    email: user.email,
    emailVerified: user.emailVerified,
    isAdmin: user.isAdmin,
    role: user.role,
    canUpload: user.canUpload,
  };
}

function toAppUser(user) {
  const email = stringOrDefault(user.email, '');
  const name = stringOrDefault(user.name, email || String(user.id || ''));
  return {
    userId: String(user.id),
    name,
    email,
    emailVerified: Boolean(user.emailVerified),
    role: normalizeUserRole(user.role),
    isAdmin: normalizeUserRole(user.role) === USER_ROLE_ADMIN,
    canUpload: normalizeUserRole(user.role) !== USER_ROLE_READONLY,
  };
}

function validateEnv(env) {
  if (!env || typeof env !== 'object' || env.DB === undefined || env.DB === null) {
    throw new HttpError(500, 'DB binding is not configured');
  }
  if (!getAuthSecret(env)) {
    throw new HttpError(500, 'BETTER_AUTH_SECRET is not configured');
  }
  if (!String(env.AUTH_BASE_URL || '').trim()) {
    throw new HttpError(500, 'AUTH_BASE_URL is not configured');
  }
}

function getAuthSecret(env) {
  return String(env.BETTER_AUTH_SECRET || '').trim();
}

function hasGoogleProvider(env) {
  return Boolean(String(env.GOOGLE_CLIENT_ID || '').trim() && String(env.GOOGLE_CLIENT_SECRET || '').trim());
}

function turnstileSiteKey(env) {
  const siteKey = String(
    env.TURNSTILE_SITE_KEY
      || env.CLOUDFLARE_TURNSTILE_SITE_KEY
      || '',
  ).trim();
  return siteKey || null;
}

function turnstileSecretKey(env) {
  return String(
    env.TURNSTILE_SECRET_KEY
      || env.CLOUDFLARE_TURNSTILE_SECRET_KEY
      || '',
  ).trim();
}

function parseDesktopAuthorizeRequestFromUrl(url) {
  const clientId = requireQueryString(url.searchParams.get('client_id') || url.searchParams.get('clientId'), 'client_id is required');
  const redirectUri = requireLoopbackRedirectUri(
    requireQueryString(url.searchParams.get('redirect_uri') || url.searchParams.get('redirectUri'), 'redirect_uri is required'),
  );
  const state = requireQueryString(url.searchParams.get('state'), 'state is required');
  const codeChallenge = requireQueryString(
    url.searchParams.get('code_challenge') || url.searchParams.get('codeChallenge'),
    'code_challenge is required',
  );
  const codeChallengeMethod = requireQueryString(
    url.searchParams.get('code_challenge_method') || url.searchParams.get('codeChallengeMethod'),
    'code_challenge_method is required',
  );
  if (codeChallengeMethod !== 'S256') {
    throw new HttpError(400, 'code_challenge_method must be S256');
  }
  return {
    clientId,
    redirectUri,
    state,
    codeChallenge,
    codeChallengeMethod,
  };
}

function parseDesktopAuthorizeRequestFromForm(form) {
  const clientId = requireFormString(form.get('client_id') || form.get('clientId'), 'client_id is required');
  const redirectUri = requireLoopbackRedirectUri(
    requireFormString(form.get('redirect_uri') || form.get('redirectUri'), 'redirect_uri is required'),
  );
  const state = requireFormString(form.get('state'), 'state is required');
  const codeChallenge = requireFormString(
    form.get('code_challenge') || form.get('codeChallenge'),
    'code_challenge is required',
  );
  const codeChallengeMethod = requireFormString(
    form.get('code_challenge_method') || form.get('codeChallengeMethod'),
    'code_challenge_method is required',
  );
  if (codeChallengeMethod !== 'S256') {
    throw new HttpError(400, 'code_challenge_method must be S256');
  }
  return {
    clientId,
    redirectUri,
    state,
    codeChallenge,
    codeChallengeMethod,
  };
}

function requireLoopbackRedirectUri(value) {
  const redirectUri = String(value || '').trim();
  if (!redirectUri) {
    throw new HttpError(400, 'redirect_uri is required');
  }
  let parsed;
  try {
    parsed = new URL(redirectUri);
  } catch (error) {
    throw new HttpError(400, 'redirect_uri must be a valid URL');
  }
  const hostname = String(parsed.hostname || '').trim().toLowerCase();
  const allowedHost = hostname === '127.0.0.1' || hostname === 'localhost' || hostname === '::1';
  if (parsed.protocol !== 'http:' || !allowedHost) {
    throw new HttpError(400, 'redirect_uri must use http://127.0.0.1, http://localhost, or http://[::1]');
  }
  return parsed.toString();
}

function requireFormString(value, message) {
  if (typeof value !== 'string') {
    throw new HttpError(400, message);
  }
  const text = value.trim();
  if (!text) {
    throw new HttpError(400, message);
  }
  return text;
}

async function signInDesktopBrowserUser({ auth, authorizeUrl, request, email, password }) {
  const headers = new Headers();
  headers.set('Accept', 'application/json');
  headers.set('Content-Type', 'application/json');
  headers.set('Origin', new URL(request.url).origin);
  headers.set('Referer', authorizeUrl);
  copyHeaderIfPresent(request.headers, headers, 'User-Agent');
  copyHeaderIfPresent(request.headers, headers, 'Accept-Language');
  const signInRequest = new Request(new URL(`${AUTH_BASE_PATH}/sign-in/email`, request.url), {
    method: 'POST',
    headers,
    body: JSON.stringify({
      email,
      password,
    }),
  });
  const signInResponse = await auth.handler(signInRequest);
  const responseText = await signInResponse.text();
  if (!signInResponse.ok) {
    let message = `sign-in failed with HTTP ${signInResponse.status}`;
    if (responseText) {
      try {
        const parsed = JSON.parse(responseText);
        message = errorMessageFromPayload(parsed) || message;
      } catch {
        message = responseText;
      }
    }
    throw new HttpError(signInResponse.status, message);
  }
  const sessionCookie = extractCookieFromSetCookie(signInResponse.headers.get('set-cookie') || '');
  if (!sessionCookie) {
    throw new HttpError(500, 'sign-in succeeded but no session cookie was returned');
  }
  return sessionCookie;
}

async function startDesktopSocialSignIn({ auth, request, providerId, callbackURL, errorCallbackURL }) {
  const headers = new Headers();
  headers.set('Accept', 'application/json');
  headers.set('Content-Type', 'application/json');
  headers.set('Origin', new URL(request.url).origin);
  headers.set('Referer', request.url);
  copyHeaderIfPresent(request.headers, headers, 'User-Agent');
  copyHeaderIfPresent(request.headers, headers, 'Accept-Language');
  const socialRequest = new Request(new URL(`${AUTH_BASE_PATH}/sign-in/social`, request.url), {
    method: 'POST',
    headers,
    body: JSON.stringify({
      provider: providerId,
      callbackURL,
      errorCallbackURL,
    }),
  });
  const socialResponse = await auth.handler(socialRequest);
  const responseText = await socialResponse.text();
  if (!socialResponse.ok) {
    let message = `social sign-in failed with HTTP ${socialResponse.status}`;
    if (responseText) {
      try {
        const parsed = JSON.parse(responseText);
        message = errorMessageFromPayload(parsed) || message;
      } catch {
        message = responseText;
      }
    }
    throw new HttpError(socialResponse.status, message);
  }
  let authorizationUrl = String(socialResponse.headers.get('location') || '').trim();
  if (!authorizationUrl) {
    try {
      const payload = JSON.parse(responseText);
      if (payload && typeof payload.url === 'string') {
        authorizationUrl = String(payload.url).trim();
      }
    } catch {
      authorizationUrl = '';
    }
  }
  if (!authorizationUrl) {
    throw new HttpError(500, 'social sign-in did not return an authorization URL');
  }
  const redirectHeaders = new Headers();
  const setCookie = socialResponse.headers.get('set-cookie');
  if (setCookie) {
    redirectHeaders.set('set-cookie', setCookie);
  }
  redirectHeaders.set('Cache-Control', 'no-store');
  redirectHeaders.set('Location', authorizationUrl);
  return new Response(null, {
    status: 302,
    headers: redirectHeaders,
  });
}

async function redirectWithDesktopAuthorizationCode({ db, desktopRequest, userId }) {
  await purgeExpiredDesktopAuthorizationCodesIfDue(db);
  const now = Date.now();
  const code = generateId(48);
  const expiresAt = now + DESKTOP_AUTHORIZATION_CODE_TTL_SECONDS * 1000;
  await db.prepare(
    `INSERT INTO desktop_authorization_codes
      (code, user_id, client_id, redirect_uri, code_challenge, code_challenge_method, created_at, expires_at, used_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)`,
  )
    .bind(
      code,
      String(userId),
      desktopRequest.clientId,
      desktopRequest.redirectUri,
      desktopRequest.codeChallenge,
      desktopRequest.codeChallengeMethod,
      now,
      expiresAt,
    )
    .run();
  const redirectUrl = new URL(desktopRequest.redirectUri);
  redirectUrl.searchParams.set('code', code);
  redirectUrl.searchParams.set('state', desktopRequest.state);
  return Response.redirect(redirectUrl.toString(), 302);
}

async function loadDesktopAuthorizationCodeRecord(db, code) {
  await purgeExpiredDesktopAuthorizationCodesIfDue(db);
  const row = await db.prepare(
    `SELECT
      dac.code,
      dac.user_id,
      dac.client_id,
      dac.redirect_uri,
      dac.code_challenge,
      dac.code_challenge_method,
      dac.created_at,
      dac.expires_at,
      dac.used_at,
      u.id,
      u.name,
      u.email,
      u.emailVerified,
      u.role
    FROM desktop_authorization_codes dac
    LEFT JOIN user u ON u.id = dac.user_id
    WHERE dac.code = ?`,
  )
    .bind(code)
    .first();
  if (row === null) {
    return null;
  }
  return {
    code: String(row.code || ''),
    userId: String(row.user_id || ''),
    clientId: String(row.client_id || ''),
    redirectUri: String(row.redirect_uri || ''),
    codeChallenge: String(row.code_challenge || ''),
    codeChallengeMethod: String(row.code_challenge_method || ''),
    createdAt: Number(row.created_at || 0),
    expiresAt: Number(row.expires_at || 0),
    usedAt: row.used_at === null || row.used_at === undefined ? null : Number(row.used_at),
    user: row.id === null || row.id === undefined ? null : appUserFromDbRow(row),
  };
}

async function purgeExpiredDesktopAuthorizationCodes(db) {
  await db.prepare('DELETE FROM desktop_authorization_codes WHERE expires_at < ?')
    .bind(Date.now())
    .run();
}

async function purgeExpiredDesktopAuthorizationCodesIfDue(db) {
  if (!shouldRunDbIntervalTask(lastDesktopAuthorizationCodePurgeByDb, db, DESKTOP_REQUEST_PURGE_INTERVAL_MS)) {
    return;
  }
  await purgeExpiredDesktopAuthorizationCodes(db);
}

function assertAllowedDesktopClientId(clientId) {
  const normalizedClientId = String(clientId || '').trim();
  if (!DESKTOP_AUTH_ALLOWED_CLIENT_IDS.has(normalizedClientId)) {
    throw new HttpError(400, 'clientId is not allowed');
  }
}

function buildWorkerRateLimitKey({ namespace, request, identity = '' }) {
  const normalizedNamespace = String(namespace || '').trim() || 'unknown';
  const normalizedIdentity = String(identity || '').trim();
  if (normalizedIdentity) {
    return `worker:${normalizedNamespace}:${normalizedIdentity}`;
  }
  const forwardedIp = String(
    request.headers.get('cf-connecting-ip')
      || request.headers.get('x-forwarded-for')
      || '',
  )
    .split(',')[0]
    .trim();
  if (forwardedIp) {
    return `worker:${normalizedNamespace}:${forwardedIp}`;
  }
  const userAgent = String(request.headers.get('user-agent') || '').trim();
  if (userAgent) {
    return `worker:${normalizedNamespace}:ua:${userAgent.slice(0, 120)}`;
  }
  return `worker:${normalizedNamespace}:anonymous`;
}

async function enforceWorkerRateLimit({ db, key, windowSeconds, maxRequests }) {
  const normalizedKey = String(key || '').trim();
  if (!normalizedKey) {
    return;
  }
  const now = Date.now();
  const windowStart = now - (Number(windowSeconds) * 1000);
  const existing = await db.prepare(
    'SELECT count, lastRequest FROM rateLimit WHERE key = ?',
  )
    .bind(normalizedKey)
    .first();
  if (existing === null || Number(existing.lastRequest || 0) < windowStart) {
    await db.prepare(
      `INSERT INTO rateLimit (key, count, lastRequest)
       VALUES (?, 1, ?)
       ON CONFLICT(key) DO UPDATE SET
         count = 1,
         lastRequest = excluded.lastRequest`,
    )
      .bind(normalizedKey, now)
      .run();
    return;
  }
  const currentCount = Number(existing.count || 0);
  if (currentCount >= Number(maxRequests)) {
    throw new HttpError(429, 'rate limit exceeded');
  }
  await db.prepare(
    `UPDATE rateLimit
     SET count = ?, lastRequest = ?
     WHERE key = ?`,
  )
    .bind(currentCount + 1, now, normalizedKey)
    .run();
}

function desktopAuthorizePageResponse(status, html, request, csrfToken) {
  const headers = new Headers({
    'Content-Type': 'text/html; charset=utf-8',
    'Cache-Control': 'no-store',
  });
  headers.append('Set-Cookie', buildSetCookieHeader({
    name: DESKTOP_AUTH_CONFIRM_CSRF_COOKIE,
    value: csrfToken,
    request,
    path: `${DESKTOP_AUTH_BASE_PATH}/authorize`,
    maxAgeSeconds: DESKTOP_AUTHORIZATION_CODE_TTL_SECONDS,
  }));
  return new Response(html, {
    status,
    headers,
  });
}

function buildSetCookieHeader({ name, value, request, path = '/', maxAgeSeconds = 0 }) {
  const parts = [
    `${String(name || '').trim()}=${encodeURIComponent(String(value || '').trim())}`,
    `Path=${path}`,
    `Max-Age=${Math.max(0, Number(maxAgeSeconds) || 0)}`,
    'HttpOnly',
    'SameSite=Lax',
  ];
  if (new URL(request.url).protocol === 'https:') {
    parts.push('Secure');
  }
  return parts.join('; ');
}

function validateDesktopAuthorizeCsrf({ request, form }) {
  const formToken = requireFormString(
    form.get('csrf_token') || form.get('csrfToken'),
    'csrf token is required',
  );
  const cookieToken = readCookieValueFromHeader(request.headers.get('cookie'), DESKTOP_AUTH_CONFIRM_CSRF_COOKIE);
  if (!cookieToken || cookieToken !== formToken) {
    throw new HttpError(403, 'desktop authorization form is invalid');
  }
}

function readCookieValueFromHeader(cookieHeader, cookieName) {
  const normalizedCookieName = String(cookieName || '').trim();
  if (!normalizedCookieName) {
    return '';
  }
  const rawHeader = String(cookieHeader || '');
  if (!rawHeader.trim()) {
    return '';
  }
  const cookies = rawHeader.split(';');
  for (const item of cookies) {
    const trimmed = item.trim();
    if (!trimmed) {
      continue;
    }
    const separatorIndex = trimmed.indexOf('=');
    if (separatorIndex <= 0) {
      continue;
    }
    const name = trimmed.slice(0, separatorIndex).trim();
    if (name !== normalizedCookieName) {
      continue;
    }
    return decodeURIComponent(trimmed.slice(separatorIndex + 1).trim());
  }
  return '';
}

function isStateChangingRequestMethod(method) {
  const normalizedMethod = String(method || '').trim().toUpperCase();
  return normalizedMethod === 'POST'
    || normalizedMethod === 'PUT'
    || normalizedMethod === 'PATCH'
    || normalizedMethod === 'DELETE';
}

function assertAllowedCookieStateChange(allowedOrigins, request) {
  if (!isStateChangingRequestMethod(request.method)) {
    return;
  }
  const origin = String(request.headers.get('origin') || '').trim();
  if (!origin) {
    throw new HttpError(403, 'Origin header is required');
  }
  if (!isAllowedOrigin(allowedOrigins, origin)) {
    throw new HttpError(403, 'Origin is not allowed');
  }
}

function isAllowedOrigin(allowedOrigins, origin) {
  const normalizedOrigin = String(origin || '').trim();
  if (!normalizedOrigin) {
    return false;
  }
  for (const allowedOrigin of normalizeAllowedOrigins(allowedOrigins)) {
    if (allowedOrigin === normalizedOrigin) {
      return true;
    }
  }
  return false;
}

function normalizeAllowedOrigins(allowedOrigins) {
  if (allowedOrigins instanceof Set) {
    return [...allowedOrigins].map((value) => String(value || '').trim()).filter(Boolean);
  }
  if (Array.isArray(allowedOrigins)) {
    return allowedOrigins.map((value) => String(value || '').trim()).filter(Boolean);
  }
  return [];
}

async function readAppUserById(db, userId) {
  const normalizedUserId = String(userId || '').trim();
  if (!normalizedUserId) {
    return null;
  }
  const row = await db.prepare(
    `SELECT id, name, email, emailVerified, role
     FROM user
     WHERE id = ?
     LIMIT 1`,
  )
    .bind(normalizedUserId)
    .first();
  return row === null ? null : appUserFromDbRow(row);
}

async function readDesktopAccessTokenUser({ db, accessToken }) {
  const accessTokenHash = await hashOpaqueToken(accessToken);
  const row = await db.prepare(
    `SELECT
       u.id,
       u.name,
       u.email,
       u.emailVerified,
       u.role
     FROM desktop_sessions ds
     JOIN user u ON u.id = ds.user_id
     WHERE ds.access_token_hash = ?
       AND ds.access_token_expires_at > ?
       AND ds.revoked_at IS NULL
     LIMIT 1`,
  )
    .bind(accessTokenHash, Date.now())
    .first();
  return row === null ? null : appUserFromDbRow(row);
}

function appUserFromDbRow(row) {
  return {
    userId: String(row.id),
    name: stringOrDefault(row.name, String(row.email || row.id || '')),
    email: stringOrDefault(row.email, ''),
    emailVerified: Number(row.emailVerified || 0) !== 0,
    role: normalizeUserRole(row.role),
    isAdmin: normalizeUserRole(row.role) === USER_ROLE_ADMIN,
    canUpload: normalizeUserRole(row.role) !== USER_ROLE_READONLY,
  };
}

async function issueDesktopTokenPair({ db, user }) {
  await purgeExpiredDesktopSessionsIfDue(db);
  const now = Date.now();
  const accessToken = generateId(64);
  const refreshToken = generateId(64);
  const accessTokenExpiresAt = now + (DESKTOP_ACCESS_TOKEN_TTL_SECONDS * 1000);
  const refreshTokenExpiresAt = now + (DESKTOP_REFRESH_TOKEN_TTL_SECONDS * 1000);
  await db.prepare(
    `INSERT INTO desktop_sessions (
       id,
       user_id,
       access_token_hash,
       access_token_expires_at,
       refresh_token_hash,
       refresh_token_expires_at,
       revoked_at,
       created_at,
       updated_at
     ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)`,
  )
    .bind(
      generateId(),
      String(user.userId),
      await hashOpaqueToken(accessToken),
      accessTokenExpiresAt,
      await hashOpaqueToken(refreshToken),
      refreshTokenExpiresAt,
      now,
      now,
    )
    .run();
  return {
    accessToken,
    accessTokenExpiresAt,
    refreshToken,
    refreshTokenExpiresAt,
    user,
  };
}

function desktopAuthResponsePayload(desktopAuth) {
  return {
    accessToken: desktopAuth.accessToken,
    accessTokenExpiresAt: timestampIso(desktopAuth.accessTokenExpiresAt),
    refreshToken: desktopAuth.refreshToken,
    refreshTokenExpiresAt: timestampIso(desktopAuth.refreshTokenExpiresAt),
    user: toApiUser(desktopAuth.user),
  };
}

async function refreshDesktopTokenPair({ db, refreshToken }) {
  await purgeExpiredDesktopSessionsIfDue(db);
  const refreshTokenHash = await hashOpaqueToken(refreshToken);
  const existing = await db.prepare(
    `SELECT
       ds.id,
       ds.user_id,
       u.id AS user_id_for_payload,
       u.name,
       u.email,
       u.emailVerified,
       u.role
     FROM desktop_sessions ds
     JOIN user u ON u.id = ds.user_id
     WHERE ds.refresh_token_hash = ?
       AND ds.refresh_token_expires_at > ?
       AND ds.revoked_at IS NULL
     LIMIT 1`,
  )
    .bind(refreshTokenHash, Date.now())
    .first();
  if (existing === null) {
    throw new HttpError(401, 'refreshToken is invalid or has expired');
  }
  const user = appUserFromDbRow({
    id: existing.user_id_for_payload,
    name: existing.name,
    email: existing.email,
    emailVerified: existing.emailVerified,
    role: existing.role,
  });
  const now = Date.now();
  const nextAccessToken = generateId(64);
  const nextRefreshToken = generateId(64);
  const nextAccessTokenExpiresAt = now + (DESKTOP_ACCESS_TOKEN_TTL_SECONDS * 1000);
  const nextRefreshTokenExpiresAt = now + (DESKTOP_REFRESH_TOKEN_TTL_SECONDS * 1000);
  const updateResult = await db.prepare(
    `UPDATE desktop_sessions
     SET access_token_hash = ?,
         access_token_expires_at = ?,
         refresh_token_hash = ?,
         refresh_token_expires_at = ?,
         updated_at = ?
     WHERE id = ?
       AND refresh_token_hash = ?
       AND revoked_at IS NULL`,
  )
    .bind(
      await hashOpaqueToken(nextAccessToken),
      nextAccessTokenExpiresAt,
      await hashOpaqueToken(nextRefreshToken),
      nextRefreshTokenExpiresAt,
      now,
      String(existing.id || ''),
      refreshTokenHash,
    )
    .run();
  if (Number(updateResult.meta?.changes || 0) < 1) {
    throw new HttpError(401, 'refreshToken is invalid or has expired');
  }
  return {
    accessToken: nextAccessToken,
    accessTokenExpiresAt: nextAccessTokenExpiresAt,
    refreshToken: nextRefreshToken,
    refreshTokenExpiresAt: nextRefreshTokenExpiresAt,
    user,
  };
}

async function revokeDesktopRefreshToken({ db, refreshToken }) {
  const refreshTokenHash = await hashOpaqueToken(refreshToken);
  await db.prepare(
    `UPDATE desktop_sessions
     SET revoked_at = ?, updated_at = ?
     WHERE refresh_token_hash = ? AND revoked_at IS NULL`,
  )
    .bind(Date.now(), Date.now(), refreshTokenHash)
    .run();
}

async function purgeExpiredDesktopSessions(db) {
  await db.prepare(
    `DELETE FROM desktop_sessions
     WHERE refresh_token_expires_at < ?
        OR revoked_at IS NOT NULL`,
  )
    .bind(Date.now())
    .run();
}

async function purgeExpiredDesktopSessionsIfDue(db) {
  if (!shouldRunDbIntervalTask(lastDesktopSessionPurgeByDb, db, DESKTOP_REQUEST_PURGE_INTERVAL_MS)) {
    return;
  }
  await purgeExpiredDesktopSessions(db);
}

function shouldRunDbIntervalTask(lastRunByDb, db, intervalMs) {
  const now = Date.now();
  const lastRunAt = Number(lastRunByDb.get(db) || 0);
  if (lastRunAt > 0 && now - lastRunAt < intervalMs) {
    return false;
  }
  lastRunByDb.set(db, now);
  return true;
}

async function hashOpaqueToken(value) {
  const digest = await crypto.subtle.digest(
    'SHA-256',
    textEncoder.encode(String(value || '')),
  );
  return bytesToHex(new Uint8Array(digest));
}

function timestampIso(value) {
  const timestamp = Number(value || 0);
  if (!Number.isFinite(timestamp) || timestamp <= 0) {
    return '';
  }
  return new Date(timestamp).toISOString();
}

async function updateAppUserName(db, { userId, name }) {
  await db.prepare(
    `UPDATE user
     SET name = ?, updatedAt = ?
     WHERE id = ?`,
  )
    .bind(String(name), Date.now(), String(userId))
    .run();
}

async function changeAppUserPassword(db, { userId, currentPassword, newPassword }) {
  const account = await db.prepare(
    `SELECT id, password
     FROM account
     WHERE userId = ? AND providerId = 'credential'
     LIMIT 1`,
  )
    .bind(String(userId))
    .first();
  if (account === null || !String(account.password || '').trim()) {
    throw new HttpError(400, 'password authentication is unavailable for this account');
  }
  const isCurrentPasswordValid = await verifyAuthPassword({
    hash: String(account.password || ''),
    password: currentPassword,
  });
  if (!isCurrentPasswordValid) {
    throw new HttpError(400, 'currentPassword is invalid');
  }
  await db.prepare(
    `UPDATE account
     SET password = ?, updatedAt = ?
     WHERE id = ?`,
  )
    .bind(await hashAuthPassword(newPassword), Date.now(), String(account.id || ''))
    .run();
  await clearAppUserSessions(db, userId);
}

async function clearAppUserSessions(db, userId) {
  await db.prepare('DELETE FROM session WHERE userId = ?')
    .bind(String(userId))
    .run();
  await db.prepare('DELETE FROM desktop_sessions WHERE user_id = ?')
    .bind(String(userId))
    .run();
}

function extractCookieFromSetCookie(setCookieHeader) {
  const text = String(setCookieHeader || '').trim();
  if (!text) {
    return '';
  }
  const firstPart = text.split(';')[0];
  return String(firstPart || '').trim();
}

function extractSessionCookieFromCookieHeader({ cookieHeader, sessionToken }) {
  const token = String(sessionToken || '').trim();
  if (!token) {
    return '';
  }
  const rawHeader = String(cookieHeader || '');
  if (!rawHeader.trim()) {
    return '';
  }
  const cookies = rawHeader.split(';');
  for (const item of cookies) {
    const trimmed = item.trim();
    if (!trimmed) {
      continue;
    }
    const separatorIndex = trimmed.indexOf('=');
    if (separatorIndex <= 0) {
      continue;
    }
    const value = trimmed.slice(separatorIndex + 1).trim();
    if (value === token) {
      return trimmed;
    }
  }
  return '';
}

async function computePkceCodeChallenge(codeVerifier) {
  const digest = await crypto.subtle.digest(
    'SHA-256',
    textEncoder.encode(String(codeVerifier || '')),
  );
  return Buffer.from(digest)
    .toString('base64')
    .replaceAll('+', '-')
    .replaceAll('/', '_')
    .replace(/=+$/g, '');
}

function copyHeaderIfPresent(sourceHeaders, targetHeaders, headerName) {
  const value = sourceHeaders.get(headerName);
  if (value) {
    targetHeaders.set(headerName, value);
  }
}

function desktopAuthorizeSocialProvider(url) {
  const normalizedProvider = String(url.searchParams.get('social_provider') || '').trim().toLowerCase();
  const shouldStart = String(url.searchParams.get('social_start') || '').trim();
  if (!normalizedProvider || shouldStart !== '1') {
    return '';
  }
  return normalizedProvider;
}

function desktopAuthorizeResumeUrl(url) {
  const resumeUrl = new URL(url.toString());
  resumeUrl.searchParams.delete('social_provider');
  resumeUrl.searchParams.delete('social_start');
  resumeUrl.searchParams.delete('error');
  resumeUrl.searchParams.delete('error_description');
  return resumeUrl.toString();
}

function desktopAuthorizeErrorCallbackUrl(url) {
  const errorUrl = new URL(desktopAuthorizeResumeUrl(url));
  return errorUrl.toString();
}

function desktopAuthorizeErrorMessage(url) {
  const error = String(url.searchParams.get('error') || '').trim();
  if (!error) {
    return '';
  }
  const description = String(url.searchParams.get('error_description') || '').trim();
  if (description) {
    return `${error}: ${description}`;
  }
  return error;
}

function errorMessageFromPayload(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return '';
  }
  if (typeof payload.message === 'string' && payload.message.trim()) {
    return payload.message.trim();
  }
  if (typeof payload.error === 'string' && payload.error.trim()) {
    return payload.error.trim();
  }
  if (typeof payload.code === 'string' && payload.code.trim()) {
    return payload.code.trim();
  }
  return '';
}

function buildDesktopAuthorizeHtml({ request, allowGoogle, allowRegistration, currentUser, csrfToken, errorMessage, email }) {
  const escapedClientId = escapeHtml(request.clientId);
  const escapedRedirectUri = escapeHtml(request.redirectUri);
  const escapedState = escapeHtml(request.state);
  const escapedCodeChallenge = escapeHtml(request.codeChallenge);
  const escapedCodeChallengeMethod = escapeHtml(request.codeChallengeMethod);
  const escapedCsrfToken = escapeHtml(csrfToken);
  const escapedEmail = escapeHtml(email);
  const escapedError = escapeHtml(errorMessage);
  const currentUserName = currentUser ? escapeHtml(currentUser.name || currentUser.email || currentUser.userId) : '';
  const currentUserEmail = currentUser ? escapeHtml(currentUser.email || '') : '';
  const googleButton = allowGoogle && allowRegistration
    ? `<form method="get" action="${DESKTOP_AUTH_BASE_PATH}/authorize" class="social-form">
        <input type="hidden" name="client_id" value="${escapedClientId}" />
        <input type="hidden" name="redirect_uri" value="${escapedRedirectUri}" />
        <input type="hidden" name="state" value="${escapedState}" />
        <input type="hidden" name="code_challenge" value="${escapedCodeChallenge}" />
        <input type="hidden" name="code_challenge_method" value="${escapedCodeChallengeMethod}" />
        <input type="hidden" name="social_provider" value="google" />
        <input type="hidden" name="social_start" value="1" />
        <button type="submit" class="secondary">Continue with Google</button>
      </form>`
    : '';
  const registrationHint = allowRegistration
    ? '<p class="muted">Use your Feel8 Asset Cloud account to continue in Feel8 Studio.</p>'
    : '<p class="muted">Registration is currently disabled. Sign in with an existing account.</p>';
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Sign in to Feel8 Asset Cloud</title>
    <style>
      :root {
        color-scheme: dark;
        font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      }
      body {
        margin: 0;
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #0b1220;
        color: #edf2ff;
      }
      main {
        width: min(92vw, 460px);
        padding: 28px;
        border: 1px solid #30456b;
        border-radius: 16px;
        background: #111b2e;
        box-shadow: 0 24px 64px rgba(0, 0, 0, 0.35);
      }
      h1 {
        margin: 0 0 8px;
        font-size: 1.6rem;
      }
      p {
        margin: 0 0 12px;
      }
      .muted {
        color: #9fb0cc;
      }
      .error {
        margin: 16px 0;
        padding: 10px 12px;
        border-radius: 10px;
        background: rgba(239, 68, 68, 0.16);
        border: 1px solid rgba(248, 113, 113, 0.45);
        color: #fecaca;
      }
      form {
        display: grid;
        gap: 14px;
        margin-top: 18px;
      }
      .section {
        margin-top: 18px;
        padding: 14px;
        border-radius: 12px;
        border: 1px solid #30456b;
        background: rgba(12, 24, 44, 0.72);
      }
      label {
        display: grid;
        gap: 6px;
        font-size: 0.95rem;
      }
      input {
        border: 1px solid #41577f;
        border-radius: 10px;
        background: #0b1426;
        color: inherit;
        padding: 12px 14px;
        font: inherit;
      }
      button {
        border: 0;
        border-radius: 10px;
        padding: 12px 16px;
        font: inherit;
        font-weight: 600;
        background: #5aa9ff;
        color: #04111f;
        cursor: pointer;
      }
      button.secondary {
        background: #e5eefc;
        color: #0b1426;
      }
      .social-form {
        margin-top: 14px;
      }
    </style>
  </head>
  <body>
    <main>
      <h1>Continue to Feel8 Studio</h1>
      <p class="muted">Sign in here and we’ll send your browser back to the running Studio session.</p>
      ${registrationHint}
      ${escapedError ? `<div class="error">${escapedError}</div>` : ''}
      ${currentUser ? `<section class="section">
        <p>Signed in as <strong>${currentUserName}</strong>${currentUserEmail ? ` <span class="muted">(${currentUserEmail})</span>` : ''}</p>
        <form method="post" action="${DESKTOP_AUTH_BASE_PATH}/authorize">
          <input type="hidden" name="client_id" value="${escapedClientId}" />
          <input type="hidden" name="redirect_uri" value="${escapedRedirectUri}" />
          <input type="hidden" name="state" value="${escapedState}" />
          <input type="hidden" name="code_challenge" value="${escapedCodeChallenge}" />
          <input type="hidden" name="code_challenge_method" value="${escapedCodeChallengeMethod}" />
          <input type="hidden" name="csrf_token" value="${escapedCsrfToken}" />
          <button type="submit">Continue as ${currentUserName}</button>
        </form>
      </section>` : ''}
      <form method="post" action="${DESKTOP_AUTH_BASE_PATH}/authorize">
        <input type="hidden" name="client_id" value="${escapedClientId}" />
        <input type="hidden" name="redirect_uri" value="${escapedRedirectUri}" />
        <input type="hidden" name="state" value="${escapedState}" />
        <input type="hidden" name="code_challenge" value="${escapedCodeChallenge}" />
        <input type="hidden" name="code_challenge_method" value="${escapedCodeChallengeMethod}" />
        <input type="hidden" name="csrf_token" value="${escapedCsrfToken}" />
        <label>
          Email
          <input type="email" name="email" value="${escapedEmail}" autocomplete="email" ${currentUser ? '' : 'required'} />
        </label>
        <label>
          Password
          <input type="password" name="password" autocomplete="current-password" ${currentUser ? '' : 'required'} />
        </label>
        <button type="submit">${currentUser ? 'Use a different account' : 'Sign in to Asset Cloud'}</button>
      </form>
      ${googleButton}
    </main>
  </body>
</html>`;
}

function htmlResponse(status, html) {
  return new Response(html, {
    status,
    headers: {
      'Content-Type': 'text/html; charset=utf-8',
      'Cache-Control': 'no-store',
    },
  });
}

function getOrCreateDbCache(cacheByDb, db) {
  let cache = cacheByDb.get(db);
  if (cache !== undefined) {
    return cache;
  }
  cache = new Map();
  cacheByDb.set(db, cache);
  return cache;
}

function buildAuthCacheKey({ allowUserRegistration, baseURL, trustedOrigins }) {
  return JSON.stringify({
    allowUserRegistration: Boolean(allowUserRegistration),
    baseURL: String(baseURL || '').trim(),
    trustedOrigins: [...trustedOrigins].sort(),
  });
}

function invalidateAuthCache(db) {
  if (db && typeof db === 'object') {
    authCacheByDb.delete(db);
  }
}

function resolveAuthBaseUrl(env, requestUrl) {
  const configured = String(env.AUTH_BASE_URL || '').trim();
  if (!configured) {
    throw new HttpError(500, 'AUTH_BASE_URL is not configured');
  }
  try {
    return new URL(configured).toString().replace(/\/+$/g, '');
  } catch (error) {
    throw new HttpError(500, 'AUTH_BASE_URL must be a valid absolute URL');
  }
}

function calculateAllowedOrigins(env, requestUrl = null) {
  const origins = new Set();
  addOrigin(origins, resolveAuthBaseUrl(env));
  addLoopbackRequestOrigin(origins, requestUrl);
  const extra = String(env.CORS_ALLOWED_ORIGINS || '').trim();
  if (extra) {
    for (const value of extra.split(',')) {
      addOrigin(origins, value);
    }
  }
  return origins;
}

function resolveTrustedOrigins(env, requestUrl, baseURL) {
  return [...calculateAllowedOrigins(env, requestUrl)];
}

function addOrigin(target, value) {
  const text = String(value || '').trim();
  if (!text) {
    return;
  }
  try {
    target.add(new URL(text).origin);
  } catch (error) {
    console.warn('Ignoring invalid trusted origin', text);
  }
}

function addLoopbackRequestOrigin(target, requestUrl) {
  if (!(requestUrl instanceof URL)) {
    return;
  }
  if (!isLoopbackHostname(requestUrl.hostname)) {
    return;
  }
  target.add(requestUrl.origin);
}

function isLoopbackHostname(hostname) {
  const value = String(hostname || '').trim().toLowerCase();
  return value === 'localhost' || value === '127.0.0.1' || value === '::1' || value === '[::1]';
}

function resolveAllowedOrigin(env, origin, requestUrl = null) {
  const normalizedOrigin = String(origin || '').trim();
  if (!normalizedOrigin) {
    return null;
  }
  return isAllowedOrigin(calculateAllowedOrigins(env, requestUrl), normalizedOrigin) ? normalizedOrigin : null;
}

function validateIdentityName(value, { allowReserved = false } = {}) {
  const text = String(value || '').trim();
  if (!text) {
    return false;
  }
  const canonical = canonicalizeIdentityName(text);
  if (RESERVED_IDENTITY_NAMES.has(canonical) && !allowReserved) {
    return false;
  }
  return isSafeDisplayText(text);
}

function isReservedIdentityName(value) {
  const text = String(value || '').trim();
  if (!text) {
    return false;
  }
  return RESERVED_IDENTITY_NAMES.has(canonicalizeIdentityName(text));
}

function validateDisplayName(value, { allowedReservedValue = '', allowReserved = false } = {}) {
  const text = String(value || '').trim();
  if (!text) {
    return false;
  }
  const canonical = canonicalizeIdentityName(text);
  if (
    RESERVED_IDENTITY_NAMES.has(canonical)
    && !allowReserved
    && canonical !== canonicalizeIdentityName(allowedReservedValue)
  ) {
    return false;
  }
  return isSafeDisplayText(text);
}

function assertReservedNameAllowedForRole({ name, role }) {
  if (!isReservedIdentityName(name)) {
    return;
  }
  if (normalizeUserRole(role) === USER_ROLE_ADMIN) {
    return;
  }
  throw APIError.fromStatus('BAD_REQUEST', {
    message: 'reserved names are only available to admins',
  });
}

function requireUserProfileName(value, { currentName = '', allowReserved = false } = {}) {
  const name = requireBodyString(value, 'name is required');
  if (!validateDisplayName(name, { allowedReservedValue: currentName, allowReserved })) {
    throw new HttpError(400, 'name must be 2-64 visible characters and not reserved');
  }
  return name;
}

function isSafeDisplayText(value) {
  const text = String(value || '').trim();
  if (text.length < 2 || text.length > 64) {
    return false;
  }
  if (/[\u0000-\u001F\u007F]/.test(text)) {
    return false;
  }
  return true;
}

function canonicalizeIdentityName(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[\s._-]+/g, '');
}

function parseManagementUserPath(pathname) {
  const prefix = `${MANAGEMENT_API_BASE_PATH}/users/`;
  const tail = decodeURIComponent(pathname.slice(prefix.length));
  const parts = tail.split('/').filter((part) => part.length > 0);
  return {
    userId: parts[0] || '',
    suffix: parts.slice(1),
  };
}

function decodeSinglePathValue(pathname, prefix) {
  const value = decodeURIComponent(pathname.slice(prefix.length));
  if (!value || value.includes('/')) {
    return '';
  }
  return value;
}

async function readJsonBody(request) {
  const contentEncoding = String(request.headers.get('Content-Encoding') || '').toLowerCase();
  let raw = '';
  if (contentEncoding.includes('gzip')) {
    const compressedBody = await readRequestBytesWithLimit(request, MAX_REQUEST_COMPRESSED_BYTES);
    if (compressedBody.byteLength === 0) {
      return {};
    }
    try {
      raw = await decompressGzipBodyWithLimit(compressedBody, MAX_REQUEST_JSON_BYTES);
    } catch (error) {
      if (error instanceof HttpError) {
        throw error;
      }
      throw new HttpError(400, 'request body gzip decompression failed');
    }
  } else {
    raw = await readRequestTextWithLimit(request, MAX_REQUEST_JSON_BYTES);
  }
  if (!raw) {
    return {};
  }
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new HttpError(400, 'request body must be a JSON object');
  }
  if (!isPlainObject(parsed)) {
    throw new HttpError(400, 'request body must be a JSON object');
  }
  return parsed;
}

async function readRequestTextWithLimit(request, maxBytes) {
  const bytes = await readRequestBytesWithLimit(request, maxBytes);
  return new TextDecoder().decode(bytes);
}

async function readRequestBytesWithLimit(request, maxBytes) {
  if (request.body === null) {
    return new Uint8Array();
  }
  const reader = request.body.getReader();
  const chunks = [];
  let totalBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    const chunk = value instanceof Uint8Array ? value : new Uint8Array(value);
    totalBytes += chunk.byteLength;
    if (totalBytes > Number(maxBytes)) {
      throw new HttpError(413, 'request body is too large');
    }
    chunks.push(chunk);
  }
  return concatUint8Arrays(chunks, totalBytes);
}

async function decompressGzipBodyWithLimit(buffer, maxBytes) {
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'));
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let totalBytes = 0;
  let output = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    const chunk = value instanceof Uint8Array ? value : new Uint8Array(value);
    totalBytes += chunk.byteLength;
    if (totalBytes > Number(maxBytes)) {
      throw new HttpError(413, 'request body is too large');
    }
    output += decoder.decode(chunk, { stream: true });
  }
  output += decoder.decode();
  return output;
}

function concatUint8Arrays(chunks, totalBytes) {
  const output = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output;
}

function handleError(error) {
  if (error instanceof HttpError) {
    return jsonResponse(error.status, { message: error.message, ...error.payload });
  }
  if (error instanceof ApiException) {
    const errors = error.buildResponse();
    const primaryMessage = errors.length > 0 ? String(errors[0].message || '') : String(error.message || '');
    return jsonResponse(error.status, {
      message: primaryMessage || 'request failed',
      errors,
    });
  }
  const apiError = toAuthApiErrorPayload(error);
  if (apiError !== null) {
    return jsonResponse(apiError.status, apiError.payload);
  }
  if (error instanceof AssetPermissionError) {
    return jsonResponse(403, { message: 'forbidden' });
  }
  if (error instanceof AssetNotFoundError) {
    return jsonResponse(404, { message: 'asset not found' });
  }
  if (error instanceof AssetConflictError) {
    const payload = {
      message: 'conflict',
      versionNumber: error.versionNumber,
    };
    if (error.assetType === 'variant') {
      payload.variantId = error.assetId;
    } else if (error.assetType === 'modding_recipe') {
      payload.recipeId = error.assetId;
    } else {
      payload.componentId = error.assetId;
    }
    return jsonResponse(409, payload);
  }
  if (error instanceof AssetValidationError) {
    return jsonResponse(400, { message: error.message });
  }
  if (error instanceof Error && isUniqueConstraintError(error)) {
    return jsonResponse(409, { message: 'duplicate resource' });
  }
  if (error instanceof Error && error.message) {
    if (error.message.includes('already exists') || error.message.includes('duplicate')) {
      return jsonResponse(409, { message: 'duplicate resource' });
    }
  }
  console.error('Unhandled asset worker error', error);
  return jsonResponse(500, { message: 'internal error' });
}

function toAuthApiErrorPayload(error) {
  const status = Number(error?.statusCode || 0);
  if (!Number.isInteger(status) || status < 400 || status > 599) {
    return null;
  }
  const body = isPlainObject(error?.body) ? error.body : {};
  const message = body.message ? String(body.message) : (error instanceof Error ? error.message : 'request failed');
  const payload = {
    ...body,
    message,
  };
  return { status, payload };
}

function isUniqueConstraintError(error) {
  const message = error instanceof Error ? error.message : '';
  const causeMessage = error?.cause instanceof Error ? error.cause.message : '';
  const causeErrcode = Number(error?.cause?.errcode || 0);
  return (
    message.includes('UNIQUE constraint failed')
    || causeMessage.includes('UNIQUE constraint failed')
    || causeErrcode === 2067
  );
}

function jsonResponse(status, payload, extraHeaders = null) {
  const headers = new Headers({
    'Content-Type': 'application/json',
  });
  appendHeaders(headers, extraHeaders);
  return new Response(JSON.stringify(payload), {
    status,
    headers,
  });
}

function assetDownloadResponse(payload, { head, versionNumber, headers: extraHeaders = null }) {
  const baseName = stringOrDefault(head && head.name, String(head ? head.asset_id : 'asset'));
  const slug = slugifyForFilename(baseName) || 'asset';
  const versionSuffix = Number.isFinite(Number(versionNumber)) ? `-${Number(versionNumber)}` : '';
  const filename = `${slug}${versionSuffix}.json`;
  const body = JSON.stringify(payload, null, 2);
  const headers = new Headers({
    'Content-Type': 'application/json',
    'Content-Disposition': `attachment; filename="${filename}"`,
    'Cache-Control': 'no-store',
  });
  appendHeaders(headers, extraHeaders);
  return new Response(body, {
    status: 200,
    headers,
  });
}

function appendHeaders(targetHeaders, sourceHeaders) {
  if (!sourceHeaders) {
    return;
  }
  const iterable = sourceHeaders instanceof Headers
    ? sourceHeaders.entries()
    : Object.entries(sourceHeaders);
  for (const [key, value] of iterable) {
    if (value === null || value === undefined || String(value).trim() === '') {
      continue;
    }
    targetHeaders.set(key, String(value));
  }
}

function isAnonymousRequest(request) {
  return !String(request.headers.get('authorization') || '').trim()
    && !String(request.headers.get('cookie') || '').trim();
}

function anonymousPublicResponseHeaders(request, shouldCache) {
  if (!shouldCache || !isAnonymousRequest(request)) {
    return null;
  }
  return {
    'Cache-Control': PUBLIC_CACHE_CONTROL_HEADER,
    Vary: 'Authorization, Cookie',
  };
}

function applyAnonymousPublicCacheHeaders(c, request, shouldCache) {
  const headers = anonymousPublicResponseHeaders(request, shouldCache);
  if (!headers) {
    return;
  }
  for (const [key, value] of Object.entries(headers)) {
    c.header(key, value);
  }
}

function slugifyForFilename(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80);
}

function frontendFallbackResponse() {
  return new Response(buildPortalFallbackHtml(), {
    status: 200,
    headers: {
      'Content-Type': 'text/html; charset=utf-8',
      'Cache-Control': 'no-store',
    },
  });
}

function buildVerifyEmailUrl({ env, baseURL, token }) {
  const configuredBaseUrl = String(env.AUTH_VERIFY_EMAIL_BASE_URL || '').trim();
  const base = configuredBaseUrl
    ? new URL(configuredBaseUrl)
    : new URL('/verify-email', baseURL);
  base.searchParams.set('token', token);
  return base.toString();
}

function buildResetPasswordUrl({ env, baseURL, token }) {
  const configuredBaseUrl = String(env.AUTH_RESET_PASSWORD_BASE_URL || '').trim();
  const base = configuredBaseUrl
    ? new URL(configuredBaseUrl)
    : new URL('/reset-password', baseURL);
  base.searchParams.set('token', token);
  return base.toString();
}

function resolveAuthActionUrl({ preferredUrl, fallbackUrl }) {
  const candidate = String(preferredUrl || '').trim();
  if (candidate) {
    return candidate;
  }
  return String(fallbackUrl || '').trim();
}

async function sendVerifyEmailMessage({ env, toEmail, recipientName, verificationUrl }) {
  await sendAuthEmail({
    env,
    debugLabel: 'verify email',
    debugUrl: verificationUrl,
    toEmail,
    subject: 'Verify your email',
    text: `Hi ${recipientName}, verify your email: ${verificationUrl}`,
    html: `<p>Hi ${escapeHtml(recipientName)},</p><p>Please verify your email:</p><p><a href="${escapeHtml(verificationUrl)}">${escapeHtml(verificationUrl)}</a></p>`,
  });
}

async function sendResetPasswordMessage({ env, toEmail, recipientName, resetUrl }) {
  await sendAuthEmail({
    env,
    debugLabel: 'reset password',
    debugUrl: resetUrl,
    toEmail,
    subject: 'Reset your password',
    text: `Hi ${recipientName}, reset your password: ${resetUrl}`,
    html: `<p>Hi ${escapeHtml(recipientName)},</p><p>Use this link to reset password:</p><p><a href="${escapeHtml(resetUrl)}">${escapeHtml(resetUrl)}</a></p>`,
  });
}

function shouldBlockPublicRegistration({ siteSettings, request }) {
  if (!isPublicRegistrationRequest(request)) {
    return false;
  }
  return !siteSettings.allowUserRegistration;
}

function isPublicRegistrationRequest(request) {
  const url = new URL(request.url);
  if (request.method !== 'POST') {
    return false;
  }
  return url.pathname === `${AUTH_BASE_PATH}/sign-up/email`;
}

async function readSiteSettings(db) {
  const row = await db.prepare(
    `SELECT allow_user_registration
     FROM site_settings
     WHERE id = 1`,
  ).first();
  if (row === null) {
    await db.prepare(
      `INSERT INTO site_settings (id, allow_user_registration, updated_at, updated_by_user_id)
       VALUES (1, 0, CURRENT_TIMESTAMP, NULL)
       ON CONFLICT(id) DO NOTHING`,
    ).run();
    return {
      allowUserRegistration: false,
    };
  }
  return {
    allowUserRegistration: Number(row?.allow_user_registration ?? 0) !== 0,
  };
}

async function computeBootstrapAdminFingerprint(env, config) {
  const payload = JSON.stringify({
    authSecret: getAuthSecret(env),
    passwordHashVersion: authPasswordHashVersion(),
    email: config.email,
    password: config.password,
    name: config.name,
  });
  const digest = await crypto.subtle.digest('SHA-256', textEncoder.encode(payload));
  return bytesToHex(new Uint8Array(digest));
}

async function readBootstrapAdminState(db) {
  const row = await db.prepare(
    `SELECT config_fingerprint, user_id
     FROM bootstrap_admin_state
     WHERE id = ?`,
  )
    .bind(BOOTSTRAP_ADMIN_STATE_ROW_ID)
    .first();
  if (row === null) {
    return null;
  }
  const configFingerprint = String(row.config_fingerprint || '').trim();
  const userId = String(row.user_id || '').trim();
  if (!configFingerprint || !userId) {
    return null;
  }
  return {
    configFingerprint,
    userId,
  };
}

async function readBootstrapAdminCredentialAccount(db, userId) {
  return db.prepare(
    `SELECT
       u.id,
       u.email,
       u.name,
       u.role,
       u.emailVerified,
       a.id AS credential_account_id
     FROM user u
     LEFT JOIN account a
       ON a.userId = u.id AND a.providerId = 'credential'
     WHERE u.id = ?
     LIMIT 1`,
  )
    .bind(String(userId))
    .first();
}

function bootstrapAdminMatchesConfig(row, config) {
  if (row === null) {
    return false;
  }
  if (row.credential_account_id === null || row.credential_account_id === undefined) {
    return false;
  }
  return (
    String(row.email || '').trim().toLowerCase() === config.email
    && String(row.name || '').trim() === config.name
    && String(row.role || '').trim() === USER_ROLE_ADMIN
    && Number(row.emailVerified || 0) !== 0
  );
}

async function upsertBootstrapAdminState(db, { configFingerprint, userId }) {
  await db.prepare(
    `INSERT INTO bootstrap_admin_state (id, config_fingerprint, user_id, synced_at)
     VALUES (?, ?, ?, CURRENT_TIMESTAMP)
     ON CONFLICT(id) DO UPDATE SET
       config_fingerprint = excluded.config_fingerprint,
       user_id = excluded.user_id,
       synced_at = CURRENT_TIMESTAMP`,
  )
    .bind(BOOTSTRAP_ADMIN_STATE_ROW_ID, String(configFingerprint), String(userId))
    .run();
}

function bytesToHex(bytes) {
  let out = '';
  for (const value of bytes) {
    out += value.toString(16).padStart(2, '0');
  }
  return out;
}

function normalizeUserRole(value) {
  const role = String(value || '').trim().toLowerCase();
  if (role === USER_ROLE_ADMIN || role === USER_ROLE_READONLY) {
    return role;
  }
  return USER_ROLE_USER;
}

function normalizeManagedUserRolePayload(payload) {
  if (payload.role !== undefined) {
    return requireUserRole(payload.role);
  }
  if (payload.isAdmin === true) {
    return USER_ROLE_ADMIN;
  }
  if (payload.canUpload === false) {
    return USER_ROLE_READONLY;
  }
  return USER_ROLE_USER;
}

function requireUserRole(value) {
  const role = String(value || '').trim().toLowerCase();
  if (role === USER_ROLE_ADMIN || role === USER_ROLE_USER || role === USER_ROLE_READONLY) {
    return role;
  }
  throw new HttpError(400, 'role must be admin, user, or readonly');
}

async function sendAuthEmail({ env, debugLabel, debugUrl, toEmail, subject, text, html }) {
  const resendApiKey = String(env.RESEND_API_KEY || '').trim();
  const fromEmail = String(env.AUTH_EMAIL_FROM || '').trim();
  if (!resendApiKey || !fromEmail) {
    if (toBoolean(env.EXPOSE_DEBUG_AUTH_LINKS)) {
      console.info(`[auth debug] ${debugLabel}: ${debugUrl}`);
    } else {
      console.warn(`[auth] email delivery skipped (RESEND_API_KEY / AUTH_EMAIL_FROM not configured): ${debugLabel}`);
    }
    return;
  }

  const response = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${resendApiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      from: fromEmail,
      to: [toEmail],
      subject,
      html,
      text,
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new HttpError(502, `failed to send auth email: ${response.status} ${body}`);
  }
}

async function serveFrontend(request, env, url) {
  if (!isPortalStaticAssetPath(url.pathname) && url.pathname !== '/favicon.ico' && !isPortalAppPath(url.pathname)) {
    return jsonResponse(404, { message: 'not found' });
  }

  const assets = getAssetsBinding(env);
  if (assets === null) {
    if (isPortalStaticAssetPath(url.pathname) || url.pathname === '/favicon.ico') {
      return jsonResponse(404, { message: 'not found' });
    }
    return frontendFallbackResponse();
  }

  let assetPath = '/';
  if (isPortalStaticAssetPath(url.pathname)) {
    assetPath = url.pathname;
  } else if (url.pathname === '/favicon.ico') {
    assetPath = '/favicon.ico';
  }

  const assetRequest = new Request(new URL(assetPath, 'https://assets.invalid'), request);
  return assets.fetch(assetRequest);
}

function isPortalAppPath(pathname) {
  const normalizedPath = String(pathname || '');
  if (normalizedPath === '/') {
    return true;
  }
  const exactRoutes = new Set([
    '/login',
    '/register',
    '/forgot-password',
    '/reset-password',
    '/verify-email',
    '/auth-callback',
    '/auth-complete',
    '/auth-error',
    '/browse',
    '/profile',
  ]);
  if (exactRoutes.has(normalizedPath)) {
    return true;
  }
  return normalizedPath.startsWith('/assets/') || normalizedPath.startsWith('/admin/');
}

function isPortalStaticAssetPath(pathname) {
  if (!pathname.startsWith(`${PORTAL_STATIC_DIR}/`)) {
    return false;
  }
  const lastSegment = pathname.split('/').pop() || '';
  return /\.[a-z0-9]+$/i.test(lastSegment);
}

function getAssetsBinding(env) {
  if (!env || typeof env !== 'object') {
    return null;
  }
  const assets = env.ASSETS;
  if (assets && typeof assets.fetch === 'function') {
    return assets;
  }
  return null;
}

function buildPortalFallbackHtml() {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Feel8 Asset Cloud</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: #0c1220; color: #edf2ff; }
      main { max-width: 760px; margin: 48px auto; padding: 24px; border: 1px solid #30456b; border-radius: 12px; background: #121c31; }
      h1 { margin-top: 0; }
      code { background: #0b1426; padding: 2px 6px; border-radius: 6px; }
    </style>
  </head>
  <body>
    <main>
      <h1>Feel8 Asset Cloud</h1>
      <p>Frontend assets are not available yet.</p>
      <p>Build the Vite app first:</p>
      <p><code>npm run web:build</code></p>
      <p>Then open <code>/</code> in the browser.</p>
    </main>
  </body>
</html>`;
}

function secondsFromEnv(value, fallback) {
  const parsed = Number.parseInt(String(value || fallback), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function escapeHtml(value) {
  return String(value || '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function requireBodyString(value, message) {
  const text = String(value || '').trim();
  if (!text) {
    throw new HttpError(400, message);
  }
  return text;
}

function requireQueryString(value, message) {
  const text = String(value || '').trim();
  if (!text) {
    throw new HttpError(400, message);
  }
  return text;
}

class HttpError extends Error {
  constructor(status, message, payload = {}) {
    super(String(message || 'request failed'));
    this.status = status;
    this.payload = payload;
  }
}
