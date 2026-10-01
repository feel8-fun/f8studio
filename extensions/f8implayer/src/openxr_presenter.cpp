#include "openxr_presenter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <glad/glad.h>
#include <spdlog/spdlog.h>

#include <SDL3/SDL_loadso.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <unknwn.h>
#include <wingdi.h>
#endif

#define XR_NO_PROTOTYPES
#define XR_USE_GRAPHICS_API_OPENGL
#if defined(_WIN32)
#define XR_USE_PLATFORM_WIN32
#endif
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

namespace f8::implayer {

namespace {

constexpr float kPi = 3.14159265358979323846f;

constexpr const char* kXrVertexShader = R"(
#version 330 core
out vec2 v_ndc;
void main() {
  vec2 pos;
  if (gl_VertexID == 0) pos = vec2(-1.0, -1.0);
  else if (gl_VertexID == 1) pos = vec2(3.0, -1.0);
  else pos = vec2(-1.0, 3.0);
  v_ndc = pos;
  gl_Position = vec4(pos, 0.0, 1.0);
}
)";

constexpr const char* kXrFragmentShader = R"(
#version 330 core
in vec2 v_ndc;
out vec4 FragColor;

uniform sampler2D uTexture;
uniform int uMode;      // 0=flat2d, 1=equirect_mono, 2=equirect_sbs
uniform int uSbsEye;    // 0=left, 1=right

// tan(fov angles) for this view: (left, right, down, up)
uniform vec4 uFovTan;

// view orientation quaternion (x,y,z,w) mapping view->world (OpenXR convention)
uniform vec4 uOrientation;

// Flat2D aspect handling
uniform float uVideoAspect;
uniform float uTargetAspect;

vec3 quat_rotate(vec4 q, vec3 v) {
  // q * v * q^-1, assuming q is normalized.
  vec3 u = q.xyz;
  float s = q.w;
  return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

void main() {
  vec2 uv = v_ndc * 0.5 + 0.5;
  if (uMode == 0) {
    // Letterbox/pillarbox to preserve video aspect.
    float va = max(uVideoAspect, 1e-6);
    float ta = max(uTargetAspect, 1e-6);
    vec2 p = uv * 2.0 - 1.0;  // [-1,1]
    if (ta > va) {
      p.x *= va / ta;
    } else {
      p.y *= ta / va;
    }
    vec2 u2 = p * 0.5 + 0.5;
    if (u2.x < 0.0 || u2.x > 1.0 || u2.y < 0.0 || u2.y > 1.0) {
      FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
      FragColor = texture(uTexture, u2);
    }
    return;
  }

  // Build a view ray using per-view FOV (OpenXR uses -Z forward).
  float tx = mix(uFovTan.x, uFovTan.y, (v_ndc.x + 1.0) * 0.5);
  float ty = mix(uFovTan.z, uFovTan.w, (v_ndc.y + 1.0) * 0.5);
  vec3 ray_view = normalize(vec3(tx, -ty, -1.0));
  vec3 d = quat_rotate(uOrientation, ray_view);

  // Convert direction to equirect UV (place seam behind: use -Z forward).
  float u = atan(d.x, -d.z) / (2.0 * 3.14159265358979323846) + 0.5;
  float v = 0.5 - asin(clamp(d.y, -1.0, 1.0)) / 3.14159265358979323846;

  if (uMode == 2) {
    if (uSbsEye == 0) u *= 0.5;
    else u = 0.5 + u * 0.5;
  }

  FragColor = texture(uTexture, vec2(fract(u), clamp(v, 0.0, 1.0)));
}
)";

constexpr const char* kHudVertexShader = R"(
#version 330 core
out vec2 v_uv;
void main() {
  vec2 pos;
  if (gl_VertexID == 0) pos = vec2(-1.0, -1.0);
  else if (gl_VertexID == 1) pos = vec2(3.0, -1.0);
  else pos = vec2(-1.0, 3.0);
  v_uv = pos * 0.5 + 0.5;
  gl_Position = vec4(pos, 0.0, 1.0);
}
)";

constexpr const char* kHudFragmentShader = R"(
#version 330 core
in vec2 v_uv;
out vec4 FragColor;

uniform float uProgress01;
uniform int uPlaying;
uniform int uProjectionMode;  // 0=flat,1=mono,2=sbs
uniform int uHoverId;         // 0=none,1=prev,2=play,3=next,4=proj,5=seek
uniform vec2 uPointerUV;      // [-1,-1] if none

float in_rect(vec2 uv, vec4 r) {
  // r = x0,y0,x1,y1
  return step(r.x, uv.x) * step(r.y, uv.y) * step(uv.x, r.z) * step(uv.y, r.w);
}

vec4 blend_over(vec4 dst, vec4 src) {
  float a = src.a + dst.a * (1.0 - src.a);
  vec3 rgb = (src.rgb * src.a + dst.rgb * dst.a * (1.0 - src.a)) / max(a, 1e-6);
  return vec4(rgb, a);
}

void main() {
  vec4 col = vec4(0.0, 0.0, 0.0, 0.0);

  // Panel background.
  vec4 panel = vec4(0.04, 0.04, 0.05, 0.70);
  col = blend_over(col, panel);

  // Layout.
  vec4 btn_prev = vec4(0.03, 0.18, 0.15, 0.82);
  vec4 btn_play = vec4(0.18, 0.18, 0.30, 0.82);
  vec4 btn_next = vec4(0.33, 0.18, 0.45, 0.82);
  vec4 btn_proj = vec4(0.86, 0.18, 0.97, 0.82);
  vec4 bar_seek = vec4(0.03, 0.05, 0.97, 0.14);

  // Hover highlight.
  float h_prev = (uHoverId == 1) ? in_rect(v_uv, btn_prev) : 0.0;
  float h_play = (uHoverId == 2) ? in_rect(v_uv, btn_play) : 0.0;
  float h_next = (uHoverId == 3) ? in_rect(v_uv, btn_next) : 0.0;
  float h_proj = (uHoverId == 4) ? in_rect(v_uv, btn_proj) : 0.0;
  float h_seek = (uHoverId == 5) ? in_rect(v_uv, bar_seek) : 0.0;
  float h = max(max(h_prev, h_play), max(max(h_next, h_proj), h_seek));
  if (h > 0.0) {
    col = blend_over(col, vec4(0.20, 0.22, 0.28, 0.35));
  }

  // Seek bar track.
  float in_bar = in_rect(v_uv, bar_seek);
  if (in_bar > 0.0) {
    col = blend_over(col, vec4(0.18, 0.18, 0.20, 0.70));
    float p = clamp(uProgress01, 0.0, 1.0);
    vec4 fill = vec4(bar_seek.x, bar_seek.y, mix(bar_seek.x, bar_seek.z, p), bar_seek.w);
    if (in_rect(v_uv, fill) > 0.0) {
      col = blend_over(col, vec4(0.35, 0.75, 0.95, 0.85));
    }
    // Handle
    float hx = mix(bar_seek.x, bar_seek.z, p);
    vec4 handle = vec4(hx - 0.006, bar_seek.y - 0.015, hx + 0.006, bar_seek.w + 0.015);
    if (in_rect(v_uv, handle) > 0.0) {
      col = blend_over(col, vec4(0.92, 0.92, 0.92, 0.95));
    }
  }

  // Simple icons (procedural shapes).
  vec3 ico = vec3(0.85, 0.85, 0.88);
  // Prev: bar + triangle.
  if (in_rect(v_uv, btn_prev) > 0.0) {
    vec2 uv = (v_uv - btn_prev.xy) / (btn_prev.zw - btn_prev.xy);
    float bar = in_rect(uv, vec4(0.18, 0.25, 0.24, 0.75));
    float tri = step(uv.x, 0.70 - uv.y) * step(uv.x, 0.70 - (1.0 - uv.y)) * step(0.24, uv.x);
    float m = max(bar, tri);
    if (m > 0.0) col = blend_over(col, vec4(ico, 0.90));
  }
  // Next: triangle + bar.
  if (in_rect(v_uv, btn_next) > 0.0) {
    vec2 uv = (v_uv - btn_next.xy) / (btn_next.zw - btn_next.xy);
    float bar = in_rect(uv, vec4(0.76, 0.25, 0.82, 0.75));
    float tri = step(uv.x, 0.76) * step(uv.y, 0.5 + uv.x - 0.18) * step(0.5 - uv.x + 0.18, uv.y);
    float m = max(bar, tri);
    if (m > 0.0) col = blend_over(col, vec4(ico, 0.90));
  }
  // Play/Pause.
  if (in_rect(v_uv, btn_play) > 0.0) {
    vec2 uv = (v_uv - btn_play.xy) / (btn_play.zw - btn_play.xy);
    if (uPlaying != 0) {
      float r1 = in_rect(uv, vec4(0.28, 0.25, 0.40, 0.75));
      float r2 = in_rect(uv, vec4(0.60, 0.25, 0.72, 0.75));
      if (max(r1, r2) > 0.0) col = blend_over(col, vec4(ico, 0.90));
    } else {
      float tri = step(uv.y, 0.18 + uv.x) * step(0.82 - uv.x, uv.y);
      if (tri > 0.0 && uv.x <= 0.82) col = blend_over(col, vec4(ico, 0.90));
    }
  }
  // Projection mode indicator: 3 dots (flat/mono/sbs) highlighted.
  if (in_rect(v_uv, btn_proj) > 0.0) {
    vec2 uv = (v_uv - btn_proj.xy) / (btn_proj.zw - btn_proj.xy);
    vec2 c0 = vec2(0.25, 0.50);
    vec2 c1 = vec2(0.50, 0.50);
    vec2 c2 = vec2(0.75, 0.50);
    float d0 = length(uv - c0);
    float d1 = length(uv - c1);
    float d2 = length(uv - c2);
    float dot0 = step(d0, 0.12);
    float dot1 = step(d1, 0.12);
    float dot2 = step(d2, 0.12);
    vec3 a0 = (uProjectionMode == 0) ? vec3(0.35, 0.75, 0.95) : ico;
    vec3 a1 = (uProjectionMode == 1) ? vec3(0.35, 0.75, 0.95) : ico;
    vec3 a2 = (uProjectionMode == 2) ? vec3(0.35, 0.75, 0.95) : ico;
    vec4 src = vec4(0.0);
    if (dot0 > 0.0) src = vec4(a0, 0.90);
    if (dot1 > 0.0) src = vec4(a1, 0.90);
    if (dot2 > 0.0) src = vec4(a2, 0.90);
    if (src.a > 0.0) col = blend_over(col, src);
  }

  // Pointer.
  if (uPointerUV.x >= 0.0 && uPointerUV.y >= 0.0) {
    float d = length(v_uv - uPointerUV);
    float ring = step(d, 0.018) * (1.0 - step(d, 0.012));
    float core = step(d, 0.006);
    if (ring > 0.0) col = blend_over(col, vec4(0.95, 0.95, 0.95, 0.85));
    if (core > 0.0) col = blend_over(col, vec4(0.35, 0.75, 0.95, 0.85));
  }

  FragColor = col;
}
)";

struct Quat {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float w = 1.0f;
};

struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

Vec3 vec_add(const Vec3& a, const Vec3& b) {
  return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}
Vec3 vec_sub(const Vec3& a, const Vec3& b) {
  return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec3 vec_mul(const Vec3& v, float s) {
  return Vec3{v.x * s, v.y * s, v.z * s};
}
float vec_dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
Vec3 vec_cross(const Vec3& a, const Vec3& b) {
  return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
float vec_len(const Vec3& v) {
  return std::sqrt(std::max(0.0f, vec_dot(v, v)));
}
Vec3 vec_norm(const Vec3& v) {
  const float l = vec_len(v);
  if (l <= 1e-6f) {
    return Vec3{0.0f, 0.0f, 0.0f};
  }
  return vec_mul(v, 1.0f / l);
}

Quat quat_mul(const Quat& a, const Quat& b) {
  Quat out;
  out.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
  out.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
  out.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
  out.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
  return out;
}

Quat quat_conjugate(const Quat& q) {
  return Quat{-q.x, -q.y, -q.z, q.w};
}

Quat quat_normalize(const Quat& q) {
  const float n2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  if (n2 <= 1e-12f) {
    return Quat{};
  }
  const float inv = 1.0f / std::sqrt(n2);
  return Quat{q.x * inv, q.y * inv, q.z * inv, q.w * inv};
}

Vec3 quat_rotate_vec3(const Quat& q_raw, const Vec3& v) {
  const Quat q = quat_normalize(q_raw);
  const Vec3 u{q.x, q.y, q.z};
  const float s = q.w;
  // v' = 2*dot(u,v)*u + (s^2 - dot(u,u))*v + 2*s*cross(u,v)
  const float uu = vec_dot(u, u);
  const float uv = vec_dot(u, v);
  const Vec3 term1 = vec_mul(u, 2.0f * uv);
  const Vec3 term2 = vec_mul(v, (s * s - uu));
  const Vec3 term3 = vec_mul(vec_cross(u, v), 2.0f * s);
  return vec_add(vec_add(term1, term2), term3);
}

Quat quat_from_yaw_pitch(float yaw_rad, float pitch_rad) {
  const float hy = 0.5f * yaw_rad;
  const float hp = 0.5f * pitch_rad;
  const float sy = std::sin(hy);
  const float cy = std::cos(hy);
  const float sp = std::sin(hp);
  const float cp = std::cos(hp);
  // Yaw about +Y, then pitch about +X (right-handed).
  const Quat qy{0.0f, sy, 0.0f, cy};
  const Quat qp{sp, 0.0f, 0.0f, cp};
  return quat_mul(qy, qp);
}

Quat quat_from_basis(const Vec3& x_axis, const Vec3& y_axis, const Vec3& z_axis) {
  // Rotation matrix with columns (x_axis, y_axis, z_axis).
  const float m00 = x_axis.x;
  const float m10 = x_axis.y;
  const float m20 = x_axis.z;
  const float m01 = y_axis.x;
  const float m11 = y_axis.y;
  const float m21 = y_axis.z;
  const float m02 = z_axis.x;
  const float m12 = z_axis.y;
  const float m22 = z_axis.z;

  Quat q;
  const float trace = m00 + m11 + m22;
  if (trace > 0.0f) {
    const float s = std::sqrt(trace + 1.0f) * 2.0f;
    q.w = 0.25f * s;
    q.x = (m21 - m12) / s;
    q.y = (m02 - m20) / s;
    q.z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const float s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
    q.w = (m21 - m12) / s;
    q.x = 0.25f * s;
    q.y = (m01 + m10) / s;
    q.z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const float s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
    q.w = (m02 - m20) / s;
    q.x = (m01 + m10) / s;
    q.y = 0.25f * s;
    q.z = (m12 + m21) / s;
  } else {
    const float s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
    q.w = (m10 - m01) / s;
    q.x = (m02 + m20) / s;
    q.y = (m12 + m21) / s;
    q.z = 0.25f * s;
  }
  return quat_normalize(q);
}

Quat quat_look_at_forward(const Vec3& forward, const Vec3& up_hint) {
  // forward is the desired direction of local -Z in world space.
  const Vec3 f = vec_norm(forward);
  if (vec_len(f) <= 1e-6f) {
    return Quat{};
  }
  const Vec3 z_axis = vec_mul(f, -1.0f);  // local +Z points opposite to forward(-Z).
  Vec3 x_axis = vec_cross(up_hint, z_axis);
  x_axis = vec_norm(x_axis);
  if (vec_len(x_axis) <= 1e-6f) {
    // up was parallel; pick another up.
    x_axis = vec_norm(vec_cross(Vec3{1.0f, 0.0f, 0.0f}, z_axis));
  }
  const Vec3 y_axis = vec_cross(z_axis, x_axis);
  return quat_from_basis(x_axis, y_axis, z_axis);
}

unsigned compile_shader(GLenum type, const char* source) {
  const GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  GLint ok = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (ok != GL_TRUE) {
    GLint len = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
    std::string info;
    if (len > 0) {
      info.resize(static_cast<std::size_t>(len));
      glGetShaderInfoLog(shader, len, nullptr, info.data());
    }
    spdlog::error("OpenXR shader compile failed: {}", info);
    glDeleteShader(shader);
    return 0;
  }
  return static_cast<unsigned>(shader);
}

unsigned link_program(unsigned vs, unsigned fs) {
  const GLuint program = glCreateProgram();
  glAttachShader(program, static_cast<GLuint>(vs));
  glAttachShader(program, static_cast<GLuint>(fs));
  glLinkProgram(program);
  GLint ok = GL_FALSE;
  glGetProgramiv(program, GL_LINK_STATUS, &ok);
  if (ok != GL_TRUE) {
    GLint len = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
    std::string info;
    if (len > 0) {
      info.resize(static_cast<std::size_t>(len));
      glGetProgramInfoLog(program, len, nullptr, info.data());
    }
    spdlog::error("OpenXR shader program link failed: {}", info);
    glDeleteProgram(program);
    return 0;
  }
  return static_cast<unsigned>(program);
}

std::string xr_result_string(XrResult r) {
  // Keep it simple: OpenXR result enums are stable enough to log as integers
  // unless xrResultToString is loaded (we don't depend on it).
  return std::to_string(static_cast<int>(r));
}

}  // namespace

struct OpenXrPresenter::Impl {
  SDL_SharedObject* loader = nullptr;
  PFN_xrGetInstanceProcAddr xrGetInstanceProcAddr = nullptr;

  PFN_xrCreateInstance xrCreateInstance = nullptr;
  PFN_xrDestroyInstance xrDestroyInstance = nullptr;
  PFN_xrGetSystem xrGetSystem = nullptr;
  PFN_xrCreateSession xrCreateSession = nullptr;
  PFN_xrDestroySession xrDestroySession = nullptr;
  PFN_xrCreateReferenceSpace xrCreateReferenceSpace = nullptr;
  PFN_xrDestroySpace xrDestroySpace = nullptr;
  PFN_xrEnumerateViewConfigurationViews xrEnumerateViewConfigurationViews = nullptr;
  PFN_xrEnumerateSwapchainFormats xrEnumerateSwapchainFormats = nullptr;
  PFN_xrCreateSwapchain xrCreateSwapchain = nullptr;
  PFN_xrDestroySwapchain xrDestroySwapchain = nullptr;
  PFN_xrEnumerateSwapchainImages xrEnumerateSwapchainImages = nullptr;
  PFN_xrAcquireSwapchainImage xrAcquireSwapchainImage = nullptr;
  PFN_xrWaitSwapchainImage xrWaitSwapchainImage = nullptr;
  PFN_xrReleaseSwapchainImage xrReleaseSwapchainImage = nullptr;
  PFN_xrPollEvent xrPollEvent = nullptr;
  PFN_xrBeginSession xrBeginSession = nullptr;
  PFN_xrEndSession xrEndSession = nullptr;
  PFN_xrWaitFrame xrWaitFrame = nullptr;
  PFN_xrBeginFrame xrBeginFrame = nullptr;
  PFN_xrEndFrame xrEndFrame = nullptr;
  PFN_xrLocateViews xrLocateViews = nullptr;
  PFN_xrLocateSpace xrLocateSpace = nullptr;

  PFN_xrStringToPath xrStringToPath = nullptr;
  PFN_xrCreateActionSet xrCreateActionSet = nullptr;
  PFN_xrDestroyActionSet xrDestroyActionSet = nullptr;
  PFN_xrCreateAction xrCreateAction = nullptr;
  PFN_xrSuggestInteractionProfileBindings xrSuggestInteractionProfileBindings = nullptr;
  PFN_xrAttachSessionActionSets xrAttachSessionActionSets = nullptr;
  PFN_xrCreateActionSpace xrCreateActionSpace = nullptr;
  PFN_xrSyncActions xrSyncActions = nullptr;
  PFN_xrGetActionStateBoolean xrGetActionStateBoolean = nullptr;
  PFN_xrGetActionStateFloat xrGetActionStateFloat = nullptr;
  PFN_xrGetActionStatePose xrGetActionStatePose = nullptr;

  PFN_xrEnumerateEnvironmentBlendModes xrEnumerateEnvironmentBlendModes = nullptr;

  PFN_xrGetOpenGLGraphicsRequirementsKHR xrGetOpenGLGraphicsRequirementsKHR = nullptr;

  SDL_Window* sdl_window = nullptr;
  SDL_GLContext gl_context = nullptr;

  XrInstance instance = XR_NULL_HANDLE;
  XrSystemId system_id = XR_NULL_SYSTEM_ID;
  XrSession session = XR_NULL_HANDLE;
  XrSpace reference_space = XR_NULL_HANDLE;

  XrSessionState session_state = XR_SESSION_STATE_UNKNOWN;
  bool session_running = false;
  bool exit_requested = false;

  struct Swapchain {
    XrSwapchain handle = XR_NULL_HANDLE;
    int32_t width = 0;
    int32_t height = 0;
    std::vector<XrSwapchainImageOpenGLKHR> images;
    std::vector<GLuint> fbos;
  };

  std::vector<XrViewConfigurationView> view_config_views;
  std::vector<XrView> views;
  std::vector<Swapchain> swapchains;
  Swapchain quad_swapchain;
  Swapchain hud_swapchain;
  XrEnvironmentBlendMode blend_mode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;

  // Action system (controllers).
  XrActionSet action_set = XR_NULL_HANDLE;
  XrPath hand_left = XR_NULL_PATH;
  XrPath hand_right = XR_NULL_PATH;
  XrAction aim_pose_action = XR_NULL_HANDLE;
  XrAction trigger_value_action = XR_NULL_HANDLE;
  XrAction trigger_click_action = XR_NULL_HANDLE;
  XrAction squeeze_value_action = XR_NULL_HANDLE;
  XrAction thumbstick_x_action = XR_NULL_HANDLE;
  XrAction thumbstick_y_action = XR_NULL_HANDLE;
  XrAction action_play_pause = XR_NULL_HANDLE;
  XrAction action_next = XR_NULL_HANDLE;
  XrAction action_prev = XR_NULL_HANDLE;
  XrAction action_cycle_projection = XR_NULL_HANDLE;
  XrSpace aim_space_left = XR_NULL_HANDLE;
  XrSpace aim_space_right = XR_NULL_HANDLE;

  bool prev_play_pause = false;
  bool prev_next = false;
  bool prev_prev = false;
  bool prev_cycle = false;
  bool prev_trigger_down = false;
  XrTime last_seek_step_time = 0;

  // Flat2D screen placement.
  XrPosef screen_pose{};
  float screen_width_m = 1.2f;
  float screen_distance_m = 1.6f;

  GLuint program = 0;
  GLuint vao = 0;
  GLint u_texture = -1;
  GLint u_mode = -1;
  GLint u_sbs_eye = -1;
  GLint u_fov_tan = -1;
  GLint u_orientation = -1;
  GLint u_video_aspect = -1;
  GLint u_target_aspect = -1;

  GLuint hud_program = 0;
  GLuint hud_vao = 0;
  GLint hud_u_progress = -1;
  GLint hud_u_playing = -1;
  GLint hud_u_proj_mode = -1;
  GLint hud_u_hover = -1;
  GLint hud_u_pointer_uv = -1;

  bool started = false;

  void unload_loader() {
    if (loader) {
      SDL_UnloadObject(loader);
      loader = nullptr;
    }
    xrGetInstanceProcAddr = nullptr;
  }

  template <typename T>
  bool load_global(const char* name, T* out, std::string& err) {
    if (!xrGetInstanceProcAddr) {
      err = "OpenXR loader missing xrGetInstanceProcAddr";
      return false;
    }
    PFN_xrVoidFunction fn = nullptr;
    const XrResult r = xrGetInstanceProcAddr(XR_NULL_HANDLE, name, &fn);
    if (XR_FAILED(r) || !fn) {
      err = std::string("OpenXR: failed to load ") + name + " result=" + xr_result_string(r);
      return false;
    }
    *out = reinterpret_cast<T>(fn);
    return true;
  }

  template <typename T>
  bool load_instance(const char* name, T* out, std::string& err) {
    if (!xrGetInstanceProcAddr || instance == XR_NULL_HANDLE) {
      err = "OpenXR instance not ready";
      return false;
    }
    PFN_xrVoidFunction fn = nullptr;
    const XrResult r = xrGetInstanceProcAddr(instance, name, &fn);
    if (XR_FAILED(r) || !fn) {
      err = std::string("OpenXR: failed to load ") + name + " result=" + xr_result_string(r);
      return false;
    }
    *out = reinterpret_cast<T>(fn);
    return true;
  }

  void destroy_gl_resources() {
    for (auto& sc : swapchains) {
      if (!sc.fbos.empty()) {
        glDeleteFramebuffers(static_cast<GLsizei>(sc.fbos.size()), sc.fbos.data());
        sc.fbos.clear();
      }
    }
    if (!quad_swapchain.fbos.empty()) {
      glDeleteFramebuffers(static_cast<GLsizei>(quad_swapchain.fbos.size()), quad_swapchain.fbos.data());
      quad_swapchain.fbos.clear();
    }
    if (!hud_swapchain.fbos.empty()) {
      glDeleteFramebuffers(static_cast<GLsizei>(hud_swapchain.fbos.size()), hud_swapchain.fbos.data());
      hud_swapchain.fbos.clear();
    }
    if (vao != 0) {
      glDeleteVertexArrays(1, &vao);
      vao = 0;
    }
    if (program != 0) {
      glDeleteProgram(program);
      program = 0;
    }
    if (hud_vao != 0) {
      glDeleteVertexArrays(1, &hud_vao);
      hud_vao = 0;
    }
    if (hud_program != 0) {
      glDeleteProgram(hud_program);
      hud_program = 0;
    }
  }

  void shutdown() {
    started = false;
    session_running = false;
    exit_requested = false;
    session_state = XR_SESSION_STATE_UNKNOWN;

    if (sdl_window && gl_context) {
      (void)SDL_GL_MakeCurrent(sdl_window, gl_context);
    }

    destroy_gl_resources();

    if (quad_swapchain.handle != XR_NULL_HANDLE && xrDestroySwapchain) {
      (void)xrDestroySwapchain(quad_swapchain.handle);
    }
    quad_swapchain.handle = XR_NULL_HANDLE;
    quad_swapchain.images.clear();
    quad_swapchain.fbos.clear();

    if (hud_swapchain.handle != XR_NULL_HANDLE && xrDestroySwapchain) {
      (void)xrDestroySwapchain(hud_swapchain.handle);
    }
    hud_swapchain.handle = XR_NULL_HANDLE;
    hud_swapchain.images.clear();
    hud_swapchain.fbos.clear();

    for (auto& sc : swapchains) {
      if (sc.handle != XR_NULL_HANDLE && xrDestroySwapchain) {
        (void)xrDestroySwapchain(sc.handle);
      }
      sc.handle = XR_NULL_HANDLE;
      sc.images.clear();
      sc.fbos.clear();
    }
    swapchains.clear();

    if (aim_space_left != XR_NULL_HANDLE && xrDestroySpace) {
      (void)xrDestroySpace(aim_space_left);
      aim_space_left = XR_NULL_HANDLE;
    }
    if (aim_space_right != XR_NULL_HANDLE && xrDestroySpace) {
      (void)xrDestroySpace(aim_space_right);
      aim_space_right = XR_NULL_HANDLE;
    }

    if (reference_space != XR_NULL_HANDLE && xrDestroySpace) {
      (void)xrDestroySpace(reference_space);
      reference_space = XR_NULL_HANDLE;
    }

    if (action_set != XR_NULL_HANDLE && xrDestroyActionSet) {
      (void)xrDestroyActionSet(action_set);
      action_set = XR_NULL_HANDLE;
    }
    hand_left = XR_NULL_PATH;
    hand_right = XR_NULL_PATH;
    aim_pose_action = XR_NULL_HANDLE;
    trigger_value_action = XR_NULL_HANDLE;
    trigger_click_action = XR_NULL_HANDLE;
    squeeze_value_action = XR_NULL_HANDLE;
    thumbstick_x_action = XR_NULL_HANDLE;
    thumbstick_y_action = XR_NULL_HANDLE;
    action_play_pause = XR_NULL_HANDLE;
    action_next = XR_NULL_HANDLE;
    action_prev = XR_NULL_HANDLE;
    action_cycle_projection = XR_NULL_HANDLE;
    prev_play_pause = false;
    prev_next = false;
    prev_prev = false;
    prev_cycle = false;
    last_seek_step_time = 0;

    if (session != XR_NULL_HANDLE && xrDestroySession) {
      (void)xrDestroySession(session);
      session = XR_NULL_HANDLE;
    }

    if (instance != XR_NULL_HANDLE && xrDestroyInstance) {
      (void)xrDestroyInstance(instance);
      instance = XR_NULL_HANDLE;
    }

    unload_loader();
  }
};

OpenXrPresenter::OpenXrPresenter() : impl_(new Impl()) {}

OpenXrPresenter::~OpenXrPresenter() {
  stop();
  delete impl_;
  impl_ = nullptr;
}

bool OpenXrPresenter::start(SDL_Window* sdl_window, SDL_GLContext gl_context, std::string& err) {
  if (!impl_) {
    err = "OpenXR presenter not initialized";
    return false;
  }
  if (impl_->started) {
    return true;
  }
  if (!sdl_window || !gl_context) {
    err = "OpenXR requires a valid SDL OpenGL window/context";
    return false;
  }

  impl_->sdl_window = sdl_window;
  impl_->gl_context = gl_context;
  if (!SDL_GL_MakeCurrent(sdl_window, gl_context)) {
    err = std::string("SDL_GL_MakeCurrent failed: ") + SDL_GetError();
    return false;
  }

  // Load OpenXR loader dynamically so we don't add a build-time dependency.
  std::array<const char*, 4> candidates{};
#if defined(_WIN32)
  candidates = {"openxr_loader.dll", "OpenXRLoader.dll", "openxr_loader", nullptr};
#else
  candidates = {"libopenxr_loader.so.1", "libopenxr_loader.so", "openxr_loader", nullptr};
#endif

  for (const char* name : candidates) {
    if (!name)
      continue;
    impl_->loader = SDL_LoadObject(name);
    if (impl_->loader)
      break;
  }
  if (!impl_->loader) {
    err = "OpenXR loader not found (expected openxr_loader.* on system PATH)";
    return false;
  }

  impl_->xrGetInstanceProcAddr =
      reinterpret_cast<PFN_xrGetInstanceProcAddr>(SDL_LoadFunction(impl_->loader, "xrGetInstanceProcAddr"));
  if (!impl_->xrGetInstanceProcAddr) {
    err = "OpenXR loader missing xrGetInstanceProcAddr export";
    impl_->unload_loader();
    return false;
  }

  if (!impl_->load_global("xrCreateInstance", &impl_->xrCreateInstance, err)) {
    impl_->shutdown();
    return false;
  }

  XrInstanceCreateInfo ci{XR_TYPE_INSTANCE_CREATE_INFO};
  std::strncpy(ci.applicationInfo.applicationName, "f8implayer", sizeof(ci.applicationInfo.applicationName) - 1);
  std::strncpy(ci.applicationInfo.engineName, "f8studio", sizeof(ci.applicationInfo.engineName) - 1);
  ci.applicationInfo.applicationVersion = 1;
  ci.applicationInfo.engineVersion = 1;
  ci.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;

  std::vector<const char*> exts;
  exts.push_back(XR_KHR_OPENGL_ENABLE_EXTENSION_NAME);
  ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
  ci.enabledExtensionNames = exts.data();

  const XrResult r_create = impl_->xrCreateInstance(&ci, &impl_->instance);
  if (XR_FAILED(r_create) || impl_->instance == XR_NULL_HANDLE) {
    err = "xrCreateInstance failed result=" + xr_result_string(r_create);
    impl_->shutdown();
    return false;
  }

	  if (!impl_->load_instance("xrDestroyInstance", &impl_->xrDestroyInstance, err) ||
	      !impl_->load_instance("xrGetSystem", &impl_->xrGetSystem, err) ||
	      !impl_->load_instance("xrCreateSession", &impl_->xrCreateSession, err) ||
	      !impl_->load_instance("xrDestroySession", &impl_->xrDestroySession, err) ||
	      !impl_->load_instance("xrCreateReferenceSpace", &impl_->xrCreateReferenceSpace, err) ||
	      !impl_->load_instance("xrDestroySpace", &impl_->xrDestroySpace, err) ||
	      !impl_->load_instance("xrEnumerateViewConfigurationViews", &impl_->xrEnumerateViewConfigurationViews, err) ||
	      !impl_->load_instance("xrEnumerateSwapchainFormats", &impl_->xrEnumerateSwapchainFormats, err) ||
	      !impl_->load_instance("xrCreateSwapchain", &impl_->xrCreateSwapchain, err) ||
	      !impl_->load_instance("xrDestroySwapchain", &impl_->xrDestroySwapchain, err) ||
	      !impl_->load_instance("xrEnumerateSwapchainImages", &impl_->xrEnumerateSwapchainImages, err) ||
	      !impl_->load_instance("xrAcquireSwapchainImage", &impl_->xrAcquireSwapchainImage, err) ||
	      !impl_->load_instance("xrWaitSwapchainImage", &impl_->xrWaitSwapchainImage, err) ||
	      !impl_->load_instance("xrReleaseSwapchainImage", &impl_->xrReleaseSwapchainImage, err) ||
	      !impl_->load_instance("xrPollEvent", &impl_->xrPollEvent, err) ||
	      !impl_->load_instance("xrBeginSession", &impl_->xrBeginSession, err) ||
	      !impl_->load_instance("xrEndSession", &impl_->xrEndSession, err) ||
	      !impl_->load_instance("xrWaitFrame", &impl_->xrWaitFrame, err) ||
	      !impl_->load_instance("xrBeginFrame", &impl_->xrBeginFrame, err) ||
	      !impl_->load_instance("xrEndFrame", &impl_->xrEndFrame, err) ||
	      !impl_->load_instance("xrLocateViews", &impl_->xrLocateViews, err) ||
	      !impl_->load_instance("xrLocateSpace", &impl_->xrLocateSpace, err) ||
	      !impl_->load_instance("xrStringToPath", &impl_->xrStringToPath, err) ||
	      !impl_->load_instance("xrCreateActionSet", &impl_->xrCreateActionSet, err) ||
	      !impl_->load_instance("xrDestroyActionSet", &impl_->xrDestroyActionSet, err) ||
	      !impl_->load_instance("xrCreateAction", &impl_->xrCreateAction, err) ||
	      !impl_->load_instance("xrSuggestInteractionProfileBindings", &impl_->xrSuggestInteractionProfileBindings, err) ||
	      !impl_->load_instance("xrAttachSessionActionSets", &impl_->xrAttachSessionActionSets, err) ||
	      !impl_->load_instance("xrCreateActionSpace", &impl_->xrCreateActionSpace, err) ||
	      !impl_->load_instance("xrSyncActions", &impl_->xrSyncActions, err) ||
	      !impl_->load_instance("xrGetActionStateBoolean", &impl_->xrGetActionStateBoolean, err) ||
	      !impl_->load_instance("xrGetActionStateFloat", &impl_->xrGetActionStateFloat, err) ||
	      !impl_->load_instance("xrGetActionStatePose", &impl_->xrGetActionStatePose, err) ||
	      !impl_->load_instance("xrEnumerateEnvironmentBlendModes", &impl_->xrEnumerateEnvironmentBlendModes, err) ||
	      !impl_->load_instance("xrGetOpenGLGraphicsRequirementsKHR", &impl_->xrGetOpenGLGraphicsRequirementsKHR, err)) {
	    impl_->shutdown();
	    return false;
	  }

  XrSystemGetInfo sgi{XR_TYPE_SYSTEM_GET_INFO};
  sgi.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
  const XrResult r_sys = impl_->xrGetSystem(impl_->instance, &sgi, &impl_->system_id);
  if (XR_FAILED(r_sys) || impl_->system_id == XR_NULL_SYSTEM_ID) {
    err = "xrGetSystem failed result=" + xr_result_string(r_sys);
    impl_->shutdown();
    return false;
  }

#if defined(_WIN32)
  // Validate OpenGL requirements.
  XrGraphicsRequirementsOpenGLKHR glreq{XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_KHR};
  const XrResult r_req = impl_->xrGetOpenGLGraphicsRequirementsKHR(impl_->instance, impl_->system_id, &glreq);
  if (XR_FAILED(r_req)) {
    err = "xrGetOpenGLGraphicsRequirementsKHR failed result=" + xr_result_string(r_req);
    impl_->shutdown();
    return false;
  }

  // Create session using the current WGL context.
  const HGLRC hglrc = wglGetCurrentContext();
  const HDC hdc = wglGetCurrentDC();
  if (!hglrc || !hdc) {
    err = "OpenXR requires a current WGL context (wglGetCurrentContext/DC returned null)";
    impl_->shutdown();
    return false;
  }

  XrGraphicsBindingOpenGLWin32KHR gb{XR_TYPE_GRAPHICS_BINDING_OPENGL_WIN32_KHR};
  gb.hDC = hdc;
  gb.hGLRC = hglrc;

  XrSessionCreateInfo sci{XR_TYPE_SESSION_CREATE_INFO};
  sci.next = &gb;
  sci.systemId = impl_->system_id;
  const XrResult r_sess = impl_->xrCreateSession(impl_->instance, &sci, &impl_->session);
  if (XR_FAILED(r_sess) || impl_->session == XR_NULL_HANDLE) {
    err = "xrCreateSession failed result=" + xr_result_string(r_sess);
    impl_->shutdown();
    return false;
  }
#else
  (void)gl_context;
  err = "OpenXR OpenGL binding not implemented for this platform (currently Windows-only)";
  impl_->shutdown();
  return false;
#endif

  XrReferenceSpaceCreateInfo rs{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
  rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
  rs.poseInReferenceSpace.orientation.w = 1.0f;
  const XrResult r_space = impl_->xrCreateReferenceSpace(impl_->session, &rs, &impl_->reference_space);
  if (XR_FAILED(r_space) || impl_->reference_space == XR_NULL_HANDLE) {
    err = "xrCreateReferenceSpace failed result=" + xr_result_string(r_space);
    impl_->shutdown();
    return false;
  }

  // Action set: controller input (Quest Touch, etc).
  if (impl_->xrStringToPath(impl_->instance, "/user/hand/left", &impl_->hand_left) != XR_SUCCESS ||
      impl_->xrStringToPath(impl_->instance, "/user/hand/right", &impl_->hand_right) != XR_SUCCESS) {
    err = "xrStringToPath failed for /user/hand/*";
    impl_->shutdown();
    return false;
  }

  {
    XrActionSetCreateInfo asci{XR_TYPE_ACTION_SET_CREATE_INFO};
    std::strncpy(asci.actionSetName, "f8implayer", sizeof(asci.actionSetName) - 1);
    std::strncpy(asci.localizedActionSetName, "f8implayer", sizeof(asci.localizedActionSetName) - 1);
    asci.priority = 0;
    const XrResult r_as = impl_->xrCreateActionSet(impl_->instance, &asci, &impl_->action_set);
    if (XR_FAILED(r_as) || impl_->action_set == XR_NULL_HANDLE) {
      err = "xrCreateActionSet failed result=" + xr_result_string(r_as);
      impl_->shutdown();
      return false;
    }
  }

  auto create_action = [&](XrActionType type, const char* name, const char* localized, uint32_t sub_count,
                           const XrPath* subs, XrAction* out) -> bool {
    XrActionCreateInfo aci{XR_TYPE_ACTION_CREATE_INFO};
    aci.actionType = type;
    std::strncpy(aci.actionName, name, sizeof(aci.actionName) - 1);
    std::strncpy(aci.localizedActionName, localized, sizeof(aci.localizedActionName) - 1);
    aci.countSubactionPaths = sub_count;
    aci.subactionPaths = subs;
    const XrResult r = impl_->xrCreateAction(impl_->action_set, &aci, out);
    if (XR_FAILED(r) || !out || *out == XR_NULL_HANDLE) {
      err = std::string("xrCreateAction failed name=") + (name ? name : "(null)") + " result=" + xr_result_string(r);
      return false;
    }
    return true;
  };

  const XrPath both_hands[2] = {impl_->hand_left, impl_->hand_right};
  const XrPath right_hand[1] = {impl_->hand_right};
  const XrPath left_hand[1] = {impl_->hand_left};

  if (!create_action(XR_ACTION_TYPE_POSE_INPUT, "aim_pose", "Aim Pose", 2, both_hands, &impl_->aim_pose_action) ||
      !create_action(XR_ACTION_TYPE_FLOAT_INPUT, "trigger_value", "Trigger Value", 1, right_hand,
                     &impl_->trigger_value_action) ||
      !create_action(XR_ACTION_TYPE_BOOLEAN_INPUT, "trigger_click", "Trigger Click", 1, right_hand,
                     &impl_->trigger_click_action) ||
      !create_action(XR_ACTION_TYPE_FLOAT_INPUT, "squeeze_value", "Squeeze Value", 1, right_hand,
                     &impl_->squeeze_value_action) ||
      !create_action(XR_ACTION_TYPE_FLOAT_INPUT, "thumbstick_x", "Thumbstick X", 1, right_hand,
                     &impl_->thumbstick_x_action) ||
      !create_action(XR_ACTION_TYPE_FLOAT_INPUT, "thumbstick_y", "Thumbstick Y", 1, right_hand,
                     &impl_->thumbstick_y_action) ||
      !create_action(XR_ACTION_TYPE_BOOLEAN_INPUT, "play_pause", "Play/Pause", 1, right_hand,
                     &impl_->action_play_pause) ||
      !create_action(XR_ACTION_TYPE_BOOLEAN_INPUT, "playlist_next", "Playlist Next", 1, right_hand,
                     &impl_->action_next) ||
      !create_action(XR_ACTION_TYPE_BOOLEAN_INPUT, "playlist_prev", "Playlist Prev", 1, left_hand,
                     &impl_->action_prev) ||
      !create_action(XR_ACTION_TYPE_BOOLEAN_INPUT, "cycle_projection", "Cycle Projection", 1, left_hand,
                     &impl_->action_cycle_projection)) {
    impl_->shutdown();
    return false;
  }

  auto to_path = [&](const char* s, XrPath& out) -> bool {
    const XrResult r = impl_->xrStringToPath(impl_->instance, s, &out);
    if (XR_FAILED(r) || out == XR_NULL_PATH) {
      spdlog::warn("OpenXR xrStringToPath failed for {} result={}", s ? s : "(null)", xr_result_string(r));
      return false;
    }
    return true;
  };

  auto suggest_bindings = [&](const char* profile_str, const std::vector<std::pair<XrAction, const char*>>& pairs) {
    XrPath profile = XR_NULL_PATH;
    if (!to_path(profile_str, profile))
      return;
    std::vector<XrActionSuggestedBinding> bindings;
    bindings.reserve(pairs.size());
    for (const auto& it : pairs) {
      XrPath p = XR_NULL_PATH;
      if (!to_path(it.second, p))
        continue;
      bindings.push_back(XrActionSuggestedBinding{it.first, p});
    }
    if (bindings.empty())
      return;
    XrInteractionProfileSuggestedBinding sb{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
    sb.interactionProfile = profile;
    sb.countSuggestedBindings = static_cast<uint32_t>(bindings.size());
    sb.suggestedBindings = bindings.data();
    const XrResult r = impl_->xrSuggestInteractionProfileBindings(impl_->instance, &sb);
    if (XR_FAILED(r)) {
      spdlog::warn("OpenXR suggest bindings failed profile={} result={}", profile_str ? profile_str : "(null)",
                   xr_result_string(r));
    }
  };

  suggest_bindings("/interaction_profiles/oculus/touch_controller",
                   {
                       {impl_->aim_pose_action, "/user/hand/left/input/aim/pose"},
                       {impl_->aim_pose_action, "/user/hand/right/input/aim/pose"},
                       {impl_->trigger_value_action, "/user/hand/right/input/trigger/value"},
                       {impl_->trigger_click_action, "/user/hand/right/input/trigger/click"},
                       {impl_->squeeze_value_action, "/user/hand/right/input/squeeze/value"},
                       {impl_->thumbstick_x_action, "/user/hand/right/input/thumbstick/x"},
                       {impl_->thumbstick_y_action, "/user/hand/right/input/thumbstick/y"},
                       {impl_->action_play_pause, "/user/hand/right/input/a/click"},
                       {impl_->action_next, "/user/hand/right/input/b/click"},
                       {impl_->action_prev, "/user/hand/left/input/x/click"},
                       {impl_->action_cycle_projection, "/user/hand/left/input/y/click"},
                   });
  suggest_bindings("/interaction_profiles/khr/simple_controller",
                   {
                       {impl_->aim_pose_action, "/user/hand/left/input/aim/pose"},
                       {impl_->aim_pose_action, "/user/hand/right/input/aim/pose"},
                       {impl_->trigger_click_action, "/user/hand/right/input/select/click"},
                       {impl_->action_cycle_projection, "/user/hand/left/input/menu/click"},
                   });

  {
    XrSessionActionSetsAttachInfo ainfo{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
    ainfo.countActionSets = 1;
    ainfo.actionSets = &impl_->action_set;
    const XrResult r_attach = impl_->xrAttachSessionActionSets(impl_->session, &ainfo);
    if (XR_FAILED(r_attach)) {
      err = "xrAttachSessionActionSets failed result=" + xr_result_string(r_attach);
      impl_->shutdown();
      return false;
    }
  }

  {
    XrActionSpaceCreateInfo asci{XR_TYPE_ACTION_SPACE_CREATE_INFO};
    asci.action = impl_->aim_pose_action;
    asci.poseInActionSpace.orientation.w = 1.0f;

    asci.subactionPath = impl_->hand_left;
    const XrResult r0 = impl_->xrCreateActionSpace(impl_->session, &asci, &impl_->aim_space_left);
    if (XR_FAILED(r0) || impl_->aim_space_left == XR_NULL_HANDLE) {
      err = "xrCreateActionSpace(left) failed result=" + xr_result_string(r0);
      impl_->shutdown();
      return false;
    }
    asci.subactionPath = impl_->hand_right;
    const XrResult r1 = impl_->xrCreateActionSpace(impl_->session, &asci, &impl_->aim_space_right);
    if (XR_FAILED(r1) || impl_->aim_space_right == XR_NULL_HANDLE) {
      err = "xrCreateActionSpace(right) failed result=" + xr_result_string(r1);
      impl_->shutdown();
      return false;
    }
  }

  {
    const Quat q = quat_look_at_forward(Vec3{0.0f, 0.0f, 1.0f}, Vec3{0.0f, 1.0f, 0.0f});
    impl_->screen_pose.orientation.x = q.x;
    impl_->screen_pose.orientation.y = q.y;
    impl_->screen_pose.orientation.z = q.z;
    impl_->screen_pose.orientation.w = q.w;
    impl_->screen_pose.position.x = 0.0f;
    impl_->screen_pose.position.y = 0.0f;
    impl_->screen_pose.position.z = -impl_->screen_distance_m;
  }

  uint32_t view_count = 0;
  const XrResult r_vc0 = impl_->xrEnumerateViewConfigurationViews(
      impl_->instance, impl_->system_id, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &view_count, nullptr);
  if (XR_FAILED(r_vc0) || view_count == 0) {
    err = "xrEnumerateViewConfigurationViews failed result=" + xr_result_string(r_vc0);
    impl_->shutdown();
    return false;
  }
  impl_->view_config_views.assign(view_count, XrViewConfigurationView{XR_TYPE_VIEW_CONFIGURATION_VIEW});
  const XrResult r_vc1 = impl_->xrEnumerateViewConfigurationViews(
      impl_->instance, impl_->system_id, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, view_count, &view_count,
      impl_->view_config_views.data());
  if (XR_FAILED(r_vc1)) {
    err = "xrEnumerateViewConfigurationViews (2) failed result=" + xr_result_string(r_vc1);
    impl_->shutdown();
    return false;
  }

  uint32_t fmt_count = 0;
  const XrResult r_f0 = impl_->xrEnumerateSwapchainFormats(impl_->session, 0, &fmt_count, nullptr);
  if (XR_FAILED(r_f0) || fmt_count == 0) {
    err = "xrEnumerateSwapchainFormats failed result=" + xr_result_string(r_f0);
    impl_->shutdown();
    return false;
  }
  std::vector<int64_t> formats(fmt_count, 0);
  const XrResult r_f1 = impl_->xrEnumerateSwapchainFormats(impl_->session, fmt_count, &fmt_count, formats.data());
  if (XR_FAILED(r_f1)) {
    err = "xrEnumerateSwapchainFormats (2) failed result=" + xr_result_string(r_f1);
    impl_->shutdown();
    return false;
  }

  const int64_t preferred_formats[] = {
      static_cast<int64_t>(GL_SRGB8_ALPHA8),
      static_cast<int64_t>(GL_RGBA8),
  };
  int64_t chosen_format = 0;
  for (int64_t pf : preferred_formats) {
    if (std::find(formats.begin(), formats.end(), pf) != formats.end()) {
      chosen_format = pf;
      break;
    }
  }
  if (chosen_format == 0) {
    // Fallback: just use the first format the runtime reports.
    chosen_format = formats.front();
  }

  impl_->swapchains.clear();
  impl_->swapchains.resize(view_count);
	  for (uint32_t i = 0; i < view_count; ++i) {
	    auto& sc = impl_->swapchains[i];
    sc.width = static_cast<int32_t>(impl_->view_config_views[i].recommendedImageRectWidth);
    sc.height = static_cast<int32_t>(impl_->view_config_views[i].recommendedImageRectHeight);

    XrSwapchainCreateInfo sci{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    sci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
    sci.format = chosen_format;
    sci.sampleCount = impl_->view_config_views[i].recommendedSwapchainSampleCount;
    sci.width = static_cast<uint32_t>(sc.width);
    sci.height = static_cast<uint32_t>(sc.height);
    sci.faceCount = 1;
    sci.arraySize = 1;
    sci.mipCount = 1;

    const XrResult r_sc = impl_->xrCreateSwapchain(impl_->session, &sci, &sc.handle);
    if (XR_FAILED(r_sc) || sc.handle == XR_NULL_HANDLE) {
      err = "xrCreateSwapchain failed result=" + xr_result_string(r_sc);
      impl_->shutdown();
      return false;
    }

    uint32_t img_count = 0;
    const XrResult r_i0 = impl_->xrEnumerateSwapchainImages(sc.handle, 0, &img_count, nullptr);
    if (XR_FAILED(r_i0) || img_count == 0) {
      err = "xrEnumerateSwapchainImages failed result=" + xr_result_string(r_i0);
      impl_->shutdown();
      return false;
    }
    sc.images.assign(img_count, XrSwapchainImageOpenGLKHR{XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR});
    const XrResult r_i1 = impl_->xrEnumerateSwapchainImages(
        sc.handle, img_count, &img_count, reinterpret_cast<XrSwapchainImageBaseHeader*>(sc.images.data()));
    if (XR_FAILED(r_i1)) {
      err = "xrEnumerateSwapchainImages (2) failed result=" + xr_result_string(r_i1);
      impl_->shutdown();
      return false;
    }

	    sc.fbos.assign(img_count, 0);
	    glGenFramebuffers(static_cast<GLsizei>(img_count), sc.fbos.data());
	  }

  // Flat2D quad swapchain (single surface used with XrCompositionLayerQuad).
  {
    auto& sc = impl_->quad_swapchain;
    sc.width = 1920;
    sc.height = 1080;
    XrSwapchainCreateInfo sci{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    sci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
    sci.format = chosen_format;
    sci.sampleCount = 1;
    sci.width = static_cast<uint32_t>(sc.width);
    sci.height = static_cast<uint32_t>(sc.height);
    sci.faceCount = 1;
    sci.arraySize = 1;
    sci.mipCount = 1;

    const XrResult r_sc = impl_->xrCreateSwapchain(impl_->session, &sci, &sc.handle);
    if (XR_FAILED(r_sc) || sc.handle == XR_NULL_HANDLE) {
      err = "xrCreateSwapchain(quad) failed result=" + xr_result_string(r_sc);
      impl_->shutdown();
      return false;
    }

    uint32_t img_count = 0;
    const XrResult r_i0 = impl_->xrEnumerateSwapchainImages(sc.handle, 0, &img_count, nullptr);
    if (XR_FAILED(r_i0) || img_count == 0) {
      err = "xrEnumerateSwapchainImages(quad) failed result=" + xr_result_string(r_i0);
      impl_->shutdown();
      return false;
    }
    sc.images.assign(img_count, XrSwapchainImageOpenGLKHR{XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR});
    const XrResult r_i1 = impl_->xrEnumerateSwapchainImages(
        sc.handle, img_count, &img_count, reinterpret_cast<XrSwapchainImageBaseHeader*>(sc.images.data()));
    if (XR_FAILED(r_i1)) {
      err = "xrEnumerateSwapchainImages(quad2) failed result=" + xr_result_string(r_i1);
      impl_->shutdown();
      return false;
    }
    sc.fbos.assign(img_count, 0);
    glGenFramebuffers(static_cast<GLsizei>(img_count), sc.fbos.data());
  }

  // HUD swapchain (small alpha-blended quad layer).
  {
    auto& sc = impl_->hud_swapchain;
    sc.width = 1024;
    sc.height = 256;
    XrSwapchainCreateInfo sci{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    sci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
    sci.format = chosen_format;
    sci.sampleCount = 1;
    sci.width = static_cast<uint32_t>(sc.width);
    sci.height = static_cast<uint32_t>(sc.height);
    sci.faceCount = 1;
    sci.arraySize = 1;
    sci.mipCount = 1;

    const XrResult r_sc = impl_->xrCreateSwapchain(impl_->session, &sci, &sc.handle);
    if (XR_FAILED(r_sc) || sc.handle == XR_NULL_HANDLE) {
      err = "xrCreateSwapchain(hud) failed result=" + xr_result_string(r_sc);
      impl_->shutdown();
      return false;
    }

    uint32_t img_count = 0;
    const XrResult r_i0 = impl_->xrEnumerateSwapchainImages(sc.handle, 0, &img_count, nullptr);
    if (XR_FAILED(r_i0) || img_count == 0) {
      err = "xrEnumerateSwapchainImages(hud) failed result=" + xr_result_string(r_i0);
      impl_->shutdown();
      return false;
    }
    sc.images.assign(img_count, XrSwapchainImageOpenGLKHR{XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR});
    const XrResult r_i1 = impl_->xrEnumerateSwapchainImages(
        sc.handle, img_count, &img_count, reinterpret_cast<XrSwapchainImageBaseHeader*>(sc.images.data()));
    if (XR_FAILED(r_i1)) {
      err = "xrEnumerateSwapchainImages(hud2) failed result=" + xr_result_string(r_i1);
      impl_->shutdown();
      return false;
    }
    sc.fbos.assign(img_count, 0);
    glGenFramebuffers(static_cast<GLsizei>(img_count), sc.fbos.data());
  }

  // Create simple full-screen renderer.
  const unsigned vs = compile_shader(GL_VERTEX_SHADER, kXrVertexShader);
  const unsigned fs = compile_shader(GL_FRAGMENT_SHADER, kXrFragmentShader);
  if (vs == 0 || fs == 0) {
    if (vs)
      glDeleteShader(static_cast<GLuint>(vs));
    if (fs)
      glDeleteShader(static_cast<GLuint>(fs));
    err = "OpenXR shader compilation failed";
    impl_->shutdown();
    return false;
  }
  impl_->program = link_program(vs, fs);
  glDeleteShader(static_cast<GLuint>(vs));
  glDeleteShader(static_cast<GLuint>(fs));
  if (impl_->program == 0) {
    err = "OpenXR shader link failed";
    impl_->shutdown();
    return false;
  }

  glGenVertexArrays(1, &impl_->vao);
  impl_->u_texture = glGetUniformLocation(impl_->program, "uTexture");
  impl_->u_mode = glGetUniformLocation(impl_->program, "uMode");
  impl_->u_sbs_eye = glGetUniformLocation(impl_->program, "uSbsEye");
  impl_->u_fov_tan = glGetUniformLocation(impl_->program, "uFovTan");
  impl_->u_orientation = glGetUniformLocation(impl_->program, "uOrientation");
  impl_->u_video_aspect = glGetUniformLocation(impl_->program, "uVideoAspect");
  impl_->u_target_aspect = glGetUniformLocation(impl_->program, "uTargetAspect");

  // HUD renderer.
  {
    const unsigned hvs = compile_shader(GL_VERTEX_SHADER, kHudVertexShader);
    const unsigned hfs = compile_shader(GL_FRAGMENT_SHADER, kHudFragmentShader);
    if (hvs == 0 || hfs == 0) {
      if (hvs)
        glDeleteShader(static_cast<GLuint>(hvs));
      if (hfs)
        glDeleteShader(static_cast<GLuint>(hfs));
      err = "OpenXR HUD shader compilation failed";
      impl_->shutdown();
      return false;
    }
    impl_->hud_program = link_program(hvs, hfs);
    glDeleteShader(static_cast<GLuint>(hvs));
    glDeleteShader(static_cast<GLuint>(hfs));
    if (impl_->hud_program == 0) {
      err = "OpenXR HUD shader link failed";
      impl_->shutdown();
      return false;
    }
    glGenVertexArrays(1, &impl_->hud_vao);
    impl_->hud_u_progress = glGetUniformLocation(impl_->hud_program, "uProgress01");
    impl_->hud_u_playing = glGetUniformLocation(impl_->hud_program, "uPlaying");
    impl_->hud_u_proj_mode = glGetUniformLocation(impl_->hud_program, "uProjectionMode");
    impl_->hud_u_hover = glGetUniformLocation(impl_->hud_program, "uHoverId");
    impl_->hud_u_pointer_uv = glGetUniformLocation(impl_->hud_program, "uPointerUV");
  }

  // Choose blend mode.
  uint32_t blend_count = 0;
  const XrResult r_b0 = impl_->xrEnumerateEnvironmentBlendModes(impl_->instance, impl_->system_id,
                                                               XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0,
                                                               &blend_count, nullptr);
  if (XR_SUCCEEDED(r_b0) && blend_count > 0) {
    std::vector<XrEnvironmentBlendMode> blends(blend_count);
    const XrResult r_b1 = impl_->xrEnumerateEnvironmentBlendModes(
        impl_->instance, impl_->system_id, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, blend_count, &blend_count,
        blends.data());
    if (XR_SUCCEEDED(r_b1) && !blends.empty()) {
      impl_->blend_mode = blends[0];
      for (const auto bm : blends) {
        if (bm == XR_ENVIRONMENT_BLEND_MODE_OPAQUE) {
          impl_->blend_mode = bm;
          break;
        }
      }
    }
  }

  impl_->views.assign(view_count, XrView{XR_TYPE_VIEW});
  impl_->session_state = XR_SESSION_STATE_UNKNOWN;
  impl_->session_running = false;
  impl_->exit_requested = false;

  impl_->started = true;
  spdlog::info("OpenXR started: views={} format={} swapchain={}x{}", view_count,
               static_cast<long long>(chosen_format), impl_->swapchains[0].width, impl_->swapchains[0].height);
  return true;
}

void OpenXrPresenter::stop() {
  if (!impl_)
    return;
  if (!impl_->started && !impl_->loader && impl_->instance == XR_NULL_HANDLE)
    return;
  impl_->shutdown();
}

bool OpenXrPresenter::renderFrame(const FrameParams& frame, Events* events, std::string& err) {
  if (!impl_ || !impl_->started) {
    err = "OpenXR not started";
    return false;
  }
  if (events) {
    *events = Events{};
  }
  if (impl_->exit_requested) {
    err = "OpenXR exit requested";
    return false;
  }

  if (!SDL_GL_MakeCurrent(impl_->sdl_window, impl_->gl_context)) {
    err = std::string("SDL_GL_MakeCurrent failed: ") + SDL_GetError();
    return false;
  }

  // Poll events.
  if (impl_->xrPollEvent) {
    XrEventDataBuffer ev{XR_TYPE_EVENT_DATA_BUFFER};
    while (true) {
      const XrResult r = impl_->xrPollEvent(impl_->instance, &ev);
      if (r == XR_EVENT_UNAVAILABLE)
        break;
      if (XR_FAILED(r)) {
        err = "xrPollEvent failed result=" + xr_result_string(r);
        return false;
      }
      if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
        const auto* s = reinterpret_cast<const XrEventDataSessionStateChanged*>(&ev);
        impl_->session_state = s->state;
        if (impl_->session_state == XR_SESSION_STATE_READY && !impl_->session_running) {
          XrSessionBeginInfo bi{XR_TYPE_SESSION_BEGIN_INFO};
          bi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
          const XrResult r_begin = impl_->xrBeginSession(impl_->session, &bi);
          if (XR_SUCCEEDED(r_begin)) {
            impl_->session_running = true;
            spdlog::info("OpenXR session begun");
          } else {
            err = "xrBeginSession failed result=" + xr_result_string(r_begin);
            return false;
          }
        } else if (impl_->session_state == XR_SESSION_STATE_STOPPING && impl_->session_running) {
          const XrResult r_end = impl_->xrEndSession(impl_->session);
          if (XR_SUCCEEDED(r_end)) {
            impl_->session_running = false;
            spdlog::info("OpenXR session ended");
          } else {
            err = "xrEndSession failed result=" + xr_result_string(r_end);
            return false;
          }
        } else if (impl_->session_state == XR_SESSION_STATE_EXITING ||
                   impl_->session_state == XR_SESSION_STATE_LOSS_PENDING) {
          impl_->exit_requested = true;
        }
      }
      ev = XrEventDataBuffer{XR_TYPE_EVENT_DATA_BUFFER};
    }
  }

  if (impl_->exit_requested) {
    err = "OpenXR exit requested";
    return false;
  }
  if (!impl_->session_running) {
    err.clear();
    return true;
  }

  XrFrameWaitInfo wi{XR_TYPE_FRAME_WAIT_INFO};
  XrFrameState fs{XR_TYPE_FRAME_STATE};
  const XrResult r_wait = impl_->xrWaitFrame(impl_->session, &wi, &fs);
  if (XR_FAILED(r_wait)) {
    err = "xrWaitFrame failed result=" + xr_result_string(r_wait);
    return false;
  }

  XrFrameBeginInfo bi{XR_TYPE_FRAME_BEGIN_INFO};
  const XrResult r_beg = impl_->xrBeginFrame(impl_->session, &bi);
  if (XR_FAILED(r_beg)) {
    err = "xrBeginFrame failed result=" + xr_result_string(r_beg);
    return false;
  }

  XrViewLocateInfo li{XR_TYPE_VIEW_LOCATE_INFO};
  li.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
  li.displayTime = fs.predictedDisplayTime;
  li.space = impl_->reference_space;

  XrViewState vs{XR_TYPE_VIEW_STATE};
  uint32_t view_count = 0;
  const XrResult r_loc = impl_->xrLocateViews(impl_->session, &li, &vs,
                                             static_cast<uint32_t>(impl_->views.size()), &view_count,
                                             impl_->views.data());
  if (XR_FAILED(r_loc)) {
    err = "xrLocateViews failed result=" + xr_result_string(r_loc);
    return false;
  }
  view_count = std::min<uint32_t>(view_count, static_cast<uint32_t>(impl_->swapchains.size()));

  // Sync actions and build one-shot events.
  auto get_bool = [&](XrAction action, XrPath subaction, bool& out_pressed) -> bool {
    if (action == XR_NULL_HANDLE)
      return false;
    XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
    gi.action = action;
    gi.subactionPath = subaction;
    XrActionStateBoolean st{XR_TYPE_ACTION_STATE_BOOLEAN};
    const XrResult r = impl_->xrGetActionStateBoolean(impl_->session, &gi, &st);
    if (XR_FAILED(r) || !st.isActive) {
      out_pressed = false;
      return false;
    }
    out_pressed = (st.currentState == XR_TRUE);
    return true;
  };

  auto get_float = [&](XrAction action, XrPath subaction, float& out_value) -> bool {
    if (action == XR_NULL_HANDLE)
      return false;
    XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
    gi.action = action;
    gi.subactionPath = subaction;
    XrActionStateFloat st{XR_TYPE_ACTION_STATE_FLOAT};
    const XrResult r = impl_->xrGetActionStateFloat(impl_->session, &gi, &st);
    if (XR_FAILED(r) || !st.isActive) {
      out_value = 0.0f;
      return false;
    }
    out_value = st.currentState;
    return true;
  };

  auto locate_aim_ray = [&](XrSpace aim_space, Vec3& out_origin, Vec3& out_dir) -> bool {
    if (aim_space == XR_NULL_HANDLE)
      return false;
    XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION};
    const XrResult r = impl_->xrLocateSpace(aim_space, impl_->reference_space, fs.predictedDisplayTime, &loc);
    if (XR_FAILED(r))
      return false;
    const XrSpaceLocationFlags ok_pos = XR_SPACE_LOCATION_POSITION_VALID_BIT;
    const XrSpaceLocationFlags ok_ori = XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
    if ((loc.locationFlags & ok_pos) == 0 || (loc.locationFlags & ok_ori) == 0)
      return false;
    out_origin = Vec3{loc.pose.position.x, loc.pose.position.y, loc.pose.position.z};
    const Quat q{loc.pose.orientation.x, loc.pose.orientation.y, loc.pose.orientation.z, loc.pose.orientation.w};
    out_dir = vec_norm(quat_rotate_vec3(q, Vec3{0.0f, 0.0f, -1.0f}));
    return vec_len(out_dir) > 1e-6f;
  };

  auto ray_intersect_quad = [&](const XrPosef& pose, const Vec3& ro, const Vec3& rd, float w_m, float h_m, float& out_u,
                                float& out_v) -> bool {
    const Vec3 sp{pose.position.x, pose.position.y, pose.position.z};
    const Quat sq{pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w};
    const Vec3 n = quat_rotate_vec3(sq, Vec3{0.0f, 0.0f, -1.0f});
    const float denom = vec_dot(rd, n);
    if (std::abs(denom) <= 1e-6f)
      return false;
    const float t = vec_dot(vec_sub(sp, ro), n) / denom;
    if (t <= 0.0f)
      return false;
    const Vec3 hit = vec_add(ro, vec_mul(rd, t));
    const Vec3 rel = vec_sub(hit, sp);
    const Quat inv = quat_conjugate(sq);
    const Vec3 local = quat_rotate_vec3(inv, rel);
    if (std::abs(local.x) > 0.5f * w_m || std::abs(local.y) > 0.5f * h_m)
      return false;
    out_u = (local.x / w_m) + 0.5f;
    out_v = 0.5f - (local.y / h_m);
    return out_u >= 0.0f && out_u <= 1.0f && out_v >= 0.0f && out_v <= 1.0f;
  };

  if (impl_->xrSyncActions && impl_->action_set != XR_NULL_HANDLE) {
    XrActiveActionSet aas{};
    aas.actionSet = impl_->action_set;
    aas.subactionPath = XR_NULL_PATH;
    XrActionsSyncInfo si{XR_TYPE_ACTIONS_SYNC_INFO};
    si.countActiveActionSets = 1;
    si.activeActionSets = &aas;
    (void)impl_->xrSyncActions(impl_->session, &si);
  }

  // Button mappings:
  // - Right A: play/pause
  // - Right B: next
  // - Left X: prev
  // - Left Y: cycle projection
  bool b_play_pause = false;
  bool b_next = false;
  bool b_prev = false;
  bool b_cycle = false;
  (void)get_bool(impl_->action_play_pause, impl_->hand_right, b_play_pause);
  (void)get_bool(impl_->action_next, impl_->hand_right, b_next);
  (void)get_bool(impl_->action_prev, impl_->hand_left, b_prev);
  (void)get_bool(impl_->action_cycle_projection, impl_->hand_left, b_cycle);

  if (events) {
    if (b_play_pause && !impl_->prev_play_pause)
      events->play_pause_pressed = true;
    if (b_next && !impl_->prev_next)
      events->playlist_next_pressed = true;
    if (b_prev && !impl_->prev_prev)
      events->playlist_prev_pressed = true;
    if (b_cycle && !impl_->prev_cycle)
      events->cycle_projection_pressed = true;
  }
  impl_->prev_play_pause = b_play_pause;
  impl_->prev_next = b_next;
  impl_->prev_prev = b_prev;
  impl_->prev_cycle = b_cycle;

  float trigger_value = 0.0f;
  bool trigger_click = false;
  float squeeze_value = 0.0f;
  float thumb_x = 0.0f;
  float thumb_y = 0.0f;
  (void)get_float(impl_->trigger_value_action, impl_->hand_right, trigger_value);
  (void)get_float(impl_->squeeze_value_action, impl_->hand_right, squeeze_value);
  (void)get_float(impl_->thumbstick_x_action, impl_->hand_right, thumb_x);
  (void)get_float(impl_->thumbstick_y_action, impl_->hand_right, thumb_y);
  (void)get_bool(impl_->trigger_click_action, impl_->hand_right, trigger_click);

  // Discrete seek with thumbstick X (rate-limited).
  if (events) {
    constexpr XrTime kSeekIntervalNs = 200 * 1000 * 1000;  // 200ms
    if (std::abs(thumb_x) > 0.85f && (impl_->last_seek_step_time == 0 || (fs.predictedDisplayTime - impl_->last_seek_step_time) > kSeekIntervalNs)) {
      events->seek_delta_seconds = (thumb_x > 0.0f) ? 10.0 : -10.0;
      impl_->last_seek_step_time = fs.predictedDisplayTime;
    }
  }

  // Flat2D screen grab (squeeze) and "progress bar drag" (trigger on bottom region).
  Vec3 aim_origin{};
  Vec3 aim_dir{};
  const bool have_aim = locate_aim_ray(impl_->aim_space_right, aim_origin, aim_dir);
  const bool have_head = view_count > 0 && !impl_->views.empty();
  const Vec3 head_pos{have_head ? impl_->views[0].pose.position.x : 0.0f, have_head ? impl_->views[0].pose.position.y : 0.0f,
                      have_head ? impl_->views[0].pose.position.z : 0.0f};

  const bool grabbing = squeeze_value > 0.70f;
  if (grabbing && have_aim) {
    impl_->screen_distance_m = std::clamp(impl_->screen_distance_m + thumb_y * 0.03f, 0.4f, 5.0f);
    const Vec3 screen_pos = vec_add(aim_origin, vec_mul(aim_dir, impl_->screen_distance_m));
    const Vec3 forward = vec_sub(head_pos, screen_pos);
    const Quat q = quat_look_at_forward(forward, Vec3{0.0f, 1.0f, 0.0f});
    impl_->screen_pose.position.x = screen_pos.x;
    impl_->screen_pose.position.y = screen_pos.y;
    impl_->screen_pose.position.z = screen_pos.z;
    impl_->screen_pose.orientation.x = q.x;
    impl_->screen_pose.orientation.y = q.y;
    impl_->screen_pose.orientation.z = q.z;
    impl_->screen_pose.orientation.w = q.w;
  }

  const float video_aspect = (frame.src_height > 0) ? (static_cast<float>(frame.src_width) / static_cast<float>(frame.src_height)) : (16.0f / 9.0f);
  const float screen_w_m = impl_->screen_width_m;
  const float screen_h_m = screen_w_m / std::max(1e-6f, video_aspect);

  const bool trigger_down = (trigger_click || trigger_value > 0.65f);

  // HUD pose and ray interaction.
  // IDs: 1=prev,2=play,3=next,4=proj,5=seek
  constexpr int kHoverNone = 0;
  constexpr int kHoverPrev = 1;
  constexpr int kHoverPlay = 2;
  constexpr int kHoverNext = 3;
  constexpr int kHoverProj = 4;
  constexpr int kHoverSeek = 5;

  // Must match the shader layout.
  constexpr float kBtnY0 = 0.18f;
  constexpr float kBtnY1 = 0.82f;
  constexpr float kPrevX0 = 0.03f;
  constexpr float kPrevX1 = 0.15f;
  constexpr float kPlayX0 = 0.18f;
  constexpr float kPlayX1 = 0.30f;
  constexpr float kNextX0 = 0.33f;
  constexpr float kNextX1 = 0.45f;
  constexpr float kProjX0 = 0.86f;
  constexpr float kProjX1 = 0.97f;
  constexpr float kSeekX0 = 0.03f;
  constexpr float kSeekX1 = 0.97f;
  constexpr float kSeekY0 = 0.05f;
  constexpr float kSeekY1 = 0.14f;

  const double progress01 =
      (frame.duration_seconds > 0.0) ? std::clamp(frame.position_seconds / frame.duration_seconds, 0.0, 1.0) : 0.0;
  const int proj_mode = (frame.mode == SdlVideoWindow::ProjectionMode::EquirectMono)
                            ? 1
                            : (frame.mode == SdlVideoWindow::ProjectionMode::EquirectSbs ? 2 : 0);

  XrPosef hud_pose{};
  float hud_w_m = 0.75f;
  float hud_h_m = 0.12f;
  if (frame.mode == SdlVideoWindow::ProjectionMode::Flat2D) {
    // Attach HUD to the screen quad (slightly below and closer to the user).
    hud_pose = impl_->screen_pose;
    const Quat sq{hud_pose.orientation.x, hud_pose.orientation.y, hud_pose.orientation.z, hud_pose.orientation.w};
    const float offset_y = (screen_h_m * 0.5f) + (hud_h_m * 0.5f) + 0.03f;
    const Vec3 off_local{0.0f, -offset_y, -0.02f};
    const Vec3 off_world = quat_rotate_vec3(sq, off_local);
    hud_pose.position.x += off_world.x;
    hud_pose.position.y += off_world.y;
    hud_pose.position.z += off_world.z;
    hud_w_m = std::min(0.95f * screen_w_m, hud_w_m);
  } else if (have_head) {
    const Quat hq{impl_->views[0].pose.orientation.x, impl_->views[0].pose.orientation.y, impl_->views[0].pose.orientation.z,
                  impl_->views[0].pose.orientation.w};
    const Vec3 forward = vec_norm(quat_rotate_vec3(hq, Vec3{0.0f, 0.0f, -1.0f}));
    const Vec3 up = vec_norm(quat_rotate_vec3(hq, Vec3{0.0f, 1.0f, 0.0f}));
    const Vec3 hud_pos = vec_add(vec_add(head_pos, vec_mul(forward, 1.15f)), vec_mul(up, -0.32f));
    const Vec3 to_head = vec_sub(head_pos, hud_pos);
    const Quat q = quat_look_at_forward(to_head, Vec3{0.0f, 1.0f, 0.0f});
    hud_pose.orientation.x = q.x;
    hud_pose.orientation.y = q.y;
    hud_pose.orientation.z = q.z;
    hud_pose.orientation.w = q.w;
    hud_pose.position.x = hud_pos.x;
    hud_pose.position.y = hud_pos.y;
    hud_pose.position.z = hud_pos.z;
  } else {
    hud_pose.orientation.w = 1.0f;
    hud_pose.position.z = -1.0f;
  }

  float hud_u = -1.0f;
  float hud_v = -1.0f;
  int hover_id = kHoverNone;
  if (have_aim) {
    float u = 0.0f;
    float v = 0.0f;
    if (ray_intersect_quad(hud_pose, aim_origin, aim_dir, hud_w_m, hud_h_m, u, v)) {
      hud_u = u;
      hud_v = v;
      auto in = [&](float x0, float y0, float x1, float y1) {
        return (u >= x0 && u <= x1 && v >= y0 && v <= y1);
      };
      if (in(kSeekX0, kSeekY0, kSeekX1, kSeekY1)) {
        hover_id = kHoverSeek;
      } else if (in(kPrevX0, kBtnY0, kPrevX1, kBtnY1)) {
        hover_id = kHoverPrev;
      } else if (in(kPlayX0, kBtnY0, kPlayX1, kBtnY1)) {
        hover_id = kHoverPlay;
      } else if (in(kNextX0, kBtnY0, kNextX1, kBtnY1)) {
        hover_id = kHoverNext;
      } else if (in(kProjX0, kBtnY0, kProjX1, kBtnY1)) {
        hover_id = kHoverProj;
      }
    }
  }

  const bool trigger_rising = trigger_down && !impl_->prev_trigger_down;
  impl_->prev_trigger_down = trigger_down;

  if (events) {
    if (trigger_down && hover_id == kHoverSeek) {
      const double frac = (kSeekX1 > kSeekX0) ? (static_cast<double>(hud_u - kSeekX0) / static_cast<double>(kSeekX1 - kSeekX0)) : 0.0;
      events->seek_absolute_valid = true;
      events->seek_absolute_fraction01 = std::clamp(frac, 0.0, 1.0);
    }
    if (trigger_rising) {
      if (hover_id == kHoverPrev)
        events->playlist_prev_pressed = true;
      else if (hover_id == kHoverNext)
        events->playlist_next_pressed = true;
      else if (hover_id == kHoverPlay)
        events->play_pause_pressed = true;
      else if (hover_id == kHoverProj)
        events->cycle_projection_pressed = true;
    }
  }

  // Render either a projection layer (360) or a quad layer (flat).
  const bool have_texture = frame.src_texture != 0;
  const int mode = (frame.mode == SdlVideoWindow::ProjectionMode::EquirectMono)
                       ? 1
                       : (frame.mode == SdlVideoWindow::ProjectionMode::EquirectSbs ? 2 : 0);

  std::vector<XrCompositionLayerProjectionView> layer_views;
  layer_views.resize(view_count);

  XrCompositionLayerQuad quad_layer{XR_TYPE_COMPOSITION_LAYER_QUAD};
  XrCompositionLayerProjection proj_layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
  XrCompositionLayerQuad hud_layer{XR_TYPE_COMPOSITION_LAYER_QUAD};
  const XrCompositionLayerBaseHeader* layers[2] = {nullptr, nullptr};
  uint32_t layer_count = 0;

  if (frame.mode == SdlVideoWindow::ProjectionMode::Flat2D) {
    auto& sc = impl_->quad_swapchain;
    uint32_t index = 0;
    XrSwapchainImageAcquireInfo ai{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    const XrResult r_acq = impl_->xrAcquireSwapchainImage(sc.handle, &ai, &index);
    if (XR_FAILED(r_acq)) {
      err = "xrAcquireSwapchainImage(quad) failed result=" + xr_result_string(r_acq);
      return false;
    }
    XrSwapchainImageWaitInfo swi{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    swi.timeout = XR_INFINITE_DURATION;
    const XrResult r_wi = impl_->xrWaitSwapchainImage(sc.handle, &swi);
    if (XR_FAILED(r_wi)) {
      err = "xrWaitSwapchainImage(quad) failed result=" + xr_result_string(r_wi);
      return false;
    }

    const GLuint tex = sc.images[index].image;
    const GLuint fbo = sc.fbos[index];
    glUseProgram(impl_->program);
    glBindVertexArray(impl_->vao);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glViewport(0, 0, sc.width, sc.height);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    if (have_texture) {
      if (impl_->u_mode >= 0)
        glUniform1i(impl_->u_mode, 0);
      if (impl_->u_sbs_eye >= 0)
        glUniform1i(impl_->u_sbs_eye, 0);
      if (impl_->u_video_aspect >= 0)
        glUniform1f(impl_->u_video_aspect, video_aspect);
      const float target_aspect = (sc.height > 0) ? (static_cast<float>(sc.width) / static_cast<float>(sc.height)) : 1.0f;
      if (impl_->u_target_aspect >= 0)
        glUniform1f(impl_->u_target_aspect, target_aspect);

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(frame.src_texture));
      if (impl_->u_texture >= 0)
        glUniform1i(impl_->u_texture, 0);

      glDisable(GL_BLEND);
      glDisable(GL_DEPTH_TEST);
      glDisable(GL_CULL_FACE);
      glDrawArrays(GL_TRIANGLES, 0, 3);

      glBindTexture(GL_TEXTURE_2D, 0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);

    XrSwapchainImageReleaseInfo ri{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    const XrResult r_rel = impl_->xrReleaseSwapchainImage(sc.handle, &ri);
    if (XR_FAILED(r_rel)) {
      err = "xrReleaseSwapchainImage(quad) failed result=" + xr_result_string(r_rel);
      return false;
    }

    quad_layer.space = impl_->reference_space;
    quad_layer.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
    quad_layer.pose = impl_->screen_pose;
    quad_layer.size = XrExtent2Df{screen_w_m, screen_h_m};
    quad_layer.subImage.swapchain = sc.handle;
    quad_layer.subImage.imageRect.offset = {0, 0};
    quad_layer.subImage.imageRect.extent = {sc.width, sc.height};
    quad_layer.subImage.imageArrayIndex = 0;

    quad_layer.layerFlags = 0;
    layers[0] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&quad_layer);
    layer_count = 1;
  } else {
    if (have_texture) {
      glUseProgram(impl_->program);
      glBindVertexArray(impl_->vao);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(frame.src_texture));
      if (impl_->u_texture >= 0)
        glUniform1i(impl_->u_texture, 0);
      if (impl_->u_mode >= 0)
        glUniform1i(impl_->u_mode, mode);
      if (impl_->u_video_aspect >= 0)
        glUniform1f(impl_->u_video_aspect, video_aspect);
    }

    for (uint32_t i = 0; i < view_count; ++i) {
      auto& sc = impl_->swapchains[i];
      uint32_t index = 0;
      XrSwapchainImageAcquireInfo ai{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
      const XrResult r_acq = impl_->xrAcquireSwapchainImage(sc.handle, &ai, &index);
      if (XR_FAILED(r_acq)) {
        err = "xrAcquireSwapchainImage failed result=" + xr_result_string(r_acq);
        return false;
      }
      XrSwapchainImageWaitInfo swi{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
      swi.timeout = XR_INFINITE_DURATION;
      const XrResult r_wi = impl_->xrWaitSwapchainImage(sc.handle, &swi);
      if (XR_FAILED(r_wi)) {
        err = "xrWaitSwapchainImage failed result=" + xr_result_string(r_wi);
        return false;
      }

      const GLuint tex = sc.images[index].image;
      const GLuint fbo = sc.fbos[index];
      glBindFramebuffer(GL_FRAMEBUFFER, fbo);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
      glViewport(0, 0, sc.width, sc.height);
      glClearColor(0.f, 0.f, 0.f, 1.f);
      glClear(GL_COLOR_BUFFER_BIT);

      const float target_aspect = (sc.height > 0) ? (static_cast<float>(sc.width) / static_cast<float>(sc.height)) : 1.0f;
      if (have_texture) {
        if (impl_->u_target_aspect >= 0)
          glUniform1f(impl_->u_target_aspect, target_aspect);
      }

      // Per-eye SBS selection.
      const int sbs_eye = (frame.mode == SdlVideoWindow::ProjectionMode::EquirectSbs) ? static_cast<int>(i % 2) : 0;
      if (have_texture) {
        if (impl_->u_sbs_eye >= 0)
          glUniform1i(impl_->u_sbs_eye, sbs_eye);
      }

      // Combine headset orientation with manual offsets.
      Quat q_view{impl_->views[i].pose.orientation.x, impl_->views[i].pose.orientation.y, impl_->views[i].pose.orientation.z,
                  impl_->views[i].pose.orientation.w};
      const float yaw_rad = frame.yaw_offset_deg * (kPi / 180.0f);
      const float pitch_rad = frame.pitch_offset_deg * (kPi / 180.0f);
      const Quat q_off = quat_from_yaw_pitch(yaw_rad, pitch_rad);
      const Quat q = quat_mul(q_view, q_off);
      if (have_texture) {
        if (impl_->u_orientation >= 0)
          glUniform4f(impl_->u_orientation, q.x, q.y, q.z, q.w);
      }

      const XrFovf& fov = impl_->views[i].fov;
      const float tl = std::tan(fov.angleLeft);
      const float tr = std::tan(fov.angleRight);
      const float td = std::tan(fov.angleDown);
      const float tu = std::tan(fov.angleUp);
      if (have_texture) {
        if (impl_->u_fov_tan >= 0)
          glUniform4f(impl_->u_fov_tan, tl, tr, td, tu);
      }

      if (have_texture) {
        glDisable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glDrawArrays(GL_TRIANGLES, 0, 3);
      }

      glBindFramebuffer(GL_FRAMEBUFFER, 0);

      XrSwapchainImageReleaseInfo ri{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
      const XrResult r_rel = impl_->xrReleaseSwapchainImage(sc.handle, &ri);
      if (XR_FAILED(r_rel)) {
        err = "xrReleaseSwapchainImage failed result=" + xr_result_string(r_rel);
        return false;
      }

      XrCompositionLayerProjectionView pv{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
      pv.pose = impl_->views[i].pose;
      pv.fov = impl_->views[i].fov;
      pv.subImage.swapchain = sc.handle;
      pv.subImage.imageRect.offset = {0, 0};
      pv.subImage.imageRect.extent = {sc.width, sc.height};
      pv.subImage.imageArrayIndex = 0;
      layer_views[i] = pv;
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
    glUseProgram(0);

    proj_layer.layerFlags = 0;
    proj_layer.space = impl_->reference_space;
    proj_layer.viewCount = static_cast<uint32_t>(layer_views.size());
    proj_layer.views = layer_views.data();

    layers[0] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&proj_layer);
    layer_count = 1;
  }

  // Render HUD layer (always-on; alpha blended).
  {
    auto& sc = impl_->hud_swapchain;
    uint32_t index = 0;
    XrSwapchainImageAcquireInfo ai{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    const XrResult r_acq = impl_->xrAcquireSwapchainImage(sc.handle, &ai, &index);
    if (XR_FAILED(r_acq)) {
      err = "xrAcquireSwapchainImage(hud) failed result=" + xr_result_string(r_acq);
      return false;
    }
    XrSwapchainImageWaitInfo swi{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    swi.timeout = XR_INFINITE_DURATION;
    const XrResult r_wi = impl_->xrWaitSwapchainImage(sc.handle, &swi);
    if (XR_FAILED(r_wi)) {
      err = "xrWaitSwapchainImage(hud) failed result=" + xr_result_string(r_wi);
      return false;
    }

    const GLuint tex = sc.images[index].image;
    const GLuint fbo = sc.fbos[index];
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glViewport(0, 0, sc.width, sc.height);
    glClearColor(0.f, 0.f, 0.f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(impl_->hud_program);
    glBindVertexArray(impl_->hud_vao);

    if (impl_->hud_u_progress >= 0)
      glUniform1f(impl_->hud_u_progress, static_cast<float>(progress01));
    if (impl_->hud_u_playing >= 0)
      glUniform1i(impl_->hud_u_playing, frame.playing ? 1 : 0);
    if (impl_->hud_u_proj_mode >= 0)
      glUniform1i(impl_->hud_u_proj_mode, proj_mode);
    if (impl_->hud_u_hover >= 0)
      glUniform1i(impl_->hud_u_hover, hover_id);
    if (impl_->hud_u_pointer_uv >= 0)
      glUniform2f(impl_->hud_u_pointer_uv, hud_u, hud_v);

    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindVertexArray(0);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    XrSwapchainImageReleaseInfo ri{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    const XrResult r_rel = impl_->xrReleaseSwapchainImage(sc.handle, &ri);
    if (XR_FAILED(r_rel)) {
      err = "xrReleaseSwapchainImage(hud) failed result=" + xr_result_string(r_rel);
      return false;
    }

    hud_layer.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
    hud_layer.space = impl_->reference_space;
    hud_layer.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
    hud_layer.pose = hud_pose;
    hud_layer.size = XrExtent2Df{hud_w_m, hud_h_m};
    hud_layer.subImage.swapchain = sc.handle;
    hud_layer.subImage.imageRect.offset = {0, 0};
    hud_layer.subImage.imageRect.extent = {sc.width, sc.height};
    hud_layer.subImage.imageArrayIndex = 0;
  }

  if (layer_count < 2) {
    layers[layer_count] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&hud_layer);
    layer_count += 1;
  }

  XrFrameEndInfo ei{XR_TYPE_FRAME_END_INFO};
  ei.displayTime = fs.predictedDisplayTime;
  ei.environmentBlendMode = impl_->blend_mode;
  if (fs.shouldRender) {
    ei.layerCount = layer_count;
    ei.layers = (layer_count > 0) ? layers : nullptr;
  } else {
    ei.layerCount = 0;
    ei.layers = nullptr;
  }

  const XrResult r_end = impl_->xrEndFrame(impl_->session, &ei);
  if (XR_FAILED(r_end)) {
    err = "xrEndFrame failed result=" + xr_result_string(r_end);
    return false;
  }
  err.clear();
  return true;
}

bool OpenXrPresenter::started() const {
  return impl_ && impl_->started;
}

bool OpenXrPresenter::exitRequested() const {
  return impl_ && impl_->exit_requested;
}

}  // namespace f8::implayer
