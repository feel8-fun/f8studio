#include <cstdlib>
#include <stdexcept>
#include "tracking_service.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <thread>
#include <system_error>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

#include "f8cppsdk/describe_schema.h"
#include "f8cppsdk/time_utils.h"
#include "../common/service_runtime_utils.h"

namespace f8::cvkit::tracking {

using json = nlohmann::json;
using f8::cppsdk::describe::schema_array;
using f8::cppsdk::describe::schema_boolean;
using f8::cppsdk::describe::schema_integer;
using f8::cppsdk::describe::schema_number;
using f8::cppsdk::describe::schema_object;
using f8::cppsdk::describe::schema_string;
using f8::cppsdk::describe::schema_string_enum;
using f8::cppsdk::describe::state_field;
using f8::cppsdk::describe::video_frame_port;

namespace {

namespace fs = std::filesystem;

constexpr std::int64_t kModelDownloadRetryCooldownMs = 30000;
constexpr long kModelDownloadTimeoutSeconds = 300;
constexpr std::int64_t kInitVideoReadLogDelayMs = 1000;
constexpr std::int64_t kInitVideoReadErrorDelayMs = 15000;
constexpr std::int64_t kInitVideoReadLogIntervalMs = 5000;
constexpr std::chrono::milliseconds kInitVideoReadTimeout{100};

bool json_number_to_int(const json& v, int& out) {
  return service_runtime::parse_json_int(v, out);
}

bool json_number_to_double(const json& v, double& out) {
  return service_runtime::parse_json_double(v, out);
}

std::optional<double> extract_score_from_object(const json& obj) {
  if (!obj.is_object())
    return std::nullopt;
  static const std::array<const char*, 5> kScoreKeys = {"score", "conf", "confidence", "probability", "prob"};
  for (const char* key : kScoreKeys) {
    if (!obj.contains(key))
      continue;
    double s = 0.0;
    if (json_number_to_double(obj.at(key), s))
      return s;
  }
  return std::nullopt;
}

bool rect_from_xywh_values(const json& x_v, const json& y_v, const json& w_v, const json& h_v, cv::Rect& out) {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  if (!json_number_to_int(x_v, x) || !json_number_to_int(y_v, y) || !json_number_to_int(w_v, w) ||
      !json_number_to_int(h_v, h)) {
    return false;
  }
  if (w <= 0 || h <= 0)
    return false;
  out = cv::Rect(x, y, w, h);
  return true;
}

bool rect_from_xyxy_values(const json& x1_v, const json& y1_v, const json& x2_v, const json& y2_v, cv::Rect& out) {
  int x1 = 0;
  int y1 = 0;
  int x2 = 0;
  int y2 = 0;
  if (!json_number_to_int(x1_v, x1) || !json_number_to_int(y1_v, y1) || !json_number_to_int(x2_v, x2) ||
      !json_number_to_int(y2_v, y2)) {
    return false;
  }
  const int w = x2 - x1;
  const int h = y2 - y1;
  if (w <= 0 || h <= 0)
    return false;
  out = cv::Rect(x1, y1, w, h);
  return true;
}

bool try_extract_candidate_from_object(const json& obj, TrackingInitCandidate& out) {
  if (!obj.is_object())
    return false;

  std::optional<double> score = extract_score_from_object(obj);
  cv::Rect rect;
  if (obj.contains("x") && obj.contains("y") && obj.contains("w") && obj.contains("h")) {
    if (rect_from_xywh_values(obj.at("x"), obj.at("y"), obj.at("w"), obj.at("h"), rect)) {
      out.bbox = rect;
      out.score = score;
      return true;
    }
  }
  if (obj.contains("x1") && obj.contains("y1") && obj.contains("x2") && obj.contains("y2")) {
    if (rect_from_xyxy_values(obj.at("x1"), obj.at("y1"), obj.at("x2"), obj.at("y2"), rect)) {
      out.bbox = rect;
      out.score = score;
      return true;
    }
  }
  if (obj.contains("left") && obj.contains("top") && obj.contains("right") && obj.contains("bottom")) {
    if (rect_from_xyxy_values(obj.at("left"), obj.at("top"), obj.at("right"), obj.at("bottom"), rect)) {
      out.bbox = rect;
      out.score = score;
      return true;
    }
  }
  if (obj.contains("bbox")) {
    const json& bbox = obj.at("bbox");
    if (bbox.is_array() && bbox.size() >= 4) {
      if (rect_from_xyxy_values(bbox.at(0), bbox.at(1), bbox.at(2), bbox.at(3), rect)) {
        out.bbox = rect;
        out.score = score;
        return true;
      }
    }
    if (bbox.is_object()) {
      if (try_extract_candidate_from_object(bbox, out)) {
        if (score.has_value())
          out.score = score;
        return true;
      }
    }
  }
  return false;
}

void collect_bbox_candidates(const json& root, std::vector<TrackingInitCandidate>& out, int depth) {
  if (depth > 24 || out.size() >= 256)
    return;

  TrackingInitCandidate candidate;
  if (try_extract_candidate_from_object(root, candidate)) {
    out.push_back(candidate);
    if (out.size() >= 256)
      return;
  }

  if (root.is_array()) {
    for (const auto& item : root) {
      collect_bbox_candidates(item, out, depth + 1);
      if (out.size() >= 256)
        return;
    }
    return;
  }
  if (root.is_object()) {
    for (auto it = root.begin(); it != root.end(); ++it) {
      collect_bbox_candidates(it.value(), out, depth + 1);
      if (out.size() >= 256)
        return;
    }
  }
}

TrackingInitSelectMode parse_init_select_mode(const std::string& raw, std::string& normalized, bool& ok) {
  const std::string s = service_runtime::to_lower_ascii_copy(service_runtime::trim_copy(raw));
  if (s == "first_box" || s == "first" || s == "ordered_first") {
    normalized = "first_box";
    ok = true;
    return TrackingInitSelectMode::FirstBox;
  }
  if (s.empty() || s == "closest_center" || s == "closest" || s == "center") {
    normalized = "closest_center";
    ok = true;
    return TrackingInitSelectMode::ClosestCenter;
  }
  if (s == "largest_area" || s == "largest" || s == "area") {
    normalized = "largest_area";
    ok = true;
    return TrackingInitSelectMode::LargestArea;
  }
  if (s == "highest_score" || s == "score") {
    normalized = "highest_score";
    ok = true;
    return TrackingInitSelectMode::HighestScore;
  }
  normalized = "closest_center";
  ok = false;
  return TrackingInitSelectMode::ClosestCenter;
}

TrackerKind parse_tracker_kind(const std::string& raw, std::string& normalized, bool& ok) {
  const std::string s = service_runtime::to_lower_ascii_copy(service_runtime::trim_copy(raw));
  if (s.empty() || s == "csrt") {
    normalized = "csrt";
    ok = true;
    return TrackerKind::Csrt;
  }
  if (s == "kcf") {
    normalized = "kcf";
    ok = true;
    return TrackerKind::Kcf;
  }
  if (s == "mil") {
    normalized = "mil";
    ok = true;
    return TrackerKind::Mil;
  }
  if (s == "nano" || s == "nanotrack") {
    normalized = "nano";
    ok = true;
    return TrackerKind::Nano;
  }
  if (s == "vit" || s == "vittrack") {
    normalized = "vit";
    ok = true;
    return TrackerKind::Vit;
  }
  normalized = "csrt";
  ok = false;
  return TrackerKind::Csrt;
}

std::string tracker_kind_to_string(TrackerKind kind) {
  if (kind == TrackerKind::Csrt) {
    return "csrt";
  }
  if (kind == TrackerKind::Kcf) {
    return "kcf";
  }
  if (kind == TrackerKind::Mil) {
    return "mil";
  }
  if (kind == TrackerKind::Nano) {
    return "nano";
  }
  return "vit";
}

bool tracker_kind_uses_model_files(TrackerKind kind) {
  return kind == TrackerKind::Nano || kind == TrackerKind::Vit;
}

std::string default_model_dir_state() {
  return "";
}

fs::path default_model_dir_path() {
  const char* configured = std::getenv("F8_MODEL_ROOT");
  if (configured != nullptr && configured[0] != '\0') {
    const fs::path root(configured);
    if (!root.is_absolute()) {
      throw std::invalid_argument("F8_MODEL_ROOT must be absolute");
    }
    return root / "tracking";
  }
#ifdef _WIN32
  const char* base = std::getenv("LOCALAPPDATA");
  if (base != nullptr && base[0] != '\0') return fs::path(base) / "f8studio" / "models" / "tracking";
#else
  const char* home = std::getenv("HOME");
#ifdef __APPLE__
  if (home != nullptr && home[0] != '\0') return fs::path(home) / "Library" / "Application Support" / "f8studio" / "models" / "tracking";
#else
  const char* data = std::getenv("XDG_DATA_HOME");
  if (data != nullptr && data[0] != '\0') return fs::path(data) / "f8studio" / "models" / "tracking";
  if (home != nullptr && home[0] != '\0') return fs::path(home) / ".local" / "share" / "f8studio" / "models" / "tracking";
#endif
#endif
  throw std::runtime_error("Cannot resolve model storage; set F8_MODEL_ROOT");
}

fs::path resolve_model_dir_path(const std::string& raw) {
  const std::string trimmed = service_runtime::trim_copy(raw);
  if (trimmed.empty()) {
    return default_model_dir_path();
  }
  const fs::path candidate(trimmed);
  if (candidate.is_absolute()) {
    return candidate.lexically_normal();
  }
  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  if (ec) {
    return candidate.lexically_normal();
  }
  return (cwd / candidate).lexically_normal();
}

std::string normalize_model_dir_state(const std::string& raw) {
  const std::string trimmed = service_runtime::trim_copy(raw);
  if (trimmed.empty()) {
    return default_model_dir_state();
  }
  return trimmed;
}

bool file_exists_nonempty(const fs::path& path) {
  std::error_code ec;
  if (!fs::exists(path, ec) || ec) {
    return false;
  }
  const auto size = fs::file_size(path, ec);
  if (ec) {
    return false;
  }
  return size > 0;
}

void remove_path_if_exists(const fs::path& path) {
  std::error_code ec;
  if (fs::is_directory(path, ec) && !ec) {
    fs::remove_all(path, ec);
    return;
  }
  ec.clear();
  fs::remove(path, ec);
}

std::string quote_shell_arg(const std::string& value) {
#ifdef _WIN32
  std::string out = "\"";
  for (char ch : value) {
    if (ch == '"' || ch == '\\') {
      out.push_back('\\');
    }
    out.push_back(ch);
  }
  out.push_back('"');
  return out;
#else
  std::string out = "'";
  for (char ch : value) {
    if (ch == '\'') {
      out += "'\\''";
      continue;
    }
    out.push_back(ch);
  }
  out.push_back('\'');
  return out;
#endif
}

bool run_command(const std::string& command, std::string& error_message) {
  const int rc = std::system(command.c_str());
  if (rc == 0) {
    return true;
  }
  error_message = "command failed rc=" + std::to_string(rc) + ": " + command;
  return false;
}

bool acquire_lock_dir(const fs::path& lock_dir, std::int64_t timeout_ms, std::string& error_message) {
  const auto start = std::chrono::steady_clock::now();
  while (true) {
    std::error_code ec;
    if (fs::create_directory(lock_dir, ec)) {
      return true;
    }
    if (ec && ec.value() != static_cast<int>(std::errc::file_exists)) {
      error_message = "create lock dir failed: " + lock_dir.string() + " : " + ec.message();
      return false;
    }
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    if (elapsed_ms >= timeout_ms) {
      error_message = "timed out waiting download lock: " + lock_dir.string();
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}

struct DownloadedAsset {
  const char* local_filename;
  const char* url;
};

const std::array<DownloadedAsset, 2>& nano_assets() {
  static const std::array<DownloadedAsset, 2> kAssets = {
      DownloadedAsset{"backbone.onnx",
                      "https://github.com/HonglinChu/SiamTrackers/raw/master/NanoTrack/models/nanotrackv2/nanotrack_backbone_sim.onnx"},
      DownloadedAsset{"neckhead.onnx",
                      "https://github.com/HonglinChu/SiamTrackers/raw/master/NanoTrack/models/nanotrackv2/nanotrack_head_sim.onnx"},
  };
  return kAssets;
}

const std::array<DownloadedAsset, 1>& vit_assets() {
  static const std::array<DownloadedAsset, 1> kAssets = {
      DownloadedAsset{"vitTracker.onnx",
                      "https://huggingface.co/opencv/object_tracking_vittrack/resolve/main/object_tracking_vittrack_2023sep.onnx?download=true"},
  };
  return kAssets;
}

bool download_asset_via_curl(const std::string& url, const fs::path& dst_path, std::string& error_message) {
  std::error_code ec;
  fs::create_directories(dst_path.parent_path(), ec);
  if (ec) {
    error_message = "create model dir failed: " + dst_path.parent_path().string() + " : " + ec.message();
    return false;
  }

  const fs::path tmp_path = fs::path(dst_path.string() + ".download.part");
  remove_path_if_exists(tmp_path);

  std::string command = "curl --fail --location --silent --show-error --max-time " +
                        std::to_string(kModelDownloadTimeoutSeconds) + " --output " +
                        quote_shell_arg(tmp_path.string()) + " " + quote_shell_arg(url);
  if (!run_command(command, error_message)) {
    remove_path_if_exists(tmp_path);
    return false;
  }
  if (!file_exists_nonempty(tmp_path)) {
    remove_path_if_exists(tmp_path);
    error_message = "download produced empty file: " + dst_path.string() + " url=" + url;
    return false;
  }

  remove_path_if_exists(dst_path);
  fs::rename(tmp_path, dst_path, ec);
  if (ec) {
    error_message = "rename downloaded file failed: " + tmp_path.string() + " -> " + dst_path.string() + " : " +
                    ec.message();
    remove_path_if_exists(tmp_path);
    return false;
  }
  return true;
}

bool ensure_plain_asset(const fs::path& model_dir, const DownloadedAsset& asset, bool auto_download_models,
                        std::string& error_message) {
  const fs::path asset_path = model_dir / asset.local_filename;
  std::error_code ec;
  fs::create_directories(model_dir, ec);
  if (ec) {
    error_message = "create model dir failed: " + model_dir.string() + " : " + ec.message();
    spdlog::error("cvkit_tracking failed to create model dir dir={} error={}", model_dir.string(), error_message);
    return false;
  }
  if (file_exists_nonempty(asset_path)) {
    spdlog::info("cvkit_tracking model asset ready path={}", asset_path.string());
    return true;
  }
  if (!auto_download_models) {
    error_message = "missing tracker model file: " + asset_path.string() + " ; enable autoDownloadModels or place it manually";
    spdlog::warn("cvkit_tracking model asset missing autoDownloadModels=false path={} url={}", asset_path.string(),
                 asset.url);
    return false;
  }

  const fs::path lock_dir = fs::path(asset_path.string() + ".download.lock");
  if (!acquire_lock_dir(lock_dir, kModelDownloadTimeoutSeconds * 1000, error_message)) {
    return false;
  }
  const auto unlock = [&lock_dir]() { remove_path_if_exists(lock_dir); };

  if (file_exists_nonempty(asset_path)) {
    spdlog::info("cvkit_tracking model asset became available while waiting for lock path={}", asset_path.string());
    unlock();
    return true;
  }
  spdlog::info("cvkit_tracking downloading model asset url={} path={}", asset.url, asset_path.string());
  const bool ok = download_asset_via_curl(asset.url, asset_path, error_message);
  if (ok) {
    spdlog::info("cvkit_tracking downloaded model asset path={}", asset_path.string());
  } else {
    spdlog::error("cvkit_tracking failed to download model asset path={} url={} error={}", asset_path.string(),
                  asset.url, error_message);
  }
  unlock();
  return ok;
}

bool ensure_tracker_models_available(TrackerKind kind, const fs::path& model_dir, bool auto_download_models,
                                     std::string& error_message) {
  if (!tracker_kind_uses_model_files(kind)) {
    return true;
  }
  spdlog::info("cvkit_tracking ensuring model assets trackerKind={} modelDir={} autoDownloadModels={}",
               tracker_kind_to_string(kind), model_dir.string(), auto_download_models);
  if (kind == TrackerKind::Nano) {
    for (const DownloadedAsset& asset : nano_assets()) {
      if (!ensure_plain_asset(model_dir, asset, auto_download_models, error_message)) {
        return false;
      }
    }
    return true;
  }
  if (kind == TrackerKind::Vit) {
    for (const DownloadedAsset& asset : vit_assets()) {
      if (!ensure_plain_asset(model_dir, asset, auto_download_models, error_message)) {
        return false;
      }
    }
    return true;
  }
  return true;
}

cv::Ptr<cv::Tracker> create_tracker_for_kind(TrackerKind kind, const fs::path& model_dir) {
  if (kind == TrackerKind::Csrt) {
    return cv::TrackerCSRT::create();
  }
  if (kind == TrackerKind::Kcf) {
    return cv::TrackerKCF::create();
  }
  if (kind == TrackerKind::Mil) {
    return cv::TrackerMIL::create();
  }
  if (kind == TrackerKind::Nano) {
    cv::TrackerNano::Params params;
    params.backbone = (model_dir / "backbone.onnx").string();
    params.neckhead = (model_dir / "neckhead.onnx").string();
    return cv::TrackerNano::create(params);
  }
  cv::TrackerVit::Params params;
  params.net = (model_dir / "vitTracker.onnx").string();
  return cv::TrackerVit::create(params);
}

std::optional<cv::Rect> pick_best_bbox(const std::vector<TrackingInitCandidate>& candidates, const cv::Rect& frame_rect,
                                       TrackingInitSelectMode mode) {
  if (candidates.empty())
    return std::nullopt;
  const double cx = static_cast<double>(frame_rect.x) + static_cast<double>(frame_rect.width) * 0.5;
  const double cy = static_cast<double>(frame_rect.y) + static_cast<double>(frame_rect.height) * 0.5;

  bool found = false;
  bool found_scored = false;
  double best_d2 = std::numeric_limits<double>::infinity();
  double best_score = -std::numeric_limits<double>::infinity();
  int best_area = -1;
  cv::Rect best;
  for (const TrackingInitCandidate& candidate : candidates) {
    const cv::Rect clamped = candidate.bbox & frame_rect;
    if (clamped.width <= 0 || clamped.height <= 0)
      continue;
    if (mode == TrackingInitSelectMode::FirstBox) {
      return clamped;
    }
    const double bx = static_cast<double>(clamped.x) + static_cast<double>(clamped.width) * 0.5;
    const double by = static_cast<double>(clamped.y) + static_cast<double>(clamped.height) * 0.5;
    const double dx = bx - cx;
    const double dy = by - cy;
    const double d2 = dx * dx + dy * dy;
    const int area = clamped.area();

    if (mode == TrackingInitSelectMode::LargestArea) {
      if (!found || area > best_area || (area == best_area && d2 < best_d2)) {
        found = true;
        best_area = area;
        best_d2 = d2;
        best = clamped;
      }
      continue;
    }
    if (mode == TrackingInitSelectMode::HighestScore) {
      if (candidate.score.has_value()) {
        const double score = candidate.score.value();
        if (!found_scored || score > best_score || (score == best_score && d2 < best_d2)) {
          found = true;
          found_scored = true;
          best_score = score;
          best_d2 = d2;
          best = clamped;
        }
        continue;
      }
      if (found_scored) {
        continue;
      }
      if (!found || d2 < best_d2) {
        found = true;
        best_d2 = d2;
        best = clamped;
      }
      continue;
    }

    if (!found || d2 < best_d2) {
      found = true;
      best_d2 = d2;
      best = clamped;
    }
  }
  if (!found)
    return std::nullopt;
  return best;
}

}  // namespace

TrackingService::TrackingService(Config cfg)
    : cfg_(std::move(cfg)), stop_tracking_cooldown_ms_(std::max(0, std::min(60000, cfg_.stop_tracking_cooldown_ms))) {}

TrackingService::~TrackingService() {
  stop();
}

bool TrackingService::start() {
  if (running_.load(std::memory_order_acquire))
    return true;

  stop_tracking_cooldown_until_ms_.store(0, std::memory_order_release);

  f8::cppsdk::ServiceBus::Config bus_cfg;
  bus_cfg.service_id = cfg_.service_id;
  const auto runtime_backend = f8::cppsdk::normalize_runtime_backend_config(cfg_.runtime_backend);
  bus_cfg.apply_runtime_backend(runtime_backend);
  bus_cfg.service_class = cfg_.service_class;
  bus_cfg.service_name = "CVKit Tracking";
  bus_ = std::make_unique<f8::cppsdk::ServiceBus>(bus_cfg);
  bus_->add_lifecycle_node(this);
  bus_->add_stateful_node(this);
  bus_->add_data_node(this);
  bus_->add_command_node(this, TrackingService::describe());

  if (!bus_->start()) {
    bus_.reset();
    return false;
  }

  init_select_mode_ = TrackingInitSelectMode::ClosestCenter;
  init_select_state_ = "closest_center";
  tracker_kind_ = TrackerKind::Csrt;
  tracker_kind_state_ = tracker_kind_to_string(tracker_kind_);
  model_dir_state_ = normalize_model_dir_state(cfg_.model_dir);
  model_dir_path_ = resolve_model_dir_path(model_dir_state_);
  auto_download_models_ = cfg_.auto_download_models;
  const double configured_max_tracking_fps = std::isfinite(cfg_.max_tracking_fps) ? cfg_.max_tracking_fps : 30.0;
  max_tracking_fps_.store(std::max(0.0, std::min(240.0, configured_max_tracking_fps)), std::memory_order_release);
  model_download_retry_after_ms_ = 0;

  if (!cfg_.tracker_kind.empty()) {
    std::string normalized;
    bool ok = false;
    const TrackerKind parsed = parse_tracker_kind(cfg_.tracker_kind, normalized, ok);
    if (ok) {
      tracker_kind_ = parsed;
      tracker_kind_state_ = normalized;
    } else {
      spdlog::warn("invalid trackerKind in config: {} ; defaulting to csrt", cfg_.tracker_kind);
    }
  }

  publish_state_if_changed("serviceClass", cfg_.service_class, "init", json::object());
  publish_state_if_changed("initSelect", init_select_state_, "init", json::object());
  publish_state_if_changed("trackerKind", tracker_kind_state_, "init", json::object());
  publish_state_if_changed("modelDir", model_dir_state_, "init", json::object());
  publish_state_if_changed("autoDownloadModels", auto_download_models_, "init", json::object());
  publish_state_if_changed("maxTrackingFps", max_tracking_fps_.load(std::memory_order_acquire), "init",
                           json::object());
  publish_state_if_changed("stopTrackingCooldownMs", stop_tracking_cooldown_ms_.load(std::memory_order_acquire), "init",
                           json::object());
  publish_state_if_changed("isTracking", false, "init", json::object());
  publish_state_if_changed("isNotTracking", true, "init", json::object());
  publish_error_if_changed("", "init", json::object());

  zenoh_video_.close();
  zenoh_video_open_key_.clear();
  frame_bgra_.clear();
  frame_bgr_.release();
  last_frame_id_ = 0;
  tracking_frame_width_ = 0;
  tracking_frame_height_ = 0;
  last_processed_frame_ts_ms_ = 0;
  next_tracking_due_ts_ms_ = 0.0;
  last_video_open_attempt_ms_ = 0;
  init_video_wait_started_ms_ = 0;
  init_video_wait_last_log_ms_ = 0;
  init_video_wait_misses_ = 0;
  init_box_messages_seen_ = 0;
  init_box_no_candidate_messages_ = 0;
  init_video_wait_error_generation_ = 0;
  init_video_wait_error_active_ = false;

  tracker_.release();
  bbox_ = cv::Rect();
  is_tracking_ = false;
  active_tracker_kind_state_.clear();
  pending_init_boxes_.clear();
  pending_init_box_generation_ = 0;
  monitor_observed_frames_ = 0;
  monitor_processed_frames_ = 0;
  monitor_window_processed_frames_ = 0;
  monitor_window_start_ms_ = 0;
  monitor_last_process_ms_ = 0.0;
  monitor_total_process_ms_ = 0.0;
  monitor_last_latency_ms_ = 0.0;
  monitor_total_latency_ms_ = 0.0;
  monitor_fps_ = 0.0;

  running_.store(true, std::memory_order_release);
  stop_requested_.store(false, std::memory_order_release);
  spdlog::info("cvkit_tracking started serviceId={} backend={}", cfg_.service_id,
               f8::cppsdk::bus_backend_to_string(runtime_backend.bus_backend));
  return true;
}

void TrackingService::stop() {
  stop_requested_.store(true, std::memory_order_release);
  if (!running_.exchange(false, std::memory_order_acq_rel))
    return;
  if (bus_)
    bus_->stop();
  bus_.reset();
}

void TrackingService::tick() {
  if (!running())
    return;
  if (bus_) {
    (void)bus_->drain_main_thread();
    if (bus_->terminate_requested()) {
      stop_requested_.store(true, std::memory_order_release);
      return;
    }
  }

  if (!active_.load(std::memory_order_acquire)) {
    return;
  }

  const std::int64_t now = f8::cppsdk::now_ms();
  const std::int64_t until = stop_tracking_cooldown_until_ms_.load(std::memory_order_acquire);
  if (until > 0 && now >= until) {
    stop_tracking_cooldown_until_ms_.store(0, std::memory_order_release);
  }

  apply_init_box_if_any();
  process_frame_once();
}

void TrackingService::publish_state_if_changed(const std::string& field, const json& value, const std::string& source,
                                               const json& meta) {
  service_runtime::publish_state_if_changed(state_mu_, published_state_, bus_.get(), cfg_.service_id, field, value,
                                            source, meta);
}

void TrackingService::publish_error_if_changed(const json& value, const std::string& source, const json& meta) {
  service_runtime::publish_error_if_changed(state_mu_, published_state_, bus_.get(), cfg_.service_id, value, source,
                                            meta);
}

void TrackingService::emit_monitor_snapshot(std::int64_t ts_ms, std::uint64_t frame_id, double process_ms,
                                            double latency_ms) {
  if (!bus_)
    return;
  (void)frame_id;
  if (monitor_window_start_ms_ <= 0) {
    monitor_window_start_ms_ = ts_ms;
  }
  ++monitor_processed_frames_;
  ++monitor_window_processed_frames_;
  monitor_last_process_ms_ = process_ms;
  monitor_total_process_ms_ += process_ms;
  monitor_last_latency_ms_ = latency_ms;
  monitor_total_latency_ms_ += latency_ms;

  const std::int64_t elapsed = ts_ms - monitor_window_start_ms_;
  if (elapsed >= 1000) {
    monitor_fps_ = static_cast<double>(monitor_window_processed_frames_) * 1000.0 / static_cast<double>(elapsed);
    monitor_window_start_ms_ = ts_ms;
    monitor_window_processed_frames_ = 0;
  }

  const std::uint64_t dropped_frames =
      monitor_observed_frames_ > monitor_processed_frames_ ? (monitor_observed_frames_ - monitor_processed_frames_) : 0;
  const double avg_process_ms = monitor_processed_frames_ > 0
                                    ? (monitor_total_process_ms_ / static_cast<double>(monitor_processed_frames_))
                                    : 0.0;
  const double avg_latency_ms = monitor_processed_frames_ > 0
                                    ? (monitor_total_latency_ms_ / static_cast<double>(monitor_processed_frames_))
                                    : 0.0;
  service_runtime::CvProcessMetrics metrics;
  metrics.observed_frames = monitor_observed_frames_;
  metrics.processed_frames = monitor_processed_frames_;
  metrics.dropped_frames = dropped_frames;
  metrics.last_process_ms = monitor_last_process_ms_;
  metrics.avg_process_ms = avg_process_ms;
  metrics.last_latency_ms = monitor_last_latency_ms_;
  metrics.avg_latency_ms = avg_latency_ms;
  metrics.process_fps = monitor_fps_;
  service_runtime::publish_cv_process_metrics(bus_.get(), metrics);
}

void TrackingService::on_lifecycle(bool active, const json& meta) {
  active_.store(active, std::memory_order_release);
  (void)meta;
}

void TrackingService::on_state(const std::string& node_id, const std::string& field, const json& value,
                               std::int64_t ts_ms, const json& meta) {
  (void)ts_ms;
  if (node_id != cfg_.service_id)
    return;
  if (field == "initSelect" && value.is_string()) {
    set_init_select(value.get<std::string>(), meta);
    return;
  }
  if (field == "trackerKind" && value.is_string()) {
    set_tracker_kind(value.get<std::string>(), meta);
    return;
  }
  if (field == "modelDir" && value.is_string()) {
    set_model_dir(value.get<std::string>(), meta);
    return;
  }
  if (field == "autoDownloadModels") {
    if (!value.is_boolean()) {
      publish_error_if_changed("invalid autoDownloadModels", "state", meta);
      return;
    }
    auto_download_models_ = value.get<bool>();
    model_download_retry_after_ms_ = 0;
    publish_state_if_changed("autoDownloadModels", auto_download_models_, "state", meta);
    publish_error_if_changed("", "state", meta);
    return;
  }
  if (field == "maxTrackingFps") {
    double fps = 0.0;
    if (!json_number_to_double(value, fps) || !std::isfinite(fps) || fps < 0.0) {
      publish_error_if_changed("invalid maxTrackingFps", "state", meta);
      return;
    }
    set_max_tracking_fps(fps, meta);
    return;
  }
  if (field == "stopTrackingCooldownMs") {
    int v = 0;
    if (!json_number_to_int(value, v)) {
      publish_error_if_changed("invalid stopTrackingCooldownMs", "state", meta);
      return;
    }
    v = std::max(0, std::min(60000, v));
    stop_tracking_cooldown_ms_.store(v, std::memory_order_release);
    if (v == 0) {
      stop_tracking_cooldown_until_ms_.store(0, std::memory_order_release);
    }
    publish_state_if_changed("stopTrackingCooldownMs", v, "state", meta);
    return;
  }
}

void TrackingService::on_data(const std::string& node_id, const std::string& port, const json& value,
                              std::int64_t ts_ms, const json& meta) {
  (void)ts_ms;
  (void)meta;
  if (node_id != cfg_.service_id)
    return;
  if (port != "initBox")
    return;
  if (stop_tracking_cooldown_until_ms_.load(std::memory_order_acquire) > f8::cppsdk::now_ms()) {
    return;
  }
  ++init_box_messages_seen_;
  std::vector<TrackingInitCandidate> candidates;
  collect_bbox_candidates(value, candidates, 0);
  if (candidates.empty()) {
    ++init_box_no_candidate_messages_;
    if (init_box_no_candidate_messages_ == 1 || (init_box_no_candidate_messages_ % 300u) == 0u) {
      spdlog::debug("cvkit_tracking initBox had no bbox candidate serviceId={} seen={} noCandidate={}",
                    cfg_.service_id, init_box_messages_seen_, init_box_no_candidate_messages_);
    }
    return;
  }

  {
    std::lock_guard<std::mutex> lock(tracking_mu_);

    // Only accept init boxes when not tracking; tracking should disable matching.
    if (is_tracking_) {
      return;
    }

    pending_init_boxes_ = std::move(candidates);
    ++pending_init_box_generation_;
  }
}

bool TrackingService::on_command(const std::string& call, const json& args, const json& meta, json& result,
                                 std::string& error_code, std::string& error_message) {
  (void)args;
  error_code.clear();
  error_message.clear();
  result = json::object();

  if (call == "stopTracking") {
    json tracking_meta = json::object();
    if (meta.is_object())
      tracking_meta = meta;
    tracking_meta["source"] = "command";
    tracking_meta["call"] = call;

    const int cooldown_ms = stop_tracking_cooldown_ms_.load(std::memory_order_acquire);
    const std::int64_t now = f8::cppsdk::now_ms();
    const std::int64_t until = cooldown_ms > 0 ? (now + static_cast<std::int64_t>(cooldown_ms)) : 0;

    bool was_tracking = false;
    {
      std::lock_guard<std::mutex> lock(tracking_mu_);
      was_tracking = is_tracking_;
      stop_tracking_internal(tracking_meta);
    }

    stop_tracking_cooldown_until_ms_.store(until, std::memory_order_release);

    result["stopped"] = true;
    result["wasTracking"] = was_tracking;
    result["cooldownMs"] = cooldown_ms;
    result["cooldownUntilTsMs"] = until;
    return true;
  }

  error_code = "UNKNOWN_CALL";
  error_message = "unknown call: " + call;
  return false;
}

void TrackingService::stop_tracking_internal(const json& meta) {
  tracker_.release();
  bbox_ = cv::Rect();
  active_tracker_kind_state_.clear();
  tracking_frame_width_ = 0;
  tracking_frame_height_ = 0;
  pending_init_boxes_.clear();
  ++pending_init_box_generation_;
  last_processed_frame_ts_ms_ = 0;
  next_tracking_due_ts_ms_ = 0.0;
  init_video_wait_error_active_ = false;
  set_tracking(false, meta);
}

void TrackingService::set_init_select(const std::string& mode, const json& meta) {
  std::string normalized;
  bool ok = false;
  const TrackingInitSelectMode parsed = parse_init_select_mode(mode, normalized, ok);
  if (!ok) {
    publish_error_if_changed("invalid initSelect: " + mode, "state", meta);
    return;
  }
  init_select_mode_ = parsed;
  init_select_state_ = normalized;
  publish_state_if_changed("initSelect", init_select_state_, "state", meta);
}

void TrackingService::set_tracker_kind(const std::string& kind, const json& meta) {
  std::string normalized;
  bool ok = false;
  const TrackerKind parsed = parse_tracker_kind(kind, normalized, ok);
  if (!ok) {
    publish_error_if_changed("invalid trackerKind: " + kind, "state", meta);
    return;
  }
  {
    std::lock_guard<std::mutex> lock(tracking_mu_);
    tracker_kind_ = parsed;
    tracker_kind_state_ = normalized;
  }
  model_download_retry_after_ms_ = 0;
  publish_state_if_changed("trackerKind", normalized, "state", meta);
  publish_error_if_changed("", "state", meta);
}

void TrackingService::set_model_dir(const std::string& model_dir, const json& meta) {
  model_dir_state_ = normalize_model_dir_state(model_dir);
  model_dir_path_ = resolve_model_dir_path(model_dir_state_);
  model_download_retry_after_ms_ = 0;
  publish_state_if_changed("modelDir", model_dir_state_, "state", meta);
  publish_error_if_changed("", "state", meta);
}

void TrackingService::set_max_tracking_fps(double fps, const json& meta) {
  if (!std::isfinite(fps)) {
    publish_error_if_changed("invalid maxTrackingFps", "state", meta);
    return;
  }
  fps = std::max(0.0, std::min(240.0, fps));
  max_tracking_fps_.store(fps, std::memory_order_release);
  last_processed_frame_ts_ms_ = 0;
  next_tracking_due_ts_ms_ = 0.0;
  publish_state_if_changed("maxTrackingFps", fps, "state", meta);
  publish_error_if_changed("", "state", meta);
}

bool TrackingService::ensure_zenoh_video_open() {
  std::string key;
  if (bus_) {
    const auto resolved = bus_->data_input_zenoh_key(cfg_.service_id, "video");
    if (resolved.has_value()) {
      key = service_runtime::trim_copy(*resolved);
    }
  }
  if (key.empty()) {
    publish_error_if_changed("missing video data input", "runtime", json::object());
    return false;
  }
  if (zenoh_video_.valid() && zenoh_video_open_key_ == key) {
    return true;
  }

  const std::int64_t now = f8::cppsdk::now_ms();
  if (last_video_open_attempt_ms_ > 0 && (now - last_video_open_attempt_ms_) < 1000) {
    return false;
  }
  last_video_open_attempt_ms_ = now;

  zenoh_video_.close();
  zenoh_video_open_key_.clear();
  const auto runtime_backend = f8::cppsdk::normalize_runtime_backend_config(cfg_.runtime_backend);
  if (!zenoh_video_.open(runtime_backend, key)) {
    publish_error_if_changed("zenoh video open failed: " + key, "runtime", json::object());
    return false;
  }
  zenoh_video_open_key_ = key;
  last_processed_frame_ts_ms_ = 0;
  next_tracking_due_ts_ms_ = 0.0;
  tracking_frame_width_ = 0;
  tracking_frame_height_ = 0;
  init_video_wait_error_active_ = false;
  publish_error_if_changed("", "runtime", json::object());
  return true;
}

bool TrackingService::copy_latest_video_frame(std::vector<std::byte>& out_payload,
                                              f8::cppsdk::LatestVideoFrame& out_frame, bool changed_only,
                                              std::uint64_t last_frame_id, std::chrono::milliseconds timeout) {
  if (!ensure_zenoh_video_open()) {
    return false;
  }
  auto frame = zenoh_video_.wait_latest(timeout);
  if (!frame.has_value()) {
    return false;
  }
  if (changed_only && frame->frame_id == last_frame_id) {
    return false;
  }
  out_frame = f8::cppsdk::LatestVideoFrame{};
  out_frame.width = frame->width;
  out_frame.height = frame->height;
  out_frame.pitch = frame->pitch;
  out_frame.format = frame->format;
  out_frame.frame_id = frame->frame_id;
  out_frame.ts_ms = frame->ts_ms;
  out_payload = std::move(frame->payload);
  return true;
}

void TrackingService::apply_init_box_if_any() {
  if (stop_tracking_cooldown_until_ms_.load(std::memory_order_acquire) > f8::cppsdk::now_ms()) {
    std::lock_guard<std::mutex> lock(tracking_mu_);
    if (!pending_init_boxes_.empty()) {
      pending_init_boxes_.clear();
      ++pending_init_box_generation_;
    }
    init_video_wait_started_ms_ = 0;
    init_video_wait_last_log_ms_ = 0;
    init_video_wait_misses_ = 0;
    return;
  }

  std::vector<TrackingInitCandidate> candidates;
  std::uint64_t candidates_generation = 0;
  {
    std::lock_guard<std::mutex> lock(tracking_mu_);
    if (is_tracking_)
      return;
    if (pending_init_boxes_.empty())
      return;
    candidates = pending_init_boxes_;
    candidates_generation = pending_init_box_generation_;
  }

  f8::cppsdk::LatestVideoFrame frame_meta{};
  if (!copy_latest_video_frame(frame_bgra_, frame_meta, false, 0, kInitVideoReadTimeout)) {
    const std::int64_t now = f8::cppsdk::now_ms();
    if (init_video_wait_started_ms_ <= 0) {
      init_video_wait_started_ms_ = now;
      init_video_wait_last_log_ms_ = 0;
      init_video_wait_misses_ = 0;
      ++init_video_wait_error_generation_;
    }
    ++init_video_wait_misses_;
    if ((now - init_video_wait_started_ms_) >= kInitVideoReadLogDelayMs) {
      const bool should_log = init_video_wait_last_log_ms_ <= 0 ||
                              (now - init_video_wait_last_log_ms_) >= kInitVideoReadLogIntervalMs;
      if (should_log) {
        init_video_wait_last_log_ms_ = now;
        spdlog::debug("cvkit_tracking waiting for video frame before tracker init serviceId={} waitMs={} misses={}",
                      cfg_.service_id, now - init_video_wait_started_ms_, init_video_wait_misses_);
      }
    }
    if ((now - init_video_wait_started_ms_) >= kInitVideoReadErrorDelayMs) {
      if (!init_video_wait_error_active_) {
        init_video_wait_error_active_ = true;
        publish_error_if_changed(
            "initBox received but no video frame arrived on tracker video input; check the video->tracker.video edge "
            "and payloadKind=video_frame",
            "runtime",
            json::object({{"waitMs", now - init_video_wait_started_ms_},
                          {"misses", init_video_wait_misses_},
                          {"generation", init_video_wait_error_generation_}}));
      }
    }
    return;
  }
  init_video_wait_started_ms_ = 0;
  init_video_wait_last_log_ms_ = 0;
  init_video_wait_misses_ = 0;
  init_video_wait_error_active_ = false;

  const auto frame_validation = service_runtime::validate_frame_buffer(
      frame_meta.format, frame_meta.width, frame_meta.height, frame_meta.pitch, frame_bgra_.size(),
      f8::cppsdk::kVideoFormatBgra32, 4u);
  if (!frame_validation.ok) {
    publish_error_if_changed(std::string("video frame cannot initialize tracker: ") + frame_validation.reason,
                             "runtime",
                             json::object({{"format", frame_meta.format},
                                           {"width", frame_meta.width},
                                           {"height", frame_meta.height},
                                           {"pitch", frame_meta.pitch},
                                           {"payloadBytes", frame_bgra_.size()}}));
    return;
  }
  const std::size_t row_bytes = frame_validation.row_bytes;
  {
    std::lock_guard<std::mutex> lock(tracking_mu_);
    if (pending_init_box_generation_ == candidates_generation) {
      pending_init_boxes_.clear();
      ++pending_init_box_generation_;
    }
  }

  cv::Mat bgra_mat(static_cast<int>(frame_meta.height), static_cast<int>(frame_meta.width), CV_8UC4,
                   const_cast<std::byte*>(frame_bgra_.data()), row_bytes);
  try {
    cv::cvtColor(bgra_mat, frame_bgr_, cv::COLOR_BGRA2BGR);
  } catch (const cv::Exception& ex) {
    publish_error_if_changed(std::string("opencv cvtColor failed: ") + ex.what(), "runtime",
                             json::object());
    return;
  }

  // Clamp to frame and pick the configured init bbox candidate.
  cv::Rect frame_rect(0, 0, static_cast<int>(frame_meta.width), static_cast<int>(frame_meta.height));
  const std::optional<cv::Rect> selected = pick_best_bbox(candidates, frame_rect, init_select_mode_);
  if (!selected.has_value()) {
    publish_error_if_changed("initBox has no valid bbox candidate", "runtime", json::object());
    return;
  }
  cv::Rect bb = selected.value();

  const std::int64_t now = f8::cppsdk::now_ms();
  if (model_download_retry_after_ms_ > 0 && now < model_download_retry_after_ms_ && tracker_kind_uses_model_files(tracker_kind_)) {
    const std::int64_t remain_ms = model_download_retry_after_ms_ - now;
    spdlog::warn("cvkit_tracking model download cooldown active trackerKind={} retryInMs={}", tracker_kind_state_,
                 remain_ms);
    publish_error_if_changed("tracker model download cooldown active for " + tracker_kind_state_ + " ; retry in " +
                                 std::to_string(std::max<std::int64_t>(1, remain_ms / 1000)) + "s",
                             "runtime", json::object({{"source", "initBox"}}));
    return;
  }

  if (tracker_kind_uses_model_files(tracker_kind_)) {
    std::string download_error;
    if (!ensure_tracker_models_available(tracker_kind_, model_dir_path_, auto_download_models_, download_error)) {
      model_download_retry_after_ms_ = now + kModelDownloadRetryCooldownMs;
      spdlog::error("cvkit_tracking model preparation failed trackerKind={} modelDir={} retryAfterMs={} error={}",
                    tracker_kind_state_, model_dir_path_.string(), model_download_retry_after_ms_, download_error);
      publish_error_if_changed(download_error, "runtime", json::object({{"source", "initBox"}}));
      return;
    }
    model_download_retry_after_ms_ = 0;
  }

  {
    std::lock_guard<std::mutex> lock(tracking_mu_);
    if (is_tracking_)
      return;
    try {
      spdlog::info("cvkit_tracking creating tracker kind={} modelDir={}", tracker_kind_state_, model_dir_path_.string());
      tracker_ = create_tracker_for_kind(tracker_kind_, model_dir_path_);
      if (tracker_.empty()) {
        spdlog::error("cvkit_tracking tracker create returned empty kind={}", tracker_kind_state_);
        publish_error_if_changed("tracker create failed: " + tracker_kind_state_, "runtime",
                                 json::object());
        return;
      }
      tracker_->init(frame_bgr_, bb);
      spdlog::info("cvkit_tracking tracker init ok kind={} bbox=[{},{},{},{}]", tracker_kind_state_, bb.x, bb.y,
                   bb.width, bb.height);
      active_tracker_kind_state_ = tracker_kind_state_;
      bbox_ = bb;
      tracking_frame_width_ = static_cast<int>(frame_meta.width);
      tracking_frame_height_ = static_cast<int>(frame_meta.height);
      last_processed_frame_ts_ms_ = 0;
      next_tracking_due_ts_ms_ = 0.0;
      publish_error_if_changed("", "runtime", json::object({{"source", "initBox"}}));
      set_tracking(true, json::object({{"source", "initBox"}, {"candidates", static_cast<int>(candidates.size())}}));
    } catch (const cv::Exception& ex) {
      spdlog::error("cvkit_tracking tracker init OpenCV exception kind={} error={}", tracker_kind_state_, ex.what());
      publish_error_if_changed(std::string("opencv tracker init failed: ") + ex.what(), "runtime",
                               json::object({{"source", "initBox"}}));
      stop_tracking_internal(json::object({{"reason", "opencv_exception"}, {"source", "initBox"}}));
      return;
    } catch (const std::exception& ex) {
      spdlog::error("cvkit_tracking tracker init exception kind={} error={}", tracker_kind_state_, ex.what());
      publish_error_if_changed(std::string("tracker init failed: ") + ex.what(), "runtime",
                               json::object({{"source", "initBox"}}));
      stop_tracking_internal(json::object({{"reason", "std_exception"}, {"source", "initBox"}}));
      return;
    }
  }
}

void TrackingService::process_frame_once() {
  const std::int64_t process_start_ms = f8::cppsdk::now_ms();
  cv::Ptr<cv::Tracker> tracker;
  cv::Rect bbox;
  std::string active_tracker_kind;
  {
    std::lock_guard<std::mutex> lock(tracking_mu_);
    if (!is_tracking_ || tracker_.empty()) {
      // Not tracking: emit a minimal status so downstream can read isTracking/isNotTracking.
      return;
    }
    tracker = tracker_;
    bbox = bbox_;
    active_tracker_kind = active_tracker_kind_state_;
  }
  f8::cppsdk::LatestVideoFrame frame_meta{};
  if (!copy_latest_video_frame(frame_bgra_, frame_meta, true, last_frame_id_, std::chrono::milliseconds(20))) {
    return;
  }
  if (frame_meta.frame_id == 0 || frame_meta.frame_id == last_frame_id_) {
    return;
  }
  ++monitor_observed_frames_;

  const double max_tracking_fps = max_tracking_fps_.load(std::memory_order_acquire);
  if (max_tracking_fps > 0.0 && next_tracking_due_ts_ms_ > 0.0) {
    const std::int64_t frame_ts_ms = frame_meta.ts_ms > 0 ? frame_meta.ts_ms : f8::cppsdk::now_ms();
    const double min_interval_ms = 1000.0 / max_tracking_fps;
    constexpr double kEarlyToleranceMs = 3.0;
    if (static_cast<double>(frame_ts_ms) + kEarlyToleranceMs < next_tracking_due_ts_ms_) {
      last_frame_id_ = frame_meta.frame_id;
      return;
    }
  }

  last_frame_id_ = frame_meta.frame_id;
  last_processed_frame_ts_ms_ = frame_meta.ts_ms > 0 ? frame_meta.ts_ms : f8::cppsdk::now_ms();
  if (max_tracking_fps > 0.0) {
    const double min_interval_ms = 1000.0 / max_tracking_fps;
    const double processed_ts_ms = static_cast<double>(last_processed_frame_ts_ms_);
    if (next_tracking_due_ts_ms_ <= 0.0) {
      next_tracking_due_ts_ms_ = processed_ts_ms + min_interval_ms;
    } else {
      next_tracking_due_ts_ms_ += min_interval_ms;
      if (next_tracking_due_ts_ms_ + min_interval_ms < processed_ts_ms) {
        next_tracking_due_ts_ms_ = processed_ts_ms + min_interval_ms;
      }
    }
  } else {
    next_tracking_due_ts_ms_ = 0.0;
  }

  const auto frame_validation = service_runtime::validate_frame_buffer(
      frame_meta.format, frame_meta.width, frame_meta.height, frame_meta.pitch, frame_bgra_.size(),
      f8::cppsdk::kVideoFormatBgra32, 4u);
  if (!frame_validation.ok) {
    publish_error_if_changed(std::string("video frame cannot update tracker: ") + frame_validation.reason,
                             "runtime",
                             json::object({{"format", frame_meta.format},
                                           {"width", frame_meta.width},
                                           {"height", frame_meta.height},
                                           {"pitch", frame_meta.pitch},
                                           {"payloadBytes", frame_bgra_.size()}}));
    return;
  }
  const std::size_t row_bytes = frame_validation.row_bytes;
  if (tracking_frame_width_ > 0 && tracking_frame_height_ > 0 &&
      (tracking_frame_width_ != static_cast<int>(frame_meta.width) ||
       tracking_frame_height_ != static_cast<int>(frame_meta.height))) {
    std::lock_guard<std::mutex> lock(tracking_mu_);
    stop_tracking_internal(json::object({{"reason", "video_dimensions_changed"}}));
    return;
  }

  cv::Mat bgra_mat(static_cast<int>(frame_meta.height), static_cast<int>(frame_meta.width), CV_8UC4,
                   const_cast<std::byte*>(frame_bgra_.data()), row_bytes);
  try {
    cv::cvtColor(bgra_mat, frame_bgr_, cv::COLOR_BGRA2BGR);
  } catch (const cv::Exception& ex) {
    publish_error_if_changed(std::string("opencv cvtColor failed: ") + ex.what(), "runtime",
                             json::object());
    std::lock_guard<std::mutex> lock(tracking_mu_);
    stop_tracking_internal(json::object({{"reason", "opencv_exception"}, {"where", "cvtColor"}}));
    return;
  }

  cv::Rect out_bbox = bbox;
  bool ok = false;
  try {
    ok = tracker->update(frame_bgr_, out_bbox);
  } catch (const cv::Exception& ex) {
    publish_error_if_changed(std::string("opencv tracker update failed: ") + ex.what(), "runtime",
                             json::object());
    std::lock_guard<std::mutex> lock(tracking_mu_);
    stop_tracking_internal(json::object({{"reason", "opencv_exception"}, {"where", "update"}}));
    return;
  } catch (const std::exception& ex) {
    publish_error_if_changed(std::string("tracker update failed: ") + ex.what(), "runtime",
                             json::object());
    std::lock_guard<std::mutex> lock(tracking_mu_);
    stop_tracking_internal(json::object({{"reason", "std_exception"}, {"where", "update"}}));
    return;
  }
  if (!ok) {
    std::lock_guard<std::mutex> lock(tracking_mu_);
    stop_tracking_internal(json::object({{"reason", "update_failed"}}));
    return;
  }
  {
    std::lock_guard<std::mutex> lock(tracking_mu_);
    if (!is_tracking_)
      return;
    bbox_ = out_bbox;
  }
  const cv::Rect emit_bbox = out_bbox;

  json out = json::object();
  out["frameId"] = frame_meta.frame_id;
  out["tsMs"] = frame_meta.ts_ms;
  out["width"] = frame_meta.width;
  out["height"] = frame_meta.height;
  out["status"] = "tracking";
  out["tracks"] = json::array(
      {json::object({{"id", 1},
                     {"bbox",
                      json::array({emit_bbox.x, emit_bbox.y, emit_bbox.x + emit_bbox.width, emit_bbox.y + emit_bbox.height})},
                     {"kind", "track"}})});
  out["tracker"] = json::object({{"kind", active_tracker_kind}, {"ok", true}});

  publish_error_if_changed("", "runtime", json::object());
  if (bus_) {
    (void)bus_->emit_data(cfg_.service_id, "tracking", out);
  }
  const std::int64_t end_ts_ms = f8::cppsdk::now_ms();
  emit_monitor_snapshot(end_ts_ms, frame_meta.frame_id, static_cast<double>(end_ts_ms - process_start_ms),
                        service_runtime::latency_ms_from_timestamps(end_ts_ms, frame_meta.ts_ms));
}

void TrackingService::set_tracking(bool tracking, const json& meta) {
  if (tracking == is_tracking_)
    return;
  is_tracking_ = tracking;
  publish_state_if_changed("isTracking", is_tracking_, "runtime", meta);
  publish_state_if_changed("isNotTracking", !is_tracking_, "runtime", meta);
}

json TrackingService::describe() {
  const json init_candidate_schema = schema_object(json{{"bbox", schema_array(schema_integer())},
                                                        {"x", schema_integer()},
                                                        {"y", schema_integer()},
                                                        {"w", schema_integer()},
                                                        {"h", schema_integer()},
                                                        {"x1", schema_integer()},
                                                        {"y1", schema_integer()},
                                                        {"x2", schema_integer()},
                                                        {"y2", schema_integer()},
                                                        {"left", schema_integer()},
                                                        {"top", schema_integer()},
                                                        {"right", schema_integer()},
                                                        {"bottom", schema_integer()},
                                                        {"score", schema_number()},
                                                        {"confidence", schema_number()},
                                                        {"probability", schema_number()},
                                                        {"prob", schema_number()},
                                                        {"conf", schema_number()},
                                                        {"cls", schema_string()}});
  const json init_box_schema = schema_object(json{{"bbox", schema_array(schema_integer())},
                                                  {"x", schema_integer()},
                                                  {"y", schema_integer()},
                                                  {"w", schema_integer()},
                                                  {"h", schema_integer()},
                                                  {"x1", schema_integer()},
                                                  {"y1", schema_integer()},
                                                  {"x2", schema_integer()},
                                                  {"y2", schema_integer()},
                                                  {"left", schema_integer()},
                                                  {"top", schema_integer()},
                                                  {"right", schema_integer()},
                                                  {"bottom", schema_integer()},
                                                  {"score", schema_number()},
                                                  {"confidence", schema_number()},
                                                  {"probability", schema_number()},
                                                  {"prob", schema_number()},
                                                  {"conf", schema_number()},
                                                  {"detections", schema_array(init_candidate_schema)},
                                                  {"items", schema_array(init_candidate_schema)}});

  const json tracking_schema =
      schema_object(json{{"frameId", schema_integer()},
                         {"tsMs", schema_integer()},
                         {"width", schema_integer()},
                         {"height", schema_integer()},
                         {"status", schema_string()},
                         {"tracks",
                          schema_array(schema_object(json{{"id", schema_integer()},
                                                          {"bbox", schema_array(schema_integer())},
                                                          {"kind", schema_string()}}))},
                         {"tracker", schema_object(json{{"kind", schema_string()}, {"ok", schema_boolean()}})}});
  json service;
  service["schemaVersion"] = "f8service/1";
  service["serviceClass"] = "f8.cvkit.tracking";
  service["label"] = "CVKit Tracking";
  service["version"] = "0.0.1";
  service["rendererClass"] = "default_svc";
  service["tags"] = json::array({"cv", "tracking"});
  service["stateFields"] = json::array({
      state_field("initSelect",
                  schema_string_enum({"first_box", "closest_center", "largest_area", "highest_score"}, "closest_center"), "rw",
                  "Init Select", "Init bbox selection strategy: first_box | closest_center | largest_area | highest_score.", true),
      state_field("trackerKind",
                  schema_string_enum({"csrt", "kcf", "mil", "nano", "vit"}, "csrt"),
                  "rw",
                  "Tracker Kind",
                  "OpenCV tracker backend: csrt | kcf | mil | nano | vit.",
                  true),
      state_field("modelDir", json{{"type", "string"}, {"default", default_model_dir_state()}}, "rw", "Model Dir",
                  "Directory containing downloaded tracker model files for nano | vit.", false),
      state_field("autoDownloadModels", json{{"type", "boolean"}, {"default", true}}, "rw", "Auto Download Models",
                  "Auto-download missing tracker model files when a model-based tracker is selected.", false),
      state_field("maxTrackingFps", schema_number(30.0, 0.0, 240.0), "rw", "Max Tracking FPS",
                  "Maximum tracker update rate. Set to 0 to process every incoming video frame.", false),
      state_field("stopTrackingCooldownMs", schema_integer(1000, 0, 60000), "rw", "Stop Cooldown (ms)",
                  "After stopTracking, ignore initBox for this many ms. Set to 0 to disable.", true),
      state_field("isTracking", schema_boolean(), "ro", "Is Tracking", "True when tracker is running.", true),
      state_field("isNotTracking", schema_boolean(), "ro", "Is Not Tracking", "Negation of isTracking.", true),
  });
  service["commands"] = json::array({
      json{{"name", "stopTracking"},
           {"description", "Stop current tracking and return to waiting for initBox."},
           {"definitionProtected", true},
           {"showOnNode", true}},
  });
  service["dataInPorts"] = json::array({
      video_frame_port("video", "Input video frame stream."),
      json{
          {"name", "initBox"},
          {"payload", {{"kind", "json"}, {"valueSchema", init_box_schema}}},
          {"description",
           "Init payload (single bbox or nested detection tree). Recursively extracts bbox candidates and uses the one "
           "selected by initSelect."},
          {"definitionProtected", true},
          {"showOnNode", true}},
  });
  service["dataOutPorts"] = json::array({
      json{{"name", "tracking"},
           {"payload", {{"kind", "json"}, {"valueSchema", tracking_schema}}},
           {"description", "Tracking output stream."},
           {"definitionProtected", true},
           {"showOnNode", true}},
  });

  json out;
  out["service"] = std::move(service);
  out["operators"] = json::array();
  return out;
}

}  // namespace f8::cvkit::tracking
