#include "template_match_service.h"

#include <cmath>
#include <cctype>
#include <algorithm>
#include <exception>
#include <utility>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include "f8cppsdk/state_kv.h"
#include "f8cppsdk/shm/sizing.h"
#include "f8cppsdk/time_utils.h"
#include "f8cvkit/base64.h"

namespace f8::cvkit::template_match {

using json = nlohmann::json;

namespace {

json schema_string() { return json{{"type", "string"}}; }
json schema_number() { return json{{"type", "number"}}; }
json schema_number(double default_value, double minimum, double maximum) {
  json s{{"type", "number"}};
  s["default"] = default_value;
  s["minimum"] = minimum;
  s["maximum"] = maximum;
  return s;
}
json schema_integer() { return json{{"type", "integer"}}; }
json schema_integer(std::int64_t default_value, std::int64_t minimum, std::int64_t maximum) {
  json s{{"type", "integer"}};
  s["default"] = default_value;
  s["minimum"] = minimum;
  s["maximum"] = maximum;
  return s;
}

json schema_object(const json& props, const json& required = json::array()) {
  json obj;
  obj["type"] = "object";
  obj["properties"] = props;
  if (required.is_array()) obj["required"] = required;
  obj["additionalProperties"] = false;
  return obj;
}

json state_field(std::string name, const json& value_schema, std::string access, std::string label = {},
                 std::string description = {}, bool show_on_node = false, std::string ui_control = {}) {
  json sf;
  sf["name"] = std::move(name);
  sf["valueSchema"] = value_schema;
  sf["access"] = std::move(access);
  if (!label.empty()) sf["label"] = std::move(label);
  if (!description.empty()) sf["description"] = std::move(description);
  if (show_on_node) sf["showOnNode"] = true;
  if (!ui_control.empty()) sf["uiControl"] = std::move(ui_control);
  return sf;
}

std::string to_lower_copy(std::string s) {
  for (char& ch : s) {
    if (ch >= 'A' && ch <= 'Z') {
      ch = static_cast<char>(ch - 'A' + 'a');
    }
  }
  return s;
}

int clamp_int(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

struct EncodedImage {
  std::string b64;
  std::string format;
  int width = 0;
  int height = 0;
  int bytes = 0;
  int b64_bytes = 0;
  std::string error;
};

EncodedImage encode_image_b64(const cv::Mat& bgr, std::string format, int quality, int max_b64_bytes, int max_width,
                              int max_height) {
  EncodedImage out;
  if (bgr.empty()) {
    out.error = "empty image";
    return out;
  }
  if (bgr.type() != CV_8UC3) {
    out.error = "unsupported image type";
    return out;
  }

  cv::Mat img = bgr;

  if (max_width > 0 || max_height > 0) {
    const int ww = img.cols;
    const int hh = img.rows;
    double scale = 1.0;
    if (max_width > 0) {
      scale = std::min(scale, static_cast<double>(max_width) / static_cast<double>(std::max(1, ww)));
    }
    if (max_height > 0) {
      scale = std::min(scale, static_cast<double>(max_height) / static_cast<double>(std::max(1, hh)));
    }
    if (scale < 1.0) {
      const int nw = std::max(1, static_cast<int>(std::lround(static_cast<double>(ww) * scale)));
      const int nh = std::max(1, static_cast<int>(std::lround(static_cast<double>(hh) * scale)));
      cv::Mat resized;
      cv::resize(img, resized, cv::Size(nw, nh), 0.0, 0.0, cv::INTER_AREA);
      img = std::move(resized);
    }
  }

  std::string fmt = to_lower_copy(std::move(format));
  if (fmt != "jpg" && fmt != "png") {
    out.error = "invalid format (expected jpg|png)";
    return out;
  }
  std::string ext = (fmt == "jpg") ? ".jpg" : ".png";

  int q = clamp_int(quality, 1, 100);

  std::string last_b64;
  std::size_t last_raw = 0;
  for (int iter = 0; iter < 16; ++iter) {
    std::vector<int> params;
    if (ext == ".jpg") {
      params = {cv::IMWRITE_JPEG_QUALITY, q};
    }

    std::vector<std::uint8_t> buf;
    const bool ok = cv::imencode(ext, img, buf, params);
    if (!ok) {
      out.error = "imencode failed";
      return out;
    }

    last_raw = buf.size();
    const std::string b64 = f8::cvkit::base64_encode(buf);
    last_b64 = b64;
    if (static_cast<int>(b64.size()) <= max_b64_bytes) {
      out.b64 = b64;
      out.format = fmt;
      out.width = img.cols;
      out.height = img.rows;
      out.bytes = static_cast<int>(buf.size());
      out.b64_bytes = static_cast<int>(b64.size());
      return out;
    }

    if (ext == ".jpg" && q > 30) {
      q = std::max(30, static_cast<int>(std::lround(static_cast<double>(q) * 0.85)));
      continue;
    }

    const int ww = img.cols;
    const int hh = img.rows;
    if (ww <= 64 || hh <= 64) {
      break;
    }
    const int nw = std::max(64, static_cast<int>(std::lround(static_cast<double>(ww) * 0.85)));
    const int nh = std::max(64, static_cast<int>(std::lround(static_cast<double>(hh) * 0.85)));
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(nw, nh), 0.0, 0.0, cv::INTER_AREA);
    img = std::move(resized);
  }

  out.error = "encoded image exceeds maxBytes=" + std::to_string(max_b64_bytes) + " (b64 len=" +
              std::to_string(last_b64.size()) + " raw=" + std::to_string(last_raw) + ")";
  return out;
}

}  // namespace

TemplateMatchService::TemplateMatchService(Config cfg) : cfg_(std::move(cfg)) {}

TemplateMatchService::~TemplateMatchService() { stop(); }

bool TemplateMatchService::start() {
  if (running_.load(std::memory_order_acquire)) return true;

  f8::cppsdk::ServiceBus::Config bus_cfg;
  bus_cfg.service_id = cfg_.service_id;
  bus_cfg.nats_url = cfg_.nats_url;
  bus_cfg.kv_memory_storage = true;
  bus_cfg.service_class = cfg_.service_class;
  bus_cfg.service_name = "CVKit Template Match";
  bus_ = std::make_unique<f8::cppsdk::ServiceBus>(bus_cfg);
  bus_->add_lifecycle_node(this);
  bus_->add_stateful_node(this);
  bus_->add_data_node(this);
  bus_->add_command_node(this);

  if (!bus_->start()) {
    bus_.reset();
    return false;
  }

  publish_state_if_changed("serviceClass", cfg_.service_class, "init", json::object());
  publish_state_if_changed("templatePngB64", "", "init", json::object());
  publish_state_if_changed("matchThreshold", match_threshold_, "init", json::object());
  publish_state_if_changed("matchingIntervalMs", matching_interval_ms_, "init", json::object());
  publish_state_if_changed("shmName", "", "init", json::object());
  publish_state_if_changed("lastError", "", "init", json::object());

  template_loaded_ = false;
  template_error_.clear();
  template_bgr_.release();
  template_png_b64_.clear();
  match_threshold_ = 0.5;
  matching_interval_ms_ = 200;
  last_match_ts_ms_ = 0;

  shm_name_override_.clear();
  video_.close();
  frame_bgra_.clear();
  last_header_.reset();
  last_frame_id_ = 0;
  last_notify_seq_ = 0;
  last_video_open_attempt_ms_ = 0;
  telemetry_observed_frames_ = 0;
  telemetry_processed_frames_ = 0;
  telemetry_window_processed_frames_ = 0;
  telemetry_window_start_ms_ = 0;
  telemetry_last_process_ms_ = 0.0;
  telemetry_total_process_ms_ = 0.0;
  telemetry_fps_ = 0.0;

  running_.store(true, std::memory_order_release);
  stop_requested_.store(false, std::memory_order_release);
  spdlog::info("cvkit_template_match started serviceId={} natsUrl={}", cfg_.service_id, cfg_.nats_url);
  return true;
}

void TemplateMatchService::stop() {
  stop_requested_.store(true, std::memory_order_release);
  if (!running_.exchange(false, std::memory_order_acq_rel)) return;
  if (bus_) {
    bus_->stop();
  }
  bus_.reset();
}

void TemplateMatchService::tick() {
  if (!running()) return;
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
  detect_once();
}

void TemplateMatchService::publish_state_if_changed(const std::string& field, const json& value, const std::string& source,
                                                    const json& meta) {
  std::lock_guard<std::mutex> lock(state_mu_);
  auto it = published_state_.find(field);
  if (it != published_state_.end() && it->second == value) return;
  published_state_[field] = value;
  if (bus_) {
    (void)f8::cppsdk::kv_set_node_state(bus_->kv(), cfg_.service_id, cfg_.service_id, field, value, source, meta);
  }
}

void TemplateMatchService::on_lifecycle(bool active, const json& meta) {
  active_.store(active, std::memory_order_release);
  (void)meta;
}

void TemplateMatchService::on_state(const std::string& node_id, const std::string& field, const json& value,
                                    std::int64_t ts_ms, const json& meta) {
  (void)ts_ms;
  if (node_id != cfg_.service_id) return;
  if (field == "templatePngB64" && value.is_string()) {
    set_template_png_b64(value.get<std::string>(), meta);
    return;
  }
  if (field == "shmName" && value.is_string()) {
    {
      std::lock_guard<std::mutex> lock(video_mu_);
      shm_name_override_ = value.get<std::string>();
      video_.close();
      last_video_open_attempt_ms_ = 0;
      last_notify_seq_ = 0;
    }
    publish_state_if_changed("shmName", shm_name_override_, "state", meta);
    return;
  }
  if (field == "matchThreshold") {
    try {
      const double v = value.is_number() ? value.get<double>() : std::stod(value.dump());
      match_threshold_ = std::min(1.0, std::max(0.0, v));
      publish_state_if_changed("matchThreshold", match_threshold_, "state", meta);
    } catch (const std::exception& ex) {
      const std::string msg = std::string("invalid matchThreshold: ") + ex.what();
      spdlog::warn("{}", msg);
      publish_state_if_changed("lastError", msg, "state", meta);
    } catch (...) {
      const std::string msg = "invalid matchThreshold: unknown exception";
      spdlog::warn("{}", msg);
      publish_state_if_changed("lastError", msg, "state", meta);
    }
    return;
  }
  if (field == "matchingIntervalMs") {
    try {
      const std::int64_t v = value.is_number_integer() ? value.get<std::int64_t>() : std::stoll(value.dump());
      matching_interval_ms_ = std::max<std::int64_t>(0, std::min<std::int64_t>(60000, v));
      publish_state_if_changed("matchingIntervalMs", matching_interval_ms_, "state", meta);
    } catch (const std::exception& ex) {
      const std::string msg = std::string("invalid matchingIntervalMs: ") + ex.what();
      spdlog::warn("{}", msg);
      publish_state_if_changed("lastError", msg, "state", meta);
    } catch (...) {
      const std::string msg = "invalid matchingIntervalMs: unknown exception";
      spdlog::warn("{}", msg);
      publish_state_if_changed("lastError", msg, "state", meta);
    }
    return;
  }
}

void TemplateMatchService::on_data(const std::string& node_id, const std::string& port, const json& value,
                                   std::int64_t ts_ms, const json& meta) {
  (void)ts_ms;
  (void)node_id;
  (void)port;
  (void)value;
  (void)meta;
  // Pull-based via video shm; no data port required.
}

void TemplateMatchService::set_template_png_b64(const std::string& b64, const json& meta) {
  std::string s = b64;
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();

  if (s == template_png_b64_) {
    publish_state_if_changed("templatePngB64", template_png_b64_, "state", meta);
    return;
  }

  template_png_b64_ = s;
  template_loaded_ = false;
  template_error_.clear();
  template_bgr_.release();
  publish_state_if_changed("templatePngB64", template_png_b64_, "state", meta);

  if (template_png_b64_.empty()) {
    template_error_ = "missing templatePngB64";
    publish_state_if_changed("lastError", template_error_, "state", meta);
    return;
  }

  const auto dec = f8::cvkit::base64_decode(template_png_b64_);
  if (!dec.error.empty()) {
    template_error_ = "base64 decode failed: " + dec.error;
    publish_state_if_changed("lastError", template_error_, "state", meta);
    return;
  }

  cv::Mat buf(1, static_cast<int>(dec.bytes.size()), CV_8UC1, const_cast<std::uint8_t*>(dec.bytes.data()));
  cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
  if (img.empty()) {
    template_error_ = "imdecode failed (templatePngB64)";
    publish_state_if_changed("lastError", template_error_, "state", meta);
    return;
  }

  template_bgr_ = std::move(img);
  template_loaded_ = true;
  publish_state_if_changed("lastError", "", "state", meta);
}

bool TemplateMatchService::ensure_video_open() {
  f8::cppsdk::VideoSharedMemoryHeader hdr{};
  if (video_.readHeader(hdr)) {
    return true;
  }

  const std::int64_t now = f8::cppsdk::now_ms();
  if (last_video_open_attempt_ms_ > 0 && (now - last_video_open_attempt_ms_) < 1000) {
    return false;
  }
  last_video_open_attempt_ms_ = now;

  const std::string shm_name = shm_name_override_;
  if (shm_name.empty()) {
    publish_state_if_changed("lastError", "missing shmName", "runtime", json::object());
    return false;
  }
  // Use the default SHM size. If producers override capacity, expose a state/config
  // field later; for now keep the contract simple and consistent.
  const std::size_t bytes = f8::cppsdk::shm::kDefaultVideoShmBytes;
  if (!video_.open(shm_name, bytes)) {
    publish_state_if_changed("lastError", "video shm open failed: " + shm_name, "runtime", json::object());
    return false;
  }
  last_notify_seq_ = 0;
  publish_state_if_changed("lastError", "", "runtime", json::object());
  return true;
}

void TemplateMatchService::detect_once() {
  if (!bus_) return;
  if (!template_loaded_) {
    if (!template_error_.empty()) {
      publish_state_if_changed("lastError", template_error_, "runtime", json::object());
    }
    return;
  }

  {
    std::lock_guard<std::mutex> lock(video_mu_);
    if (!ensure_video_open()) {
      return;
    }
    std::uint32_t observed_notify_seq = last_notify_seq_;
    std::uint32_t wait_timeout_ms = 50;
    if (matching_interval_ms_ > 0 && last_match_ts_ms_ > 0) {
      const std::int64_t now_ms = f8::cppsdk::now_ms();
      const std::int64_t remaining = matching_interval_ms_ - (now_ms - last_match_ts_ms_);
      if (remaining > 0) {
        wait_timeout_ms = static_cast<std::uint32_t>(std::min<std::int64_t>(remaining, 500));
      }
    }
    if (!video_.waitNewFrame(last_notify_seq_, wait_timeout_ms, &observed_notify_seq)) {
      return;
    }
    last_notify_seq_ = observed_notify_seq;

    f8::cppsdk::VideoSharedMemoryHeader hdr{};
    if (!video_.copyLatestFrame(frame_bgra_, hdr)) {
      return;
    }
    if (hdr.frame_id == 0 || hdr.frame_id == last_frame_id_) {
      return;
    }
    ++telemetry_observed_frames_;
    last_frame_id_ = hdr.frame_id;
    last_header_ = hdr;

    const std::int64_t now_ms = f8::cppsdk::now_ms();
    if (matching_interval_ms_ > 0 && last_match_ts_ms_ > 0 && (now_ms - last_match_ts_ms_) < matching_interval_ms_) {
      return;
    }

    if (hdr.format != 1 || hdr.width == 0 || hdr.height == 0 || hdr.pitch == 0) {
      publish_state_if_changed("lastError", "unsupported video shm format", "runtime", json::object());
      return;
    }
    const std::size_t row_bytes = static_cast<std::size_t>(hdr.pitch);
    if (row_bytes < static_cast<std::size_t>(hdr.width) * 4) {
      publish_state_if_changed("lastError", "invalid video shm pitch", "runtime", json::object());
      return;
    }
    if (frame_bgra_.size() < row_bytes * static_cast<std::size_t>(hdr.height)) {
      publish_state_if_changed("lastError", "video shm frame too small", "runtime", json::object());
      return;
    }
    if (template_bgr_.empty()) {
      template_loaded_ = false;
      template_error_ = "template empty";
      publish_state_if_changed("lastError", template_error_, "runtime", json::object());
      return;
    }

    cv::Mat bgra_mat(static_cast<int>(hdr.height), static_cast<int>(hdr.width), CV_8UC4,
                     const_cast<std::byte*>(frame_bgra_.data()), static_cast<std::size_t>(hdr.pitch));
    cv::Mat bgr;
    try {
      cv::cvtColor(bgra_mat, bgr, cv::COLOR_BGRA2BGR);
    } catch (const cv::Exception& ex) {
      publish_state_if_changed("lastError", std::string("opencv cvtColor failed: ") + ex.what(), "runtime", json::object());
      return;
    }

    if (template_bgr_.cols > bgr.cols || template_bgr_.rows > bgr.rows) {
      publish_state_if_changed("lastError", "template larger than frame", "runtime", json::object());
      return;
    }

    cv::Mat result;
    try {
      cv::matchTemplate(bgr, template_bgr_, result, cv::TM_CCOEFF_NORMED);
    } catch (const cv::Exception& ex) {
      publish_state_if_changed("lastError", std::string("opencv matchTemplate failed: ") + ex.what(), "runtime",
                               json::object());
      return;
    }
    double min_val = 0.0;
    double max_val = 0.0;
    cv::Point min_loc;
    cv::Point max_loc;
    cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

    json out = json::object();
    out["frameId"] = hdr.frame_id;
    out["tsMs"] = hdr.ts_ms;
    out["score"] = max_val;
    out["x"] = max_loc.x;
    out["y"] = max_loc.y;
    out["w"] = template_bgr_.cols;
    out["h"] = template_bgr_.rows;
    out["frameW"] = hdr.width;
    out["frameH"] = hdr.height;

    publish_state_if_changed("lastError", "", "runtime", json::object());
    last_match_ts_ms_ = now_ms;
    (void)bus_->emit_data(cfg_.service_id, "result", out);
    const std::int64_t end_ts_ms = f8::cppsdk::now_ms();
    emit_telemetry(end_ts_ms, hdr.frame_id, static_cast<double>(end_ts_ms - now_ms));
  }
}

void TemplateMatchService::emit_telemetry(std::int64_t ts_ms, std::uint64_t frame_id, double process_ms) {
  if (!bus_) return;
  if (telemetry_window_start_ms_ <= 0) {
    telemetry_window_start_ms_ = ts_ms;
  }
  ++telemetry_processed_frames_;
  ++telemetry_window_processed_frames_;
  telemetry_last_process_ms_ = process_ms;
  telemetry_total_process_ms_ += process_ms;

  const std::int64_t elapsed = ts_ms - telemetry_window_start_ms_;
  if (elapsed >= 1000) {
    telemetry_fps_ = static_cast<double>(telemetry_window_processed_frames_) * 1000.0 / static_cast<double>(elapsed);
    telemetry_window_start_ms_ = ts_ms;
    telemetry_window_processed_frames_ = 0;
  }

  const std::uint64_t dropped_frames =
      telemetry_observed_frames_ > telemetry_processed_frames_ ? (telemetry_observed_frames_ - telemetry_processed_frames_) : 0;
  const double avg_process_ms =
      telemetry_processed_frames_ > 0 ? (telemetry_total_process_ms_ / static_cast<double>(telemetry_processed_frames_)) : 0.0;

  json telemetry = json::object();
  telemetry["tsMs"] = ts_ms;
  telemetry["frameId"] = frame_id;
  telemetry["fps"] = telemetry_fps_;
  telemetry["processMs"] = telemetry_last_process_ms_;
  telemetry["avgProcessMs"] = avg_process_ms;
  telemetry["observedFrames"] = telemetry_observed_frames_;
  telemetry["processedFrames"] = telemetry_processed_frames_;
  telemetry["droppedFrames"] = dropped_frames;
  (void)bus_->emit_data(cfg_.service_id, "telemetry", telemetry);
}

bool TemplateMatchService::on_command(const std::string& call, const json& args, const json& meta, json& result,
                                      std::string& error_code, std::string& error_message) {
  (void)meta;
  error_code.clear();
  error_message.clear();
  result = json::object();
  if (call == "captureFrame") {
    std::string fmt = "jpg";
    int quality = 85;
    int max_bytes = 900000;
    int max_w = 1280;
    int max_h = 720;

    if (args.is_object()) {
      if (args.contains("format") && args["format"].is_string()) {
        fmt = args["format"].get<std::string>();
      }
      if (args.contains("quality") && args["quality"].is_number_integer()) {
        quality = args["quality"].get<int>();
      }
      if (args.contains("maxBytes") && args["maxBytes"].is_number_integer()) {
        max_bytes = args["maxBytes"].get<int>();
      }
      if (args.contains("maxWidth") && args["maxWidth"].is_number_integer()) {
        max_w = args["maxWidth"].get<int>();
      }
      if (args.contains("maxHeight") && args["maxHeight"].is_number_integer()) {
        max_h = args["maxHeight"].get<int>();
      }
    }

    quality = clamp_int(quality, 1, 100);
    max_bytes = clamp_int(max_bytes, 10000, 5000000);
    max_w = clamp_int(max_w, 0, 10000);
    max_h = clamp_int(max_h, 0, 10000);

    std::vector<std::byte> frame;
    f8::cppsdk::VideoSharedMemoryHeader hdr{};
    std::string shm_name;
    {
      std::lock_guard<std::mutex> lock(video_mu_);
      if (!ensure_video_open()) {
        error_code = "RUNTIME_ERROR";
        error_message = "video shm not available";
        return false;
      }

      if (!video_.copyLatestFrame(frame, hdr)) {
        error_code = "RUNTIME_ERROR";
        error_message = "no frame available";
        return false;
      }
      shm_name = shm_name_override_;
    }

    if (hdr.format != 1 || hdr.width == 0 || hdr.height == 0 || hdr.pitch == 0) {
      error_code = "RUNTIME_ERROR";
      error_message = "unsupported video shm format";
      return false;
    }
    const std::size_t row_bytes = static_cast<std::size_t>(hdr.pitch);
    if (row_bytes < static_cast<std::size_t>(hdr.width) * 4) {
      error_code = "RUNTIME_ERROR";
      error_message = "invalid video shm pitch";
      return false;
    }
    if (frame.size() < row_bytes * static_cast<std::size_t>(hdr.height)) {
      error_code = "RUNTIME_ERROR";
      error_message = "video shm frame too small";
      return false;
    }

    cv::Mat bgra_mat(static_cast<int>(hdr.height), static_cast<int>(hdr.width), CV_8UC4,
                     const_cast<std::byte*>(frame.data()), static_cast<std::size_t>(hdr.pitch));
    cv::Mat bgr;
    try {
      cv::cvtColor(bgra_mat, bgr, cv::COLOR_BGRA2BGR);
    } catch (const cv::Exception& ex) {
      error_code = "RUNTIME_ERROR";
      error_message = std::string("opencv cvtColor failed: ") + ex.what();
      return false;
    }

    const auto enc = encode_image_b64(bgr, fmt, quality, max_bytes, max_w, max_h);
    if (!enc.error.empty()) {
      error_code = "RUNTIME_ERROR";
      error_message = enc.error;
      return false;
    }

    result["frameId"] = hdr.frame_id;
    result["tsMs"] = hdr.ts_ms;
    result["source"] = json::object({{"width", hdr.width}, {"height", hdr.height}, {"shmName", shm_name}});
    result["image"] = json::object({{"b64", enc.b64},
                                    {"format", enc.format},
                                    {"width", enc.width},
                                    {"height", enc.height},
                                    {"bytes", enc.bytes},
                                    {"b64Bytes", enc.b64_bytes}});
    return true;
  }
  if (call == "ping") {
    result["pong"] = true;
    return true;
  }
  error_code = "UNKNOWN_CALL";
  error_message = "unknown call: " + call;
  return false;
}

json TemplateMatchService::describe() {
  const json result_schema = schema_object(
      json{{"frameId", schema_integer()},
           {"tsMs", schema_integer()},
           {"score", schema_number()},
           {"x", schema_integer()},
           {"y", schema_integer()},
           {"w", schema_integer()},
           {"h", schema_integer()},
           {"frameW", schema_integer()},
           {"frameH", schema_integer()}});
  const json telemetry_schema = schema_object(
      json{{"tsMs", schema_integer()},
           {"frameId", schema_integer()},
           {"fps", schema_number()},
           {"processMs", schema_number()},
           {"avgProcessMs", schema_number()},
           {"observedFrames", schema_integer()},
           {"processedFrames", schema_integer()},
           {"droppedFrames", schema_integer()}});

  json service;
  service["schemaVersion"] = "f8service/1";
  service["serviceClass"] = "f8.cvkit.templatematch";
  service["label"] = "CVKit Template Match";
  service["version"] = "0.0.1";
  service["rendererClass"] = "pystudio_template_tracker";
  service["tags"] = json::array({"cv", "template_match"});
  service["stateFields"] = json::array({
      state_field("templatePngB64", schema_string(), "rw", "Template PNG (Base64)", "PNG bytes encoded as base64.", false),
      state_field("matchThreshold", schema_number(0.5, 0.0, 1.0), "rw", "Match Threshold", "0..1 score threshold.", true, "slider"),
      state_field("matchingIntervalMs", schema_integer(200, 0, 60000), "rw", "Matching Interval (ms)",
                  "Minimum milliseconds between template matching passes.", false),
      state_field("shmName", schema_string(), "rw", "Video SHM", "Optional SHM name override (e.g. shm.xxx.video).", true),
      state_field("lastError", schema_string(), "ro", "Last Error", "Last error message.", false),
  });
  service["editableStateFields"] = false;
  service["commands"] = json::array({
      json{{"name", "captureFrame"},
           {"description", "Capture current SHM frame as an encoded image (base64)."},
           {"showOnNode", true},
           {"params",
            json::array({
                json{{"name", "format"}, {"valueSchema", schema_string()}, {"required", false}},
                json{{"name", "quality"}, {"valueSchema", schema_integer()}, {"required", false}},
                json{{"name", "maxBytes"}, {"valueSchema", schema_integer()}, {"required", false}},
                json{{"name", "maxWidth"}, {"valueSchema", schema_integer()}, {"required", false}},
                json{{"name", "maxHeight"}, {"valueSchema", schema_integer()}, {"required", false}},
            })}},
      json{{"name", "ping"}, {"description", "Health check."}, {"showOnNode", false}},
  });
  service["editableCommands"] = false;
  service["dataInPorts"] = json::array();
  service["dataOutPorts"] = json::array({
      json{{"name", "result"}, {"valueSchema", result_schema}, {"description", "Match result stream."}, {"required", false}},
      json{{"name", "telemetry"},
           {"valueSchema", telemetry_schema},
           {"description", "Runtime telemetry: fps/process time/dropped frames."},
           {"required", false}},
  });
  service["editableDataInPorts"] = false;
  service["editableDataOutPorts"] = false;

  json out;
  out["service"] = std::move(service);
  out["operators"] = json::array();
  return out;
}

}  // namespace f8::cvkit::template_match
