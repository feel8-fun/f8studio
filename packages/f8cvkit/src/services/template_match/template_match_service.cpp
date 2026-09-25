#include "template_match_service.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "f8cppsdk/describe_schema.h"
#include "f8cppsdk/time_utils.h"
#include "f8cvkit/base64.h"
#include "../common/service_runtime_utils.h"

namespace f8::cvkit::template_match {

using json = nlohmann::json;
using f8::cppsdk::describe::schema_array;
using f8::cppsdk::describe::schema_integer;
using f8::cppsdk::describe::schema_number;
using f8::cppsdk::describe::schema_object;
using f8::cppsdk::describe::schema_string;
using f8::cppsdk::describe::schema_string_enum;
using f8::cppsdk::describe::state_field;
using f8::cppsdk::describe::video_frame_port;

namespace {

int clamp_int(int v, int lo, int hi) {
  if (v < lo)
    return lo;
  if (v > hi)
    return hi;
  return v;
}

cv::Rect clamp_rect_to_size(const cv::Rect& rect, const cv::Size& size) {
  const int x1 = std::clamp(rect.x, 0, size.width);
  const int y1 = std::clamp(rect.y, 0, size.height);
  const int x2 = std::clamp(rect.x + rect.width, 0, size.width);
  const int y2 = std::clamp(rect.y + rect.height, 0, size.height);
  return cv::Rect(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1));
}

cv::Rect expand_rect(const cv::Rect& rect, int padding) {
  return cv::Rect(rect.x - padding, rect.y - padding, rect.width + padding * 2, rect.height + padding * 2);
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

  std::string fmt = service_runtime::to_lower_ascii_copy(std::move(format));
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

  out.error = "encoded image exceeds maxBytes=" + std::to_string(max_b64_bytes) +
              " (b64 len=" + std::to_string(last_b64.size()) + " raw=" + std::to_string(last_raw) + ")";
  return out;
}

json video_source_metadata(std::uint32_t width, std::uint32_t height) {
  return json::object({{"width", width}, {"height", height}});
}

}  // namespace

TemplateMatchService::TemplateMatchService(Config cfg) : cfg_(std::move(cfg)) {}

TemplateMatchService::~TemplateMatchService() {
  stop();
}

bool TemplateMatchService::start() {
  if (running_.load(std::memory_order_acquire))
    return true;

  f8::cppsdk::ServiceBus::Config bus_cfg;
  bus_cfg.service_id = cfg_.service_id;
  const auto runtime_backend = f8::cppsdk::normalize_runtime_backend_config(cfg_.runtime_backend);
  bus_cfg.apply_runtime_backend(runtime_backend);
  bus_cfg.service_class = cfg_.service_class;
  bus_cfg.service_name = "CVKit Template Match";
  bus_ = std::make_unique<f8::cppsdk::ServiceBus>(bus_cfg);
  bus_->add_lifecycle_node(this);
  bus_->add_stateful_node(this);
  bus_->add_data_node(this);
  bus_->add_command_node(this, TemplateMatchService::describe());

  if (!bus_->start()) {
    bus_.reset();
    return false;
  }

  publish_state_if_changed("serviceClass", cfg_.service_class, "init", json::object());
  publish_state_if_changed("templateImagePngB64", "", "init", json::object());
  publish_state_if_changed("matchThreshold", match_threshold_, "init", json::object());
  publish_state_if_changed("matchingIntervalMs", matching_interval_ms_, "init", json::object());
  publish_state_if_changed("matchColorMode", match_color_mode_, "init", json::object());
  publish_state_if_changed("searchRoiPaddingPx", search_roi_padding_px_, "init", json::object());
  publish_state_if_changed("pyramidScale", pyramid_scale_, "init", json::object());
  publish_error_if_changed("", "init", json::object());

  template_loaded_ = false;
  template_error_.clear();
  template_bgr_.release();
  template_gray_.release();
  template_png_b64_.clear();
  match_threshold_ = 0.5;
  matching_interval_ms_ = 200;
  last_match_ts_ms_ = 0;
  match_color_mode_ = "gray";
  search_roi_padding_px_ = 0;
  pyramid_scale_ = 1.0;
  has_last_detection_ = false;
  last_detection_bbox_ = cv::Rect();

  zenoh_video_.close();
  zenoh_video_open_key_.clear();
  frame_bgra_.clear();
  frame_gray_.release();
  roi_gray_.release();
  roi_small_.release();
  templ_small_.release();
  match_result_.release();
  last_frame_id_ = 0;
  last_video_open_attempt_ms_ = 0;
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
  spdlog::info("cvkit_template_match started serviceId={} backend={}", cfg_.service_id,
               f8::cppsdk::bus_backend_to_string(runtime_backend.bus_backend));
  return true;
}

void TemplateMatchService::stop() {
  stop_requested_.store(true, std::memory_order_release);
  if (!running_.exchange(false, std::memory_order_acq_rel))
    return;
  if (bus_) {
    bus_->stop();
  }
  bus_.reset();
}

void TemplateMatchService::tick() {
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
  detect_once();
}

void TemplateMatchService::publish_state_if_changed(const std::string& field, const json& value,
                                                    const std::string& source, const json& meta) {
  service_runtime::publish_state_if_changed(state_mu_, published_state_, bus_.get(), cfg_.service_id, field, value,
                                            source, meta);
}

void TemplateMatchService::publish_error_if_changed(const json& value, const std::string& source, const json& meta) {
  service_runtime::publish_error_if_changed(state_mu_, published_state_, bus_.get(), cfg_.service_id, value, source,
                                            meta);
}

void TemplateMatchService::on_lifecycle(bool active, const json& meta) {
  active_.store(active, std::memory_order_release);
  (void)meta;
}

void TemplateMatchService::on_state(const std::string& node_id, const std::string& field, const json& value,
                                    std::int64_t ts_ms, const json& meta) {
  (void)ts_ms;
  if (node_id != cfg_.service_id)
    return;
  if (field == "templateImagePngB64" && value.is_string()) {
    set_template_png_b64(value.get<std::string>(), meta);
    return;
  }
  if (field == "matchThreshold") {
    double v = 0.0;
    if (!service_runtime::parse_json_double(value, v)) {
      publish_error_if_changed("invalid matchThreshold", "state", meta);
      return;
    }
    match_threshold_ = std::clamp(v, 0.0, 1.0);
    publish_state_if_changed("matchThreshold", match_threshold_, "state", meta);
    publish_error_if_changed("", "state", meta);
    return;
  }
  if (field == "matchingIntervalMs") {
    int v = 0;
    if (!service_runtime::parse_json_int(value, v)) {
      publish_error_if_changed("invalid matchingIntervalMs", "state", meta);
      return;
    }
    matching_interval_ms_ = std::clamp<std::int64_t>(static_cast<std::int64_t>(v), 0, 60000);
    publish_state_if_changed("matchingIntervalMs", matching_interval_ms_, "state", meta);
    publish_error_if_changed("", "state", meta);
    return;
  }
  if (field == "matchColorMode" && value.is_string()) {
    const std::string mode = service_runtime::to_lower_ascii_copy(service_runtime::trim_copy(value.get<std::string>()));
    if (mode != "gray" && mode != "bgr") {
      publish_error_if_changed("invalid matchColorMode", "state", meta);
      return;
    }
    match_color_mode_ = mode;
    has_last_detection_ = false;
    publish_state_if_changed("matchColorMode", match_color_mode_, "state", meta);
    publish_error_if_changed("", "state", meta);
    return;
  }
  if (field == "searchRoiPaddingPx") {
    int v = 0;
    if (!service_runtime::parse_json_int(value, v)) {
      publish_error_if_changed("invalid searchRoiPaddingPx", "state", meta);
      return;
    }
    search_roi_padding_px_ = std::clamp(v, 0, 10000);
    publish_state_if_changed("searchRoiPaddingPx", search_roi_padding_px_, "state", meta);
    publish_error_if_changed("", "state", meta);
    return;
  }
  if (field == "pyramidScale") {
    double v = 0.0;
    if (!service_runtime::parse_json_double(value, v)) {
      publish_error_if_changed("invalid pyramidScale", "state", meta);
      return;
    }
    pyramid_scale_ = std::clamp(v, 0.25, 1.0);
    publish_state_if_changed("pyramidScale", pyramid_scale_, "state", meta);
    publish_error_if_changed("", "state", meta);
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
  // Pull-based latest-frame stream input.
}

void TemplateMatchService::set_template_png_b64(const std::string& b64, const json& meta) {
  std::string s = service_runtime::trim_copy(b64);

  if (s == template_png_b64_) {
    publish_state_if_changed("templateImagePngB64", template_png_b64_, "state", meta);
    return;
  }

  template_png_b64_ = s;
  template_loaded_ = false;
  template_error_.clear();
  template_bgr_.release();
  template_gray_.release();
  has_last_detection_ = false;
  publish_state_if_changed("templateImagePngB64", template_png_b64_, "state", meta);

  if (template_png_b64_.empty()) {
    template_error_ = "missing templateImagePngB64";
    publish_error_if_changed(template_error_, "state", meta);
    return;
  }

  const auto dec = f8::cvkit::base64_decode(template_png_b64_);
  if (!dec.error.empty()) {
    template_error_ = "base64 decode failed: " + dec.error;
    publish_error_if_changed(template_error_, "state", meta);
    return;
  }

  cv::Mat buf(1, static_cast<int>(dec.bytes.size()), CV_8UC1, const_cast<std::uint8_t*>(dec.bytes.data()));
  cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
  if (img.empty()) {
    template_error_ = "imdecode failed (templateImagePngB64)";
    publish_error_if_changed(template_error_, "state", meta);
    return;
  }

  template_bgr_ = std::move(img);
  try {
    cv::cvtColor(template_bgr_, template_gray_, cv::COLOR_BGR2GRAY);
  } catch (const cv::Exception& ex) {
    template_error_ = std::string("opencv template cvtColor failed: ") + ex.what();
    template_bgr_.release();
    template_gray_.release();
    publish_error_if_changed(template_error_, "state", meta);
    return;
  }
  template_loaded_ = true;
  publish_error_if_changed("", "state", meta);
}

bool TemplateMatchService::ensure_zenoh_video_open() {
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
  publish_error_if_changed("", "runtime", json::object());
  return true;
}

bool TemplateMatchService::copy_latest_video_frame(std::vector<std::byte>& out_payload,
                                                   f8::cppsdk::LatestVideoFrame& out_frame,
                                                   bool changed_only, std::uint64_t last_frame_id,
                                                   std::chrono::milliseconds timeout) {
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

void TemplateMatchService::detect_once() {
  if (!bus_)
    return;
  if (!template_loaded_) {
    if (!template_error_.empty()) {
      publish_error_if_changed(template_error_, "runtime", json::object());
    }
    return;
  }

  {
    std::lock_guard<std::mutex> lock(video_mu_);
    std::uint32_t wait_timeout_ms = 50;
    if (matching_interval_ms_ > 0 && last_match_ts_ms_ > 0) {
      const std::int64_t now_ms = f8::cppsdk::now_ms();
      const std::int64_t remaining = matching_interval_ms_ - (now_ms - last_match_ts_ms_);
      if (remaining > 0) {
        wait_timeout_ms = static_cast<std::uint32_t>(std::min<std::int64_t>(remaining, 500));
      }
    }
    f8::cppsdk::LatestVideoFrame frame_meta{};
    if (!copy_latest_video_frame(frame_bgra_, frame_meta, true, last_frame_id_,
                                 std::chrono::milliseconds(wait_timeout_ms))) {
      return;
    }
    if (frame_meta.frame_id == 0 || frame_meta.frame_id == last_frame_id_) {
      return;
    }
    ++monitor_observed_frames_;
    last_frame_id_ = frame_meta.frame_id;

    const std::int64_t now_ms = f8::cppsdk::now_ms();
    if (matching_interval_ms_ > 0 && last_match_ts_ms_ > 0 && (now_ms - last_match_ts_ms_) < matching_interval_ms_) {
      return;
    }

    const auto frame_validation = service_runtime::validate_frame_buffer(
        frame_meta.format, frame_meta.width, frame_meta.height, frame_meta.pitch, frame_bgra_.size(),
        f8::cppsdk::kVideoFormatBgra32, 4u);
    if (!frame_validation.ok) {
      has_last_detection_ = false;
      frame_gray_.release();
      roi_gray_.release();
      roi_small_.release();
      match_result_.release();
      return;
    }
    const std::size_t row_bytes = frame_validation.row_bytes;
    if (template_bgr_.empty()) {
      template_loaded_ = false;
      template_error_ = "template empty";
      publish_error_if_changed(template_error_, "runtime", json::object());
      return;
    }

    cv::Mat bgra_mat(static_cast<int>(frame_meta.height), static_cast<int>(frame_meta.width), CV_8UC4,
                     const_cast<std::byte*>(frame_bgra_.data()), row_bytes);

    if (template_bgr_.cols > bgra_mat.cols || template_bgr_.rows > bgra_mat.rows) {
      has_last_detection_ = false;
      json out = json::object();
      out["schemaVersion"] = "f8visionDetections/1";
      out["frameId"] = frame_meta.frame_id;
      out["tsMs"] = frame_meta.ts_ms;
      out["width"] = frame_meta.width;
      out["height"] = frame_meta.height;
      out["model"] = "cvkit.template_match";
      out["task"] = "template_match";
      out["skeletonProtocol"] = "none";
      out["detections"] = json::array();

      publish_error_if_changed("", "runtime", json::object());
      last_match_ts_ms_ = now_ms;
      (void)bus_->emit_data(cfg_.service_id, "detections", out);
      const std::int64_t end_ts_ms = f8::cppsdk::now_ms();
      emit_monitor_snapshot(end_ts_ms, frame_meta.frame_id, static_cast<double>(end_ts_ms - now_ms),
                            service_runtime::latency_ms_from_timestamps(end_ts_ms, frame_meta.ts_ms));
      return;
    }

    cv::Mat source_for_match;
    cv::Mat template_for_match;
    try {
      if (match_color_mode_ == "bgr") {
        cv::cvtColor(bgra_mat, roi_gray_, cv::COLOR_BGRA2BGR);
        source_for_match = roi_gray_;
        template_for_match = template_bgr_;
      } else {
        cv::cvtColor(bgra_mat, frame_gray_, cv::COLOR_BGRA2GRAY);
        source_for_match = frame_gray_;
        template_for_match = template_gray_;
      }
    } catch (const cv::Exception& ex) {
      publish_error_if_changed(std::string("opencv cvtColor failed: ") + ex.what(), "runtime",
                               json::object());
      return;
    }

    cv::Rect search_rect(0, 0, source_for_match.cols, source_for_match.rows);
    if (search_roi_padding_px_ > 0 && has_last_detection_) {
      search_rect = clamp_rect_to_size(expand_rect(last_detection_bbox_, search_roi_padding_px_), source_for_match.size());
      if (search_rect.width < template_for_match.cols || search_rect.height < template_for_match.rows) {
        search_rect = cv::Rect(0, 0, source_for_match.cols, source_for_match.rows);
      }
    }

    cv::Mat match_source = source_for_match(search_rect);
    cv::Mat match_template = template_for_match;
    double inverse_scale = 1.0;
    const double pyramid_scale = std::clamp(pyramid_scale_, 0.25, 1.0);
    if (pyramid_scale < 0.999) {
      const int source_w = std::max(1, static_cast<int>(std::lround(static_cast<double>(match_source.cols) * pyramid_scale)));
      const int source_h = std::max(1, static_cast<int>(std::lround(static_cast<double>(match_source.rows) * pyramid_scale)));
      const int templ_w = std::max(1, static_cast<int>(std::lround(static_cast<double>(match_template.cols) * pyramid_scale)));
      const int templ_h = std::max(1, static_cast<int>(std::lround(static_cast<double>(match_template.rows) * pyramid_scale)));
      if (source_w >= templ_w && source_h >= templ_h) {
        try {
          cv::resize(match_source, roi_small_, cv::Size(source_w, source_h), 0.0, 0.0, cv::INTER_AREA);
          cv::resize(match_template, templ_small_, cv::Size(templ_w, templ_h), 0.0, 0.0, cv::INTER_AREA);
          match_source = roi_small_;
          match_template = templ_small_;
          inverse_scale = 1.0 / pyramid_scale;
        } catch (const cv::Exception& ex) {
          publish_error_if_changed(std::string("opencv pyramid resize failed: ") + ex.what(), "runtime",
                                   json::object());
          return;
        }
      }
    }

    try {
      cv::matchTemplate(match_source, match_template, match_result_, cv::TM_CCOEFF_NORMED);
    } catch (const cv::Exception& ex) {
      publish_error_if_changed(std::string("opencv matchTemplate failed: ") + ex.what(), "runtime",
                               json::object());
      return;
    }
    double min_val = 0.0;
    double max_val = 0.0;
    cv::Point min_loc;
    cv::Point max_loc;
    cv::minMaxLoc(match_result_, &min_val, &max_val, &min_loc, &max_loc);

    const int x1 = search_rect.x + static_cast<int>(std::lround(static_cast<double>(max_loc.x) * inverse_scale));
    const int y1 = search_rect.y + static_cast<int>(std::lround(static_cast<double>(max_loc.y) * inverse_scale));
    const cv::Rect detected_bbox = clamp_rect_to_size(cv::Rect(x1, y1, template_bgr_.cols, template_bgr_.rows), bgra_mat.size());

    json detections = json::array();
    if (max_val >= match_threshold_) {
      has_last_detection_ = true;
      last_detection_bbox_ = detected_bbox;
      json det = json::object();
      det["cls"] = "template_match";
      det["score"] = max_val;
      det["bbox"] = json::array({detected_bbox.x, detected_bbox.y, detected_bbox.x + detected_bbox.width,
                                  detected_bbox.y + detected_bbox.height});
      det["keypoints"] = json::array();
      det["obb"] = json::array();
      det["skeletonProtocol"] = "none";
      detections.push_back(std::move(det));
    } else {
      has_last_detection_ = false;
    }

    json out = json::object();
    out["schemaVersion"] = "f8visionDetections/1";
    out["frameId"] = frame_meta.frame_id;
    out["tsMs"] = frame_meta.ts_ms;
    out["width"] = frame_meta.width;
    out["height"] = frame_meta.height;
    out["model"] = "cvkit.template_match";
    out["task"] = "template_match";
    out["skeletonProtocol"] = "none";
    out["detections"] = std::move(detections);

    publish_error_if_changed("", "runtime", json::object());
    last_match_ts_ms_ = now_ms;
    (void)bus_->emit_data(cfg_.service_id, "detections", out);
    const std::int64_t end_ts_ms = f8::cppsdk::now_ms();
    emit_monitor_snapshot(end_ts_ms, frame_meta.frame_id, static_cast<double>(end_ts_ms - now_ms),
                          service_runtime::latency_ms_from_timestamps(end_ts_ms, frame_meta.ts_ms));
  }
}

void TemplateMatchService::emit_monitor_snapshot(std::int64_t ts_ms, std::uint64_t frame_id, double process_ms,
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

bool TemplateMatchService::on_command(const std::string& call, const json& args, const json& meta, json& result,
                                      std::string& error_code, std::string& error_message) {
  (void)meta;
  error_code.clear();
  error_message.clear();
  result = json::object();
  if (call == "captureTemplateFrame") {
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
    f8::cppsdk::LatestVideoFrame frame_meta{};
    {
      std::lock_guard<std::mutex> lock(video_mu_);
      if (!copy_latest_video_frame(frame, frame_meta, false, 0, std::chrono::milliseconds(100))) {
        error_code = "RUNTIME_ERROR";
        error_message = "no frame available";
        return false;
      }
    }

    if (frame_meta.format != 1 || frame_meta.width == 0 || frame_meta.height == 0 || frame_meta.pitch == 0) {
      error_code = "RUNTIME_ERROR";
      error_message = "unsupported video frame format";
      return false;
    }
    const std::size_t row_bytes = static_cast<std::size_t>(frame_meta.pitch);
    if (row_bytes < static_cast<std::size_t>(frame_meta.width) * 4) {
      error_code = "RUNTIME_ERROR";
      error_message = "invalid video frame pitch";
      return false;
    }
    if (frame.size() < row_bytes * static_cast<std::size_t>(frame_meta.height)) {
      error_code = "RUNTIME_ERROR";
      error_message = "video frame too small";
      return false;
    }

    cv::Mat bgra_mat(static_cast<int>(frame_meta.height), static_cast<int>(frame_meta.width), CV_8UC4,
                     const_cast<std::byte*>(frame.data()), static_cast<std::size_t>(frame_meta.pitch));
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

    result["frameId"] = frame_meta.frame_id;
    result["tsMs"] = frame_meta.ts_ms;
    result["source"] = video_source_metadata(frame_meta.width, frame_meta.height);
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
  const json keypoint_schema =
      schema_object(json{{"x", schema_number()}, {"y", schema_number()}, {"score", schema_number()}});
  const json detection_schema = schema_object(json{{"cls", schema_string()},
                                                   {"score", schema_number()},
                                                   {"bbox", schema_array(schema_integer())},
                                                   {"keypoints", schema_array(keypoint_schema)},
                                                   {"obb", schema_array(schema_array(schema_number()))},
                                                   {"skeletonProtocol", schema_string()}});
  const json detections_schema = schema_object(json{{"schemaVersion", schema_string()},
                                                    {"frameId", schema_integer()},
                                                    {"tsMs", schema_integer()},
                                                    {"width", schema_integer()},
                                                    {"height", schema_integer()},
                                                    {"model", schema_string()},
                                                    {"task", schema_string()},
                                                    {"skeletonProtocol", schema_string()},
                                                    {"detections", schema_array(detection_schema)}});
  json service;
  service["schemaVersion"] = "f8service/1";
  service["serviceClass"] = "f8.cvkit.templatematch";
  service["label"] = "CVKit Template Match";
  service["version"] = "0.0.1";
  service["rendererClass"] = "template_match_capture";
  service["tags"] = json::array({"cv", "template_match"});
  service["stateFields"] = json::array({
      state_field("templateImagePngB64", schema_string(), "rw", "Template PNG (Base64)",
                  "PNG bytes encoded as base64. Local-only payload; cleared when exporting publish JSON.",
                  false, nullptr, true),
      state_field("matchThreshold", schema_number(0.5, 0.0, 1.0), "rw", "Match Threshold",
                  "0..1 score threshold used to emit detections.", true, json{{"kind", "slider"}}),
      state_field("matchingIntervalMs", schema_integer(200, 0, 60000), "rw", "Matching Interval (ms)",
                  "Minimum milliseconds between template matching passes.", false),
      state_field("matchColorMode", schema_string(), "rw", "Match Color Mode", "gray or bgr. gray is faster.", false),
      state_field("searchRoiPaddingPx", schema_integer(0, 0, 10000), "rw", "Search ROI Padding",
                  "If >0, search around the previous detection with this padding.", false),
      state_field("pyramidScale", schema_number(1.0, 0.25, 1.0), "rw", "Pyramid Scale",
                  "Optional downscale factor for faster coarse template matching.", false),
  });
  service["commands"] = json::array({
      json{{"name", "captureTemplateFrame"},
           {"description", "Capture current video frame as an encoded image (base64)."},
           {"required", true},
           {"showOnNode", true},
           {"params", json::array({
                          json{{"name", "format"}, {"valueSchema", schema_string()}, {"required", true}},
                          json{{"name", "quality"}, {"valueSchema", schema_integer()}, {"required", true}},
                          json{{"name", "maxBytes"}, {"valueSchema", schema_integer()}, {"required", true}},
                          json{{"name", "maxWidth"}, {"valueSchema", schema_integer()}, {"required", true}},
                          json{{"name", "maxHeight"}, {"valueSchema", schema_integer()}, {"required", true}},
                      })}},
      json{{"name", "ping"}, {"description", "Health check."}, {"required", true}, {"showOnNode", false}},
  });
  service["dataInPorts"] = json::array({
      video_frame_port("video", "Input video frame stream."),
  });
  service["dataOutPorts"] = json::array({
      json{{"name", "detections"},
           {"valueSchema", detections_schema},
           {"description", "Detection output in schema f8visionDetections/1 (single best match as 0/1 detection)."},
           {"required", true},
           {"showOnNode", true}},
  });

  json out;
  out["service"] = std::move(service);
  out["operators"] = json::array();
  return out;
}

}  // namespace f8::cvkit::template_match
