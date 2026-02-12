#include "dense_optflow_service.h"

#include <utility>

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <spdlog/spdlog.h>

#include "f8cppsdk/state_kv.h"
#include "f8cvkit/cvkit_image_io.h"

namespace f8::cvkit::dense_optflow {

using json = nlohmann::json;

namespace {

json schema_string() { return json{{"type", "string"}}; }
json schema_number() { return json{{"type", "number"}}; }
json schema_integer() { return json{{"type", "integer"}}; }
json schema_boolean() { return json{{"type", "boolean"}}; }

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

}  // namespace

DenseOptflowService::DenseOptflowService(Config cfg) : cfg_(std::move(cfg)) {}

DenseOptflowService::~DenseOptflowService() { stop(); }

bool DenseOptflowService::start() {
  if (running_.load(std::memory_order_acquire)) return true;

  f8::cppsdk::ServiceBus::Config bus_cfg;
  bus_cfg.service_id = cfg_.service_id;
  bus_cfg.nats_url = cfg_.nats_url;
  bus_cfg.kv_memory_storage = true;
  bus_cfg.service_class = cfg_.service_class;
  bus_cfg.service_name = "CVKit Dense Optical Flow";
  bus_ = std::make_unique<f8::cppsdk::ServiceBus>(bus_cfg);
  bus_->add_lifecycle_node(this);
  bus_->add_stateful_node(this);
  bus_->add_data_node(this);

  if (!bus_->start()) {
    bus_.reset();
    return false;
  }

  publish_state_if_changed("serviceClass", cfg_.service_class, "init", json::object());
  publish_state_if_changed("active", active_.load(std::memory_order_acquire), "init", json::object());
  publish_state_if_changed("lastError", "", "init", json::object());
  publish_state_if_changed("lastResult", json::object(), "init", json::object());

  running_.store(true, std::memory_order_release);
  stop_requested_.store(false, std::memory_order_release);
  spdlog::info("cvkit_dense_optflow started serviceId={} natsUrl={}", cfg_.service_id, cfg_.nats_url);
  return true;
}

void DenseOptflowService::stop() {
  stop_requested_.store(true, std::memory_order_release);
  if (!running_.exchange(false, std::memory_order_acq_rel)) return;
  if (bus_) bus_->stop();
  bus_.reset();
}

void DenseOptflowService::tick() {
  if (!running()) return;
  if (bus_) {
    (void)bus_->drain_main_thread();
    if (bus_->terminate_requested()) {
      stop_requested_.store(true, std::memory_order_release);
      return;
    }
  }
}

void DenseOptflowService::publish_state_if_changed(const std::string& field, const json& value, const std::string& source,
                                                   const json& meta) {
  std::lock_guard<std::mutex> lock(state_mu_);
  auto it = published_state_.find(field);
  if (it != published_state_.end() && it->second == value) return;
  published_state_[field] = value;
  if (bus_) {
    (void)f8::cppsdk::kv_set_node_state(bus_->kv(), cfg_.service_id, cfg_.service_id, field, value, source, meta);
  }
}

void DenseOptflowService::on_lifecycle(bool active, const json& meta) {
  active_.store(active, std::memory_order_release);
  publish_state_if_changed("active", active, "lifecycle", meta);
}

void DenseOptflowService::on_state(const std::string& node_id, const std::string& field, const json& value,
                                   std::int64_t ts_ms, const json& meta) {
  (void)ts_ms;
  (void)meta;
  if (node_id != cfg_.service_id) return;
  if (field == "active" && value.is_boolean()) {
    active_.store(value.get<bool>(), std::memory_order_release);
    publish_state_if_changed("active", active_.load(std::memory_order_acquire), "state", json::object());
  }
}

void DenseOptflowService::on_data(const std::string& node_id, const std::string& port, const json& value,
                                  std::int64_t ts_ms, const json& meta) {
  (void)ts_ms;
  if (node_id != cfg_.service_id) return;
  if (port != "request") return;
  if (!value.is_object()) {
    publish_state_if_changed("lastError", "request must be object", "data", meta);
    return;
  }
  handle_request(value, meta);
}

void DenseOptflowService::handle_request(const json& req, const json& meta) {
  if (!bus_) return;
  const std::string prev_path = req.value("prevPath", "");
  const std::string next_path = req.value("nextPath", "");
  if (prev_path.empty() || next_path.empty()) {
    publish_state_if_changed("lastError", "missing prevPath/nextPath", "data", meta);
    return;
  }
  const auto prev_r = f8::cvkit::load_image_bgr(prev_path);
  if (!prev_r.error.empty()) {
    publish_state_if_changed("lastError", prev_r.error, "data", meta);
    return;
  }
  const auto next_r = f8::cvkit::load_image_bgr(next_path);
  if (!next_r.error.empty()) {
    publish_state_if_changed("lastError", next_r.error, "data", meta);
    return;
  }
  if (prev_r.image.size() != next_r.image.size()) {
    publish_state_if_changed("lastError", "prev/next image size mismatch", "data", meta);
    return;
  }

  cv::Mat prev_gray;
  cv::Mat next_gray;
  cv::cvtColor(prev_r.image, prev_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(next_r.image, next_gray, cv::COLOR_BGR2GRAY);

  cv::Mat flow;
  cv::calcOpticalFlowFarneback(prev_gray, next_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
  std::vector<cv::Mat> ch(2);
  cv::split(flow, ch);
  cv::Mat mag, ang;
  cv::cartToPolar(ch[0], ch[1], mag, ang, true);
  const cv::Scalar mean_mag = cv::mean(mag);

  json out = json::object();
  out["prevPath"] = prev_path;
  out["nextPath"] = next_path;
  out["meanMag"] = mean_mag[0];
  publish_state_if_changed("lastError", "", "data", meta);
  publish_state_if_changed("lastResult", out, "data", meta);
  (void)bus_->emit_data(cfg_.service_id, "result", out);
}

json DenseOptflowService::describe() {
  json service;
  service["schemaVersion"] = "f8service/1";
  service["serviceClass"] = "f8.cvkit.denseoptflow";
  service["label"] = "CVKit Dense Optical Flow";
  service["version"] = "0.0.1";
  service["rendererClass"] = "defaultService";
  service["tags"] = json::array({"cv", "optical_flow"});
  service["stateFields"] = json::array({
      state_field("active", schema_boolean(), "rw", "Active", "Enable/disable service processing.", true),
      state_field("lastError", schema_string(), "ro", "Last Error", "Last error message."),
      state_field("lastResult",
                  schema_object(json{{"prevPath", schema_string()}, {"nextPath", schema_string()}, {"meanMag", schema_number()}},
                                json::array({"prevPath", "nextPath", "meanMag"})),
                  "ro", "Last Result", "Most recent result payload."),
  });
  service["editableStateFields"] = false;
  service["commands"] = json::array();
  service["editableCommands"] = false;
  service["dataInPorts"] = json::array({
      json{{"name", "request"},
           {"valueSchema",
            schema_object(json{{"prevPath", schema_string()}, {"nextPath", schema_string()}}, json::array({"prevPath", "nextPath"}))},
           {"description", "Request payload: {prevPath,nextPath} (file paths)."},
           {"required", true}},
  });
  service["dataOutPorts"] = json::array({
      json{{"name", "result"},
           {"valueSchema",
            schema_object(json{{"prevPath", schema_string()}, {"nextPath", schema_string()}, {"meanMag", schema_number()}},
                          json::array({"prevPath", "nextPath", "meanMag"}))},
           {"description", "Result payload: {prevPath,nextPath,meanMag}."},
           {"required", false}},
  });
  service["editableDataInPorts"] = false;
  service["editableDataOutPorts"] = false;

  json out;
  out["service"] = std::move(service);
  out["operators"] = json::array();
  return out;
}

}  // namespace f8::cvkit::dense_optflow
