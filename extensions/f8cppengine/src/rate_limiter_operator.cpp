#include "operator_common.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cctype>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mexce.h>

#include "f8cppengine/constants.h"
#include "f8cppsdk/runtime_node_registry.h"
#include "f8cppsdk/runtime_node.h"
#include "f8cppsdk/time_utils.h"

namespace f8::cppengine {

using f8::cppsdk::ComputableNode;
using f8::cppsdk::EntrypointContext;
using f8::cppsdk::EntrypointNode;
using f8::cppsdk::OperatorNode;
using f8::cppsdk::RuntimeNodeRegistry;
using f8::cppsdk::generated::F8RuntimeNode;

namespace {
class RateLimiterNode final : public OperatorNode, public ComputableNode {
 public:
  RateLimiterNode(const std::string& node_id, const F8RuntimeNode& node, const json& initial_state)
      : OperatorNode(node_id, data_port_names(node.dataInPorts, {"value"}), data_port_names(node.dataOutPorts, {"value"}),
                     state_names(node.stateFields, {"inMin", "inMax", "maxRateUp", "maxRateDown", "maxAccel"}),
                     strings_or(node.execInPorts, {}), strings_or(node.execOutPorts, {})) {
    in_min_ = json_number_or(initial_state.value("inMin", 0.0), 0.0);
    in_max_ = json_number_or(initial_state.value("inMax", 1.0), 1.0);
    max_rate_up_ = std::max(0.0, json_number_or(initial_state.value("maxRateUp", 2.0), 2.0));
    max_rate_down_ = std::max(0.0, json_number_or(initial_state.value("maxRateDown", 2.0), 2.0));
    max_accel_ = std::max(0.0, json_number_or(initial_state.value("maxAccel", 0.0), 0.0));
  }

  void on_lifecycle(bool active, const json& meta) override {
    (void)active;
    (void)meta;
    last_time_s_.reset();
  }

  void on_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "inMin") in_min_ = json_number_or(value, in_min_);
    if (field == "inMax") in_max_ = json_number_or(value, in_max_);
    if (field == "maxRateUp") max_rate_up_ = std::max(0.0, json_number_or(value, max_rate_up_));
    if (field == "maxRateDown") max_rate_down_ = std::max(0.0, json_number_or(value, max_rate_down_));
    if (field == "maxAccel") max_accel_ = std::max(0.0, json_number_or(value, max_accel_));
    dirty_ = true;
  }

  void on_data(const std::string& port, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)port;
    (void)value;
    (void)meta;
    const json out = compute_output("value", ts_ms);
    if (!out.is_null()) (void)emit("value", out, ts_ms);
  }

  json compute_output(const std::string& port, std::int64_t ctx_id) override {
    if (port != "value") return nullptr;
    const auto raw = pull("value", ctx_id);
    if (!raw.has_value()) return last_out_.has_value() ? json(last_out_.value()) : json(nullptr);
    const auto numeric = json_number(raw.value());
    if (!numeric.has_value()) return last_out_.has_value() ? json(last_out_.value()) : json(nullptr);
    if (!dirty_ && last_ctx_id_.has_value() && last_ctx_id_.value() == ctx_id && last_in_.has_value() &&
        last_in_.value() == numeric.value()) {
      return last_out_.has_value() ? json(last_out_.value()) : json(nullptr);
    }
    const double out = step(numeric.value());
    last_in_ = numeric.value();
    last_out_ = out;
    last_ctx_id_ = ctx_id;
    dirty_ = false;
    return out;
  }

 private:
  double step(double x) {
    double lo = in_min_;
    double hi = in_max_;
    if (lo > hi) std::swap(lo, hi);
    const double x_clip = clamp_double(x, lo, hi);
    const double now_s = now_seconds();
    if (!y_.has_value()) {
      y_ = x_clip;
      v_ = 0.0;
      last_time_s_ = now_s;
      return y_.value();
    }
    if (!last_time_s_.has_value()) {
      last_time_s_ = now_s;
      return y_.value();
    }
    const double dt = std::max(1e-6, now_s - last_time_s_.value());
    last_time_s_ = now_s;
    const double err = x_clip - y_.value();
    double desired_v = err / dt;
    if (desired_v > 0.0) desired_v = std::min(desired_v, max_rate_up_);
    else desired_v = std::max(desired_v, -max_rate_down_);
    if (max_accel_ > 0.0) {
      const double max_dv = max_accel_ * dt;
      v_ += clamp_double(desired_v - v_, -max_dv, max_dv);
    } else {
      v_ = desired_v;
    }
    double y_new = y_.value() + v_ * dt;
    if (err != 0.0 && (x_clip - y_new) * err < 0.0) {
      y_new = x_clip;
      v_ = 0.0;
    }
    y_ = clamp_double(y_new, lo, hi);
    return y_.value();
  }

  double in_min_ = 0.0;
  double in_max_ = 1.0;
  double max_rate_up_ = 2.0;
  double max_rate_down_ = 2.0;
  double max_accel_ = 0.0;
  std::optional<double> y_;
  double v_ = 0.0;
  std::optional<double> last_time_s_;
  std::optional<double> last_out_;
  std::optional<double> last_in_;
  std::optional<std::int64_t> last_ctx_id_;
  bool dirty_ = true;
};

json rate_limiter_spec() {
  return json{{"specKind", "operator"},
              {"schemaVersion", "f8operator/1"},
              {"serviceClass", kServiceClass},
              {"paletteCategory", std::string(kServiceClass) + ".signal"},
              {"operatorClass", "f8.rate_limiter"},
              {"version", "0.0.1"},
              {"label", "Rate Limiter"},
              {"description", "Limits the rate of change and optionally acceleration of an input signal."},
              {"tags", json::array({"signal", "limit", "rate", "slew", "smoothing", "transform"})},
              {"dataInPorts", json::array({data_port("value", "Input value.", number_schema(), false, true)})},
              {"dataOutPorts", json::array({data_port("value", "Rate-limited output.", number_schema(), false, true)})},
              {"stateFields",
               json::array({state_field("inMin", "Input Min", "Input/output clamp minimum.", number_schema(0.0)),
                            state_field("inMax", "Input Max", "Input/output clamp maximum.", number_schema(1.0)),
                            state_field("maxRateUp", "Max Rate Up", "Maximum rising rate (units/sec).",
                                        number_schema(2.0, 0.0), "rw", true, true),
                            state_field("maxRateDown", "Max Rate Down", "Maximum falling rate (units/sec).",
                                        number_schema(2.0, 0.0), "rw", true, true),
                            state_field("maxAccel", "Max Accel", "Maximum acceleration. 0 disables acceleration limiting.",
                                        number_schema(0.0, 0.0))})}};
}

}  // namespace

void register_rate_limiter_operator(RuntimeNodeRegistry& registry) {
  registry.register_operator_spec(rate_limiter_spec(), true);
  registry.register_operator_factory(kServiceClass, "f8.rate_limiter",
                                     [](const std::string& node_id, const F8RuntimeNode& node, const json& initial_state) {
                                       return std::make_unique<RateLimiterNode>(node_id, node, initial_state);
                                     },
                                     true);
}

}  // namespace f8::cppengine
