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
class EmaFilter {
 public:
  explicit EmaFilter(double alpha) : alpha_(alpha) {}
  double update(double value, double alpha) {
    alpha_ = alpha;
    if (!value_.has_value()) value_ = value;
    else value_ = (1.0 - alpha_) * value_.value() + alpha_ * value;
    return value_.value();
  }
  void reset() { value_.reset(); }

 private:
  double alpha_ = 0.4;
  std::optional<double> value_;
};

class DemaFilter {
 public:
  double update(double value, double alpha) {
    const double ema1 = first_.update(value, alpha);
    const double ema2 = second_.update(ema1, alpha);
    return 2.0 * ema1 - ema2;
  }
  void reset() {
    first_.reset();
    second_.reset();
  }

 private:
  EmaFilter first_{0.4};
  EmaFilter second_{0.4};
};

class OneEuroFilter {
 public:
  double min_cutoff = 1.5;
  double beta = 0.0;
  double derivative_cutoff = 1.0;
  double default_frequency = 90.0;

  double update(double value, double timestamp) {
    double dt = 1.0 / std::max(1e-6, default_frequency);
    if (last_time_.has_value()) dt = std::max(1e-6, timestamp - last_time_.value());
    last_time_ = timestamp;
    if (!estimate_.has_value()) {
      estimate_ = value;
      derivative_estimate_ = 0.0;
      return value;
    }
    const double derivative = (value - estimate_.value()) / dt;
    derivative_estimate_ = derivative_filter_.update(derivative, alpha(derivative_cutoff, dt));
    const double cutoff = min_cutoff + beta * std::abs(derivative_estimate_);
    estimate_ = value_filter_.update(value, alpha(cutoff, dt));
    return estimate_.value();
  }

 private:
  static double alpha(double cutoff, double dt) {
    const double tau = 1.0 / (2.0 * kPi * std::max(1e-9, cutoff));
    return 1.0 / (1.0 + tau / dt);
  }

  std::optional<double> estimate_;
  double derivative_estimate_ = 0.0;
  std::optional<double> last_time_;
  EmaFilter value_filter_{1.0};
  EmaFilter derivative_filter_{1.0};
};

class SmoothFilterNode final : public OperatorNode, public ComputableNode {
 public:
  SmoothFilterNode(const std::string& node_id, const F8RuntimeNode& node, const json& initial_state)
      : OperatorNode(node_id, data_port_names(node.dataInPorts, {"value"}), data_port_names(node.dataOutPorts, {"value"}),
                     state_names(node.stateFields,
                                 {"filter_type", "ema_alpha", "dema_alpha", "one_euro_min_cutoff", "one_euro_beta",
                                  "one_euro_derivative_cutoff", "one_euro_default_freq"}),
                     strings_or(node.execInPorts, {}), strings_or(node.execOutPorts, {})) {
    filter_type_ = normalize_filter_type(initial_state.value("filter_type", "EMA"));
    ema_alpha_ = clamp_double(json_number_or(initial_state.value("ema_alpha", 0.4), 0.4), 0.0, 1.0);
    dema_alpha_ = clamp_double(json_number_or(initial_state.value("dema_alpha", 0.4), 0.4), 0.0, 1.0);
    one_euro_min_cutoff_ = std::max(1e-6, json_number_or(initial_state.value("one_euro_min_cutoff", 1.5), 1.5));
    one_euro_beta_ = std::max(0.0, json_number_or(initial_state.value("one_euro_beta", 0.0), 0.0));
    one_euro_derivative_cutoff_ =
        std::max(1e-6, json_number_or(initial_state.value("one_euro_derivative_cutoff", 1.0), 1.0));
    one_euro_default_freq_ = std::max(1e-3, json_number_or(initial_state.value("one_euro_default_freq", 90.0), 90.0));
  }

  void on_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "filter_type") filter_type_ = normalize_filter_type(value);
    if (field == "ema_alpha") ema_alpha_ = clamp_double(json_number_or(value, ema_alpha_), 0.0, 1.0);
    if (field == "dema_alpha") dema_alpha_ = clamp_double(json_number_or(value, dema_alpha_), 0.0, 1.0);
    if (field == "one_euro_min_cutoff") one_euro_min_cutoff_ = std::max(1e-6, json_number_or(value, one_euro_min_cutoff_));
    if (field == "one_euro_beta") one_euro_beta_ = std::max(0.0, json_number_or(value, one_euro_beta_));
    if (field == "one_euro_derivative_cutoff") one_euro_derivative_cutoff_ = std::max(1e-6, json_number_or(value, one_euro_derivative_cutoff_));
    if (field == "one_euro_default_freq") one_euro_default_freq_ = std::max(1e-3, json_number_or(value, one_euro_default_freq_));
    reset_bank();
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
    if (!raw.has_value()) return format_number_sequence(last_output_);
    const auto sample = json_number_sequence(raw.value());
    if (sample.empty()) return format_number_sequence(last_output_);
    if (!dirty_ && last_ctx_id_.has_value() && last_ctx_id_.value() == ctx_id && sample == last_input_) {
      return format_number_sequence(last_output_);
    }
    ensure_dimension(sample.size());
    std::vector<double> output;
    output.reserve(sample.size());
    const double t = now_seconds();
    if (filter_type_ == "NONE") {
      output = sample;
    } else if (filter_type_ == "EMA") {
      for (std::size_t i = 0; i < sample.size(); ++i) output.push_back(emas_[i].update(sample[i], ema_alpha_));
    } else if (filter_type_ == "DEMA") {
      for (std::size_t i = 0; i < sample.size(); ++i) output.push_back(demas_[i].update(sample[i], dema_alpha_));
    } else {
      for (std::size_t i = 0; i < sample.size(); ++i) output.push_back(one_euros_[i].update(sample[i], t));
    }
    last_input_ = sample;
    last_output_ = output;
    last_ctx_id_ = ctx_id;
    dirty_ = false;
    return format_number_sequence(last_output_);
  }

 private:
  static std::string normalize_filter_type(const json& value) {
    std::string text = value.is_string() ? value.get<std::string>() : "NONE";
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    if (text == "EMA" || text == "DEMA" || text == "ONEEURO") return text;
    return "NONE";
  }

  void reset_bank() {
    emas_.clear();
    demas_.clear();
    one_euros_.clear();
    dirty_ = true;
  }

  void ensure_dimension(std::size_t dimension) {
    if (filter_type_ == "EMA" && emas_.size() != dimension) emas_.assign(dimension, EmaFilter(ema_alpha_));
    if (filter_type_ == "DEMA" && demas_.size() != dimension) demas_.assign(dimension, DemaFilter());
    if (filter_type_ == "ONEEURO" && one_euros_.size() != dimension) {
      one_euros_.assign(dimension, OneEuroFilter());
      for (auto& filter : one_euros_) {
        filter.min_cutoff = one_euro_min_cutoff_;
        filter.beta = one_euro_beta_;
        filter.derivative_cutoff = one_euro_derivative_cutoff_;
        filter.default_frequency = one_euro_default_freq_;
      }
    }
  }

  std::string filter_type_ = "EMA";
  double ema_alpha_ = 0.4;
  double dema_alpha_ = 0.4;
  double one_euro_min_cutoff_ = 1.5;
  double one_euro_beta_ = 0.0;
  double one_euro_derivative_cutoff_ = 1.0;
  double one_euro_default_freq_ = 90.0;
  std::vector<EmaFilter> emas_;
  std::vector<DemaFilter> demas_;
  std::vector<OneEuroFilter> one_euros_;
  std::vector<double> last_input_;
  std::vector<double> last_output_;
  std::optional<std::int64_t> last_ctx_id_;
  bool dirty_ = true;
};

json smooth_filter_spec() {
  return json{{"specKind", "operator"},
              {"schemaVersion", "f8operator/1"},
              {"serviceClass", kServiceClass},
              {"paletteCategory", std::string(kServiceClass) + ".signal"},
              {"operatorClass", "f8.smooth_filter"},
              {"version", "0.0.1"},
              {"label", "Smooth Filter"},
              {"description", "Smooths scalar or vector inputs with EMA/DEMA/One Euro filtering."},
              {"tags", json::array({"filter", "smoothing", "one_euro", "signal"})},
              {"dataInPorts", json::array({data_port("value", "Value to filter.", any_schema(), false, true)})},
              {"dataOutPorts", json::array({data_port("value", "Filtered output.", any_schema(), false, true)})},
              {"stateFields",
               json::array({state_field("filter_type", "Filter", "Filter type.",
                                        string_enum_schema("EMA", {"NONE", "EMA", "DEMA", "ONEEURO"}), "rw", true, true),
                            state_field("ema_alpha", "EMA Alpha", "EMA smoothing factor (0..1).", number_schema(0.4, 0.0, 1.0),
                                        "rw", true, true, json{{"kind", "slider"}}),
                            state_field("dema_alpha", "DEMA Alpha", "DEMA smoothing factor (0..1).", number_schema(0.4, 0.0, 1.0),
                                        "rw", true, false, json{{"kind", "slider"}}),
                            state_field("one_euro_min_cutoff", "One Euro Min Cutoff", "Minimum cutoff frequency.",
                                        number_schema(1.5, 0.01, 10.0), "rw", true, false, json{{"kind", "slider"}}),
                            state_field("one_euro_beta", "One Euro Beta", "Speed coefficient for dynamic cutoff.",
                                        number_schema(0.0, 0.0, 5.0), "rw", true, false, json{{"kind", "slider"}}),
                            state_field("one_euro_derivative_cutoff", "One Euro Derivative Cutoff",
                                        "Cutoff frequency for the derivative filter.", number_schema(1.0, 0.01, 10.0),
                                        "rw", true, false, json{{"kind", "slider"}}),
                            state_field("one_euro_default_freq", "One Euro Default Freq", "Default sampling frequency (Hz).",
                                        number_schema(90.0, 1.0, 240.0), "rw", true, false,
                                        json{{"kind", "slider"}})})}};
}

}  // namespace

void register_smooth_filter_operator(RuntimeNodeRegistry& registry) {
  registry.register_operator_spec(smooth_filter_spec(), true);
  registry.register_operator_factory(kServiceClass, "f8.smooth_filter",
                                     [](const std::string& node_id, const F8RuntimeNode& node, const json& initial_state) {
                                       return std::make_unique<SmoothFilterNode>(node_id, node, initial_state);
                                     },
                                     true);
}

}  // namespace f8::cppengine
