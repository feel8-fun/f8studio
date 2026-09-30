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
class SwitchMixerNode final : public OperatorNode, public ComputableNode {
 public:
  SwitchMixerNode(const std::string& node_id, const F8RuntimeNode& node, const json& initial_state)
      : OperatorNode(node_id, data_port_names(node.dataInPorts, {"ch1", "ch2"}), data_port_names(node.dataOutPorts, {"out", "alpha"}),
                     state_names(node.stateFields, {"currentChannel", "resolvedChannel", "fadeMs"}),
                     strings_or(node.execInPorts, {"exec"}), strings_or(node.execOutPorts, {"exec"})) {
    current_channel_ = initial_state.value("currentChannel", "ch1");
    fade_ms_ = std::max(0.0, json_number_or(initial_state.value("fadeMs", 200), 200));
    fade_tau_s_ = fade_ms_ > 0.0 ? fade_ms_ / 3000.0 : 0.0;
  }

  json validate_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "fadeMs") return static_cast<int>(std::max(0.0, json_number_or(value, 0.0)));
    if (field == "currentChannel") return value.is_string() ? value.get<std::string>() : value.dump();
    return value;
  }

  void on_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    if (field == "currentChannel") current_channel_ = validate_state(field, value, ts_ms, meta).get<std::string>();
    if (field == "fadeMs") {
      fade_ms_ = validate_state(field, value, ts_ms, meta).get<int>();
      fade_tau_s_ = fade_ms_ > 0.0 ? fade_ms_ / 3000.0 : 0.0;
    }
  }

  std::vector<std::string> on_exec(std::int64_t exec_id, const std::string& in_port) override {
    (void)in_port;
    const json out = compute_output("out", exec_id);
    const json alpha = compute_output("alpha", exec_id);
    if (!out.is_null()) (void)emit("out", out);
    if (!alpha.is_null()) (void)emit("alpha", alpha);
    return exec_out_ports();
  }

  void on_data(const std::string& port, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)meta;
    const auto numeric = json_number(value);
    if (numeric.has_value()) last_values_[port] = numeric.value();
    const json out = compute_output("out", ts_ms);
    const json alpha = compute_output("alpha", ts_ms);
    if (!out.is_null()) (void)emit("out", out, ts_ms);
    if (!alpha.is_null()) (void)emit("alpha", alpha, ts_ms);
  }

  json compute_output(const std::string& port, std::int64_t ctx_id) override {
    if (port != "out" && port != "alpha") return nullptr;
    if (last_ctx_id_.has_value() && last_ctx_id_.value() == ctx_id && cache_.contains(port)) return cache_[port];
    step(ctx_id);
    last_ctx_id_ = ctx_id;
    return object_value_or_null(cache_, port);
  }

 private:
  std::string resolve_channel() const {
    if (!current_channel_.empty() &&
        std::find(data_in_ports().begin(), data_in_ports().end(), current_channel_) != data_in_ports().end()) {
      return current_channel_;
    }
    if (!resolved_channel_.empty() &&
        std::find(data_in_ports().begin(), data_in_ports().end(), resolved_channel_) != data_in_ports().end()) {
      return resolved_channel_;
    }
    return data_in_ports().empty() ? "" : data_in_ports().front();
  }

  void step(std::int64_t ctx_id) {
    for (const auto& port : data_in_ports()) {
      const auto raw = pull(port, ctx_id);
      if (!raw.has_value()) continue;
      const auto numeric = json_number(raw.value());
      if (numeric.has_value()) last_values_[port] = numeric.value();
    }
    const double now_s = now_seconds();
    const double dt = last_time_s_.has_value() ? std::max(0.0, now_s - last_time_s_.value()) : 0.0;
    last_time_s_ = now_s;
    const std::string desired = resolve_channel();
    if (desired != resolved_channel_) {
      const bool first = resolved_channel_.empty() && !cache_.contains("out");
      transition_from_ = cache_.value("out", last_values_.count(resolved_channel_) ? last_values_[resolved_channel_] : 0.0);
      resolved_channel_ = desired;
      alpha_ = first ? 1.0 : 0.0;
      if (first && last_values_.count(desired)) transition_from_ = last_values_[desired];
      (void)set_state("resolvedChannel", resolved_channel_);
    }
    const double target = last_values_.count(resolved_channel_) ? last_values_[resolved_channel_] : transition_from_;
    if (fade_tau_s_ <= 0.0) alpha_ = 1.0;
    else alpha_ = clamp_double(alpha_ + (1.0 - std::exp(-dt / fade_tau_s_)) * (1.0 - alpha_), 0.0, 1.0);
    const double out = (1.0 - alpha_) * transition_from_ + alpha_ * target;
    if (alpha_ >= 0.999999) transition_from_ = target;
    cache_ = json{{"out", out}, {"alpha", alpha_}};
  }

  std::string current_channel_ = "ch1";
  std::string resolved_channel_;
  double fade_ms_ = 200.0;
  double fade_tau_s_ = 200.0 / 3000.0;
  std::unordered_map<std::string, double> last_values_;
  double transition_from_ = 0.0;
  double alpha_ = 1.0;
  std::optional<double> last_time_s_;
  std::optional<std::int64_t> last_ctx_id_;
  json cache_ = json::object();
};

json switch_mixer_spec() {
  return json{{"specKind", "operator"},
              {"schemaVersion", "f8operator/1"},
              {"serviceClass", kServiceClass},
              {"paletteCategory", std::string(kServiceClass) + ".motion"},
              {"operatorClass", "f8.switch_mixer"},
              {"version", "0.0.3"},
              {"label", "Switch Mixer"},
              {"description", "Switch between user-defined input channels with an optional smooth crossfade."},
              {"tags", json::array({"mix", "switch", "channel", "track", "crossfade"})},
              {"execInPorts", json::array({json{{"name", "exec"}}})},
              {"execOutPorts", json::array({json{{"name", "exec"}}})},
              {"dataInPorts",
               json::array({data_port("ch1", "Input channel 1", number_schema(), false, true),
                            data_port("ch2", "Input channel 2", number_schema(), false, true)})},
              {"dataOutPorts",
               json::array({data_port("out", "Mixed output", number_schema(), false, true),
                            data_port("alpha", "Transition progress (0..1)", number_schema(), false, true)})},
              {"editPolicy", json{{"dataInPorts", editable_collection_policy()}}},
              {"stateFields",
               json::array({state_field("currentChannel", "Current Channel",
                                        "Name of the selected input channel/track to play.", string_schema("ch1"), "rw", true, true),
                            state_field("resolvedChannel", "Resolved Channel",
                                        "Readonly currently resolved input channel after validation/fallback.", string_schema(""), "ro", true, true),
                            state_field("fadeMs", "Fade (ms)", "Transition duration in milliseconds.",
                                        integer_schema(200, 0, 60000), "rw", true, true)})}};
}

}  // namespace

void register_switch_mixer_operator(RuntimeNodeRegistry& registry) {
  registry.register_operator_spec(switch_mixer_spec(), true);
  registry.register_operator_factory(kServiceClass, "f8.switch_mixer",
                                     [](const std::string& node_id, const F8RuntimeNode& node, const json& initial_state) {
                                       return std::make_unique<SwitchMixerNode>(node_id, node, initial_state);
                                     },
                                     true);
}

}  // namespace f8::cppengine
