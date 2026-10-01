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
class QuatToEulerNode final : public OperatorNode, public ComputableNode {
 public:
  QuatToEulerNode(const std::string& node_id, const F8RuntimeNode& node, const json& initial_state)
      : OperatorNode(node_id, data_port_names(node.dataInPorts, {"quat"}), data_port_names(node.dataOutPorts, {"euler"}),
                     state_names(node.stateFields, {"order", "degrees"}), strings_or(node.execInPorts, {}),
                     strings_or(node.execOutPorts, {})) {
    order_ = normalize_order(initial_state.value("order", "ZYX"));
    degrees_ = json_bool_or(initial_state.value("degrees", true), true);
  }

  void on_state(const std::string& field, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)ts_ms;
    (void)meta;
    if (field == "order") order_ = normalize_order(value);
    if (field == "degrees") degrees_ = json_bool_or(value, degrees_);
    dirty_ = true;
  }

  void on_data(const std::string& port, const json& value, std::int64_t ts_ms, const json& meta) override {
    (void)port;
    (void)value;
    (void)meta;
    const json out = compute_output("euler", ts_ms);
    if (!out.is_null()) (void)emit("euler", out, ts_ms);
  }

  json compute_output(const std::string& port, std::int64_t ctx_id) override {
    if (port != "euler") return nullptr;
    const auto raw = pull("quat", ctx_id);
    if (!raw.has_value() || !raw->is_array() || raw->size() != 4) return last_output_;
    std::array<double, 4> q{};
    double norm = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
      const auto n = json_number(raw.value()[i]);
      if (!n.has_value()) return last_output_;
      q[i] = n.value();
      norm += q[i] * q[i];
    }
    norm = std::sqrt(norm);
    if (norm <= 1e-9) return last_output_;
    for (double& v : q) v /= norm;
    if (!dirty_ && last_ctx_id_.has_value() && last_ctx_id_.value() == ctx_id && last_quat_ == q) return last_output_;
    const auto e = quat_to_euler(q);
    const double f = degrees_ ? 180.0 / kPi : 1.0;
    last_quat_ = q;
    last_output_ = json::array({e[0] * f, e[1] * f, e[2] * f});
    last_ctx_id_ = ctx_id;
    dirty_ = false;
    return last_output_;
  }

 private:
  static double clamp_unit(double v) { return clamp_double(v, -1.0, 1.0); }
  static std::string normalize_order(const json& value) {
    std::string text = value.is_string() ? value.get<std::string>() : "ZYX";
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    static const std::vector<std::string> orders{"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"};
    for (const auto& order : orders) {
      if (text == order) return text;
    }
    return "ZYX";
  }

  std::array<double, 3> quat_to_euler(const std::array<double, 4>& q) const {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double m11 = 1.0 - 2.0 * (y * y + z * z);
    const double m12 = 2.0 * (x * y - z * w);
    const double m13 = 2.0 * (x * z + y * w);
    const double m21 = 2.0 * (x * y + z * w);
    const double m22 = 1.0 - 2.0 * (x * x + z * z);
    const double m23 = 2.0 * (y * z - x * w);
    const double m31 = 2.0 * (x * z - y * w);
    const double m32 = 2.0 * (y * z + x * w);
    const double m33 = 1.0 - 2.0 * (x * x + y * y);
    double ex = 0.0, ey = 0.0, ez = 0.0;
    if (order_ == "XYZ") {
      ey = std::asin(clamp_unit(m13));
      if (std::abs(m13) < 0.9999999) {
        ex = std::atan2(-m23, m33);
        ez = std::atan2(-m12, m11);
      } else {
        ex = std::atan2(m32, m22);
      }
    } else if (order_ == "XZY") {
      ez = std::asin(-clamp_unit(m12));
      if (std::abs(m12) < 0.9999999) {
        ex = std::atan2(m32, m22);
        ey = std::atan2(m13, m11);
      } else {
        ex = std::atan2(-m23, m33);
      }
    } else if (order_ == "YXZ") {
      ex = std::asin(-clamp_unit(m23));
      if (std::abs(m23) < 0.9999999) {
        ey = std::atan2(m13, m33);
        ez = std::atan2(m21, m22);
      } else {
        ey = std::atan2(-m31, m11);
      }
    } else if (order_ == "YZX") {
      ez = std::asin(clamp_unit(m21));
      if (std::abs(m21) < 0.9999999) {
        ex = std::atan2(-m23, m22);
        ey = std::atan2(-m31, m11);
      } else {
        ey = std::atan2(m13, m33);
      }
    } else if (order_ == "ZXY") {
      ex = std::asin(clamp_unit(m32));
      if (std::abs(m32) < 0.9999999) {
        ey = std::atan2(-m31, m33);
        ez = std::atan2(-m12, m22);
      } else {
        ez = std::atan2(m21, m11);
      }
    } else {
      ey = std::asin(-clamp_unit(m31));
      if (std::abs(m31) < 0.9999999) {
        ex = std::atan2(m32, m33);
        ez = std::atan2(m21, m11);
      } else {
        ez = std::atan2(-m12, m22);
      }
    }
    return {ex, ey, ez};
  }

  std::string order_ = "ZYX";
  bool degrees_ = true;
  std::array<double, 4> last_quat_{};
  json last_output_ = nullptr;
  std::optional<std::int64_t> last_ctx_id_;
  bool dirty_ = true;
};

json quat_to_euler_spec() {
  return json{{"specKind", "operator"},
              {"schemaVersion", "f8operator/1"},
              {"serviceClass", kServiceClass},
              {"paletteCategory", std::string(kServiceClass) + ".motion"},
              {"operatorClass", "f8.quat_to_euler"},
              {"version", "0.0.1"},
              {"label", "Quat To Euler"},
              {"description", "Converts quaternion [w,x,y,z] to Euler angles with configurable order."},
              {"tags", json::array({"math", "rotation", "quaternion", "euler", "transform"})},
              {"dataInPorts", json::array({data_port("quat", "Input quaternion [w,x,y,z].", array_schema(number_schema()), false, true)})},
              {"dataOutPorts", json::array({data_port("euler", "Euler angles [x,y,z] in selected order.", array_schema(number_schema()), false, true)})},
              {"stateFields",
               json::array({state_field("order", "Order", "Euler rotation order.",
                                        string_enum_schema("ZYX", {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"}), "rw", true, true),
                            state_field("degrees", "Degrees", "Output in degrees when true, radians when false.",
                                        boolean_schema(true), "rw", true, true)})}};
}

}  // namespace

void register_quat_to_euler_operator(RuntimeNodeRegistry& registry) {
  registry.register_operator_spec(quat_to_euler_spec(), true);
  registry.register_operator_factory(kServiceClass, "f8.quat_to_euler",
                                     [](const std::string& node_id, const F8RuntimeNode& node, const json& initial_state) {
                                       return std::make_unique<QuatToEulerNode>(node_id, node, initial_state);
                                     },
                                     true);
}

}  // namespace f8::cppengine
