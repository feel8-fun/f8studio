#pragma once

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

namespace f8::cppsdk {

// C++ capability protocols (opt-in interfaces).
//
// These mirror the protocol-first approach in f8pysdk and are intended to
// standardize optional node behaviors without forcing a deep class hierarchy.

class ClosableNode {
 public:
  virtual ~ClosableNode() = default;
  virtual void close() = 0;
};

class LifecycleNode {
 public:
  virtual ~LifecycleNode() = default;

  // Called on service activate/deactivate transitions.
  virtual void on_lifecycle(bool active, const nlohmann::json& meta) = 0;
};

class StatefulNode {
 public:
  virtual ~StatefulNode() = default;
  virtual void on_state(
      const std::string& node_id, const std::string& field, const nlohmann::json& value, std::int64_t ts_ms,
      const nlohmann::json& meta) = 0;
};

class DataReceivableNode {
 public:
  virtual ~DataReceivableNode() = default;
  virtual void on_data(const std::string& node_id, const std::string& port, const nlohmann::json& value,
                       std::int64_t ts_ms, const nlohmann::json& meta) = 0;
};

class CommandableNode {
 public:
  virtual ~CommandableNode() = default;
  virtual bool on_command(const std::string& call, const nlohmann::json& args, const nlohmann::json& meta,
                          nlohmann::json& result,
                          std::string& error_code, std::string& error_message) = 0;
};

class SetStateHandlerNode {
 public:
  virtual ~SetStateHandlerNode() = default;
  virtual bool on_set_state(const std::string& node_id, const std::string& field, const nlohmann::json& value,
                            const nlohmann::json& meta,
                            std::string& error_code, std::string& error_message) = 0;
};

class RungraphHandlerNode {
 public:
  virtual ~RungraphHandlerNode() = default;
  virtual bool on_set_rungraph(const nlohmann::json& graph_obj, const nlohmann::json& meta, std::string& error_code,
                               std::string& error_message) = 0;
};

}  // namespace f8::cppsdk
