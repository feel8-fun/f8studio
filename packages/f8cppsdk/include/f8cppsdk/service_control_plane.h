#pragma once

#include <functional>
#include <string>

#include <nlohmann/json_fwd.hpp>

namespace f8::cppsdk {

// Handler interface for built-in service control endpoints.
struct ServiceControlHandler {
  virtual ~ServiceControlHandler() = default;

  // Optional hint for `status` endpoint (defaults to true).
  virtual bool is_active() const { return true; }

  virtual void on_activate(const nlohmann::json& meta) = 0;
  virtual void on_deactivate(const nlohmann::json& meta) = 0;
  virtual void on_set_active(bool active, const nlohmann::json& meta) = 0;

  virtual bool on_set_state(const std::string& node_id, const std::string& field, const nlohmann::json& value,
                            const nlohmann::json& meta, std::string& error_code, std::string& error_message) = 0;

  virtual bool on_set_rungraph(const nlohmann::json& graph_obj, const nlohmann::json& meta, std::string& error_code,
                               std::string& error_message) = 0;

  virtual bool on_command(const std::string& call, const nlohmann::json& args, const nlohmann::json& meta,
                          nlohmann::json& result, std::string& error_code, std::string& error_message) = 0;
};

}  // namespace f8::cppsdk
