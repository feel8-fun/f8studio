#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <nlohmann/json_fwd.hpp>

#include "f8cppsdk/nats_client.h"

namespace f8::cppsdk {

class KvStore;
struct ServiceControlHandler;

// Minimal request/reply control-plane server compatible with f8pysdk ServiceBus endpoints.
class ServiceControlPlaneServer {
 public:
  struct Config {
    std::string service_id;
    std::string nats_url = "nats://127.0.0.1:4222";
  };

  ServiceControlPlaneServer(Config cfg, NatsClient* client, KvStore* kv, ServiceControlHandler* handler);
  ~ServiceControlPlaneServer();

  bool start();
  void stop();

 private:
  void handle_request(natsMsg* msg, const std::string& endpoint);
  void respond(natsMsg* req, const std::string& req_id, bool ok, const nlohmann::json& result,
               const std::string& err_code, const std::string& err_message);

  Config cfg_;
  NatsClient* client_ = nullptr;
  KvStore* kv_ = nullptr;
  ServiceControlHandler* handler_ = nullptr;

  NatsSubscription sub_activate_;
  NatsSubscription sub_deactivate_;
  NatsSubscription sub_set_active_;
  NatsSubscription sub_status_;
  NatsSubscription sub_set_state_;
  NatsSubscription sub_set_rungraph_;
  NatsSubscription sub_cmd_;
};

}  // namespace f8::cppsdk

