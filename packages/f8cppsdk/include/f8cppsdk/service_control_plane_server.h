#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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
    std::string service_name;
    std::string service_class;
  };

  ServiceControlPlaneServer(Config cfg, NatsClient* client, KvStore* kv, ServiceControlHandler* handler);
  ~ServiceControlPlaneServer();

  bool start();
  void stop();

 private:
  void handle_request(microRequest* req, const std::string& endpoint);
  void respond(microRequest* req, const std::string& req_id, bool ok, const nlohmann::json& result,
               const std::string& err_code, const std::string& err_message);

  static microError* on_micro_request(microRequest* req);

  Config cfg_;
  NatsClient* client_ = nullptr;
  KvStore* kv_ = nullptr;
  ServiceControlHandler* handler_ = nullptr;

  microService* micro_ = nullptr;
  std::string micro_name_;
  std::string micro_description_;
  std::vector<std::string> micro_metadata_kv_;
  std::vector<const char*> micro_metadata_cstrs_;
};

}  // namespace f8::cppsdk
