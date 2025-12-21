// Sketch of a daemon-side NATS handler adapter. Replace with your NATS C++ client APIs.
// Uses generated models (e.g., from openapi-generator) for request/response types.

#include <string>
#include <functional>
#include <optional>

// Placeholder generated types
struct PingRequest {
  std::string msgId;
  std::string traceId;
  std::string clientId;
  int hop;
  std::string ts;
  std::string apiVersion;
  // payload fields omitted
};

struct PingReply {
  std::string status;  // "ok"
  // capabilities/profile fields omitted
};

// Placeholder NATS subscription type. Replace with the actual client library.
struct NatsMsg {
  std::string subject;
  std::string reply;
  std::string data;
};

class NatsConnection {
 public:
  void subscribe(const std::string& subject,
                 std::function<void(const NatsMsg&)> handler) {
    // bind handler; library-specific
  }

  void publish(const std::string& subject, const std::string& payload) {
    // send payload
  }
};

PingReply handlePing(const PingRequest& req) {
  PingReply r;
  r.status = "ok";
  return r;
}

void register_ping_handler(NatsConnection& nc) {
  nc.subscribe("f8.master.ping", [&](const NatsMsg& msg) {
    // decode JSON -> PingRequest (use your JSON lib)
    PingRequest req{};
    // TODO: parse msg.data

    PingReply reply = handlePing(req);
    // encode reply as JSON (use your JSON lib)
    std::string json_reply = "{}";
    nc.publish(msg.reply, json_reply);
  });
}

// Pattern: register one handler per x-nats-subject, decode request, run handler, publish reply.
