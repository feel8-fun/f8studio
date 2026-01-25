#include "f8cppsdk/nats_client.h"

#include <nats/nats.h>

#include <cstring>

#include <spdlog/spdlog.h>

namespace f8::cppsdk {

namespace {

struct HandlerClosure {
  NatsClient::MsgHandler cb;
};

void on_msg(natsConnection*, natsSubscription*, natsMsg* msg, void* closure) {
  if (closure == nullptr) {
    natsMsg_Destroy(msg);
    return;
  }
  auto* c = static_cast<HandlerClosure*>(closure);
  try {
    if (c->cb) {
      c->cb(msg);
    }
  } catch (...) {}
  natsMsg_Destroy(msg);
}

}  // namespace

NatsSubscription::~NatsSubscription() {
  unsubscribe();
}

NatsSubscription::NatsSubscription(NatsSubscription&& other) noexcept {
  sub_ = other.sub_;
  other.sub_ = nullptr;
}

NatsSubscription& NatsSubscription::operator=(NatsSubscription&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  unsubscribe();
  sub_ = other.sub_;
  other.sub_ = nullptr;
  return *this;
}

void NatsSubscription::unsubscribe() {
  if (sub_ == nullptr) {
    return;
  }
  // Drain to ensure in-flight callbacks complete before subscription is destroyed.
  (void)natsSubscription_DrainTimeout(sub_, 500);
  (void)natsSubscription_WaitForDrainCompletion(sub_, 800);
  natsSubscription_Destroy(sub_);
  sub_ = nullptr;
}

NatsClient::~NatsClient() {
  close();
}

bool NatsClient::connect(const std::string& url) {
  close();
  const std::string u = url.empty() ? std::string(NATS_DEFAULT_URL) : url;
  natsStatus s = natsConnection_ConnectTo(&nc_, u.c_str());
  if (s != NATS_OK) {
    spdlog::error("NATS connect failed: {}", natsStatus_GetText(s));
    close();
    return false;
  }
  s = natsConnection_JetStream(&js_, nc_, nullptr);
  if (s != NATS_OK) {
    spdlog::error("JetStream init failed: {}", natsStatus_GetText(s));
    close();
    return false;
  }
  return true;
}

void NatsClient::close() {
  if (js_ != nullptr) {
    jsCtx_Destroy(js_);
    js_ = nullptr;
  }
  if (nc_ != nullptr) {
    natsConnection_Drain(nc_);
    natsConnection_Destroy(nc_);
    nc_ = nullptr;
  }
}

bool NatsClient::is_connected() const {
  return nc_ != nullptr && !natsConnection_IsClosed(nc_);
}

bool NatsClient::publish(const std::string& subject, const std::vector<std::uint8_t>& payload) {
  return publish(subject, payload.data(), payload.size());
}

bool NatsClient::publish(const std::string& subject, const void* data, std::size_t len) {
  if (!is_connected()) {
    return false;
  }
  const natsStatus s = natsConnection_Publish(nc_, subject.c_str(), data, static_cast<int>(len));
  return s == NATS_OK;
}

std::optional<std::vector<std::uint8_t>> NatsClient::request(const std::string& subject,
                                                             const std::vector<std::uint8_t>& payload,
                                                             std::int64_t timeout_ms) {
  if (!is_connected()) {
    return std::nullopt;
  }
  natsMsg* reply = nullptr;
  const natsStatus s = natsConnection_Request(&reply, nc_, subject.c_str(), payload.data(),
                                              static_cast<int>(payload.size()), timeout_ms);
  if (s != NATS_OK || reply == nullptr) {
    if (reply != nullptr) {
      natsMsg_Destroy(reply);
    }
    return std::nullopt;
  }
  const void* data = natsMsg_GetData(reply);
  const int len = natsMsg_GetDataLength(reply);
  std::vector<std::uint8_t> out;
  if (data != nullptr && len > 0) {
    out.resize(static_cast<std::size_t>(len));
    std::memcpy(out.data(), data, static_cast<std::size_t>(len));
  }
  natsMsg_Destroy(reply);
  return out;
}

NatsSubscription NatsClient::subscribe(const std::string& subject, MsgHandler handler) {
  if (!is_connected()) {
    return NatsSubscription{};
  }

  auto* closure = new HandlerClosure{std::move(handler)};
  natsSubscription* sub = nullptr;
  const natsStatus s = natsConnection_Subscribe(&sub, nc_, subject.c_str(), &on_msg, closure);
  if (s != NATS_OK || sub == nullptr) {
    delete closure;
    spdlog::error("NATS subscribe failed subject={} err={}", subject, natsStatus_GetText(s));
    return NatsSubscription{};
  }
  natsSubscription_SetOnCompleteCB(sub, [](void* c) { delete static_cast<HandlerClosure*>(c); }, closure);
  return NatsSubscription{sub};
}

std::string NatsClient::last_error() {
  const char* err = nats_GetLastError(nullptr);
  return err ? std::string(err) : std::string();
}

}  // namespace f8::cppsdk
