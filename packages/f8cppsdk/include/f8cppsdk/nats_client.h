#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nats/nats.h>

namespace f8::cppsdk {

class NatsSubscription {
 public:
  NatsSubscription() = default;
  ~NatsSubscription();
  NatsSubscription(const NatsSubscription&) = delete;
  NatsSubscription& operator=(const NatsSubscription&) = delete;
  NatsSubscription(NatsSubscription&&) noexcept;
  NatsSubscription& operator=(NatsSubscription&&) noexcept;

  void unsubscribe();
  bool valid() const { return sub_ != nullptr; }

 private:
  friend class NatsClient;
  explicit NatsSubscription(natsSubscription* sub) : sub_(sub) {}
  natsSubscription* sub_ = nullptr;
};

class NatsClient {
 public:
  using MsgHandler = std::function<void(natsMsg* msg)>;

  NatsClient() = default;
  ~NatsClient();
  NatsClient(const NatsClient&) = delete;
  NatsClient& operator=(const NatsClient&) = delete;
  NatsClient(NatsClient&&) = delete;
  NatsClient& operator=(NatsClient&&) = delete;

  bool connect(const std::string& url);
  void close();
  bool is_connected() const;

  natsConnection* raw() const { return nc_; }
  jsCtx* jetstream() const { return js_; }

  bool publish(const std::string& subject, const std::vector<std::uint8_t>& payload);
  bool publish(const std::string& subject, const void* data, std::size_t len);
  std::optional<std::vector<std::uint8_t>> request(const std::string& subject, const std::vector<std::uint8_t>& payload,
                                                   std::int64_t timeout_ms);

  NatsSubscription subscribe(const std::string& subject, MsgHandler handler);

  static std::string last_error();

 private:
  natsConnection* nc_ = nullptr;
  jsCtx* js_ = nullptr;
};

}  // namespace f8::cppsdk
