#pragma once

#include <memory>
#include <string>

#include "f8cppsdk/service_bus.h"

namespace f8::cppsdk {

// Minimal runtime facade (phase 1):
// - wraps a `ServiceBus`
// Future: add ServiceHost + Registry + Executors for high-performance engine.
class ServiceRuntime final {
 public:
  explicit ServiceRuntime(ServiceBus::Config cfg) : bus_(std::make_unique<ServiceBus>(std::move(cfg))) {}
  ~ServiceRuntime() { stop(); }
  ServiceRuntime(const ServiceRuntime&) = delete;
  ServiceRuntime& operator=(const ServiceRuntime&) = delete;
  ServiceRuntime(ServiceRuntime&&) = delete;
  ServiceRuntime& operator=(ServiceRuntime&&) = delete;

  ServiceBus& bus() { return *bus_; }
  const ServiceBus& bus() const { return *bus_; }

  bool start() { return bus_ ? bus_->start() : false; }
  void stop() {
    if (bus_) {
      bus_->stop();
    }
  }

 private:
  std::unique_ptr<ServiceBus> bus_;
};

}  // namespace f8::cppsdk
