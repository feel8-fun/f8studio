#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <nats/nats.h>

namespace f8::cppsdk {

struct KvConfig {
  std::string bucket;
  std::uint8_t history = 1;
  std::int64_t max_bytes = -1;
  std::int64_t ttl_ms = 0;
  bool memory_storage = true;
};

class KvStore {
 public:
  using WatchCallback = std::function<void(const std::string& key, const std::vector<std::uint8_t>& value)>;

  KvStore() = default;
  ~KvStore();
  KvStore(const KvStore&) = delete;
  KvStore& operator=(const KvStore&) = delete;
  KvStore(KvStore&&) noexcept;
  KvStore& operator=(KvStore&&) noexcept;

  bool open_or_create(jsCtx* js, const KvConfig& cfg);
  void close();
  bool valid() const { return kv_ != nullptr; }

  bool put(const std::string& key, const void* data, std::size_t len, std::uint64_t* out_rev = nullptr);
  bool put(const std::string& key, const std::vector<std::uint8_t>& bytes, std::uint64_t* out_rev = nullptr);
  std::optional<std::vector<std::uint8_t>> get(const std::string& key) const;

  // Starts a background watch thread. Call stop_watch() before close/destruction.
  bool watch(const std::string& key_pattern, WatchCallback cb, bool include_history = false);
  void stop_watch();

 private:
  kvStore* kv_ = nullptr;
  kvWatcher* watcher_ = nullptr;
  std::thread watch_thread_;
  std::atomic<bool> stop_{false};
};

}  // namespace f8::cppsdk
