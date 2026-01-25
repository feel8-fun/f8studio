#include "f8cppsdk/kv_store.h"

#include <nats/nats.h>

#include <cstring>

#include <spdlog/spdlog.h>

namespace f8::cppsdk {

KvStore::~KvStore() {
  stop_watch();
  close();
}

KvStore::KvStore(KvStore&& other) noexcept {
  kv_ = other.kv_;
  watcher_ = other.watcher_;
  other.kv_ = nullptr;
  other.watcher_ = nullptr;
}

KvStore& KvStore::operator=(KvStore&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  stop_watch();
  close();
  kv_ = other.kv_;
  watcher_ = other.watcher_;
  other.kv_ = nullptr;
  other.watcher_ = nullptr;
  return *this;
}

bool KvStore::open_or_create(jsCtx* js, const KvConfig& cfg) {
  if (js == nullptr) {
    return false;
  }
  close();

  kvStore* kv = nullptr;
  natsStatus s = js_KeyValue(&kv, js, cfg.bucket.c_str());
  if (s == NATS_OK && kv != nullptr) {
    kv_ = kv;
    return true;
  }

  kvConfig c;
  kvConfig_Init(&c);
  c.Bucket = cfg.bucket.c_str();
  c.History = cfg.history;
  c.MaxBytes = cfg.max_bytes;
  c.TTL = (cfg.ttl_ms <= 0) ? 0 : (cfg.ttl_ms * 1000000LL);  // ns
  c.StorageType = cfg.memory_storage ? js_MemoryStorage : js_FileStorage;
  c.Replicas = 1;

  s = js_CreateKeyValue(&kv, js, &c);
  if (s != NATS_OK || kv == nullptr) {
    spdlog::error("KV create failed bucket={} err={}", cfg.bucket, natsStatus_GetText(s));
    return false;
  }
  kv_ = kv;
  return true;
}

void KvStore::close() {
  if (kv_ != nullptr) {
    kvStore_Destroy(kv_);
    kv_ = nullptr;
  }
}

bool KvStore::put(const std::string& key, const void* data, std::size_t len, std::uint64_t* out_rev) {
  if (kv_ == nullptr) {
    return false;
  }
  uint64_t rev = 0;
  const natsStatus s = kvStore_Put(&rev, kv_, key.c_str(), data, static_cast<int>(len));
  if (out_rev != nullptr) {
    *out_rev = rev;
  }
  return s == NATS_OK;
}

bool KvStore::put(const std::string& key, const std::vector<std::uint8_t>& bytes, std::uint64_t* out_rev) {
  return put(key, bytes.data(), bytes.size(), out_rev);
}

std::optional<std::vector<std::uint8_t>> KvStore::get(const std::string& key) const {
  if (kv_ == nullptr) {
    return std::nullopt;
  }
  kvEntry* e = nullptr;
  const natsStatus s = kvStore_Get(&e, kv_, key.c_str());
  if (s != NATS_OK || e == nullptr) {
    if (e) kvEntry_Destroy(e);
    return std::nullopt;
  }
  const void* data = kvEntry_Value(e);
  const int len = kvEntry_ValueLen(e);
  std::vector<std::uint8_t> out;
  if (data != nullptr && len > 0) {
    out.resize(static_cast<std::size_t>(len));
    std::memcpy(out.data(), data, static_cast<std::size_t>(len));
  }
  kvEntry_Destroy(e);
  return out;
}

bool KvStore::watch(const std::string& key_pattern, WatchCallback cb, bool include_history) {
  stop_watch();
  if (kv_ == nullptr) {
    return false;
  }
  kvWatcher* w = nullptr;
  kvWatchOptions opts;
  kvWatchOptions_Init(&opts);
  opts.IncludeHistory = include_history;
  opts.IgnoreDeletes = false;
  opts.Timeout = 0;
  const natsStatus s = kvStore_Watch(&w, kv_, key_pattern.c_str(), &opts);
  if (s != NATS_OK || w == nullptr) {
    spdlog::error("KV watch failed pattern={} err={}", key_pattern, natsStatus_GetText(s));
    return false;
  }
  watcher_ = w;
  stop_.store(false, std::memory_order_release);

  watch_thread_ = std::thread([this, cb = std::move(cb)]() mutable {
    while (!stop_.load(std::memory_order_acquire)) {
      kvEntry* e = nullptr;
      const natsStatus s2 = kvWatcher_Next(&e, watcher_, 500);
      if (stop_.load(std::memory_order_acquire)) {
        if (e) kvEntry_Destroy(e);
        break;
      }
      if (s2 == NATS_TIMEOUT) {
        if (e) kvEntry_Destroy(e);
        continue;
      }
      if (s2 != NATS_OK) {
        if (e) kvEntry_Destroy(e);
        continue;
      }
      if (e == nullptr) {
        continue;  // initial snapshot boundary
      }
      const char* key = kvEntry_Key(e);
      const void* val = kvEntry_Value(e);
      const int len = kvEntry_ValueLen(e);
      if (key && val && len >= 0) {
        std::vector<std::uint8_t> bytes;
        bytes.resize(static_cast<std::size_t>(len));
        if (len > 0) {
          std::memcpy(bytes.data(), val, static_cast<std::size_t>(len));
        }
        try {
          cb(std::string(key), bytes);
        } catch (...) {
        }
      }
      kvEntry_Destroy(e);
    }
  });

  return true;
}

void KvStore::stop_watch() {
  stop_.store(true, std::memory_order_release);
  if (watcher_ != nullptr) {
    kvWatcher_Stop(watcher_);
  }
  if (watch_thread_.joinable()) {
    watch_thread_.join();
  }
  if (watcher_ != nullptr) {
    kvWatcher_Destroy(watcher_);
    watcher_ = nullptr;
  }
}

}  // namespace f8::cppsdk
