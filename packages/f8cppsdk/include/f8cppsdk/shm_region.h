#pragma once

#include <cstddef>
#include <string>

namespace f8::cppsdk {

class ShmRegion {
 public:
  ShmRegion() = default;
  ~ShmRegion();
  ShmRegion(const ShmRegion&) = delete;
  ShmRegion& operator=(const ShmRegion&) = delete;
  ShmRegion(ShmRegion&&) = delete;
  ShmRegion& operator=(ShmRegion&&) = delete;

  bool open_or_create(const std::string& name, std::size_t bytes);
  bool open_existing_readonly(const std::string& name, std::size_t bytes);
  bool open_existing_readwrite(const std::string& name, std::size_t bytes);
  // POSIX only: if enabled for a creator, close() will shm_unlink(name) after unmapping.
  // Readers should never enable this.
  void set_unlink_on_close(bool enabled) { unlink_on_close_ = enabled; }
  void close();

  void* data() const { return data_; }
  std::size_t size() const { return size_; }
  const std::string& name() const { return name_; }

 private:
  bool open_existing_impl(const std::string& name, std::size_t bytes, bool read_write);

  std::string name_;
  void* data_ = nullptr;
  std::size_t size_ = 0;

  // POSIX behavior flag; on Windows this is ignored.
  bool unlink_on_close_ = false;

#if defined(_WIN32)
  void* mapping_ = nullptr;
#else
  int fd_ = -1;
  bool owner_ = false;
  std::string posix_name_;
#endif
};

}  // namespace f8::cppsdk
