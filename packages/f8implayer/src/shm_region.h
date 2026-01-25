#pragma once

#include <cstddef>
#include <string>

namespace f8::implayer {

class ShmRegion {
 public:
  ShmRegion() = default;
  ~ShmRegion();
  ShmRegion(const ShmRegion&) = delete;
  ShmRegion& operator=(const ShmRegion&) = delete;
  ShmRegion(ShmRegion&&) = delete;
  ShmRegion& operator=(ShmRegion&&) = delete;

  bool open_or_create(const std::string& name, std::size_t bytes);
  void close();

  void* data() const { return data_; }
  std::size_t size() const { return size_; }
  const std::string& name() const { return name_; }

 private:
  std::string name_;
  void* data_ = nullptr;
  std::size_t size_ = 0;

#if defined(_WIN32)
  void* mapping_ = nullptr;
#else
  int fd_ = -1;
#endif
};

}  // namespace f8::implayer

