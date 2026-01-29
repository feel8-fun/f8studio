#include "f8cppsdk/shm_region.h"

#include <spdlog/spdlog.h>

#if defined(_WIN32)
#define NOMINMAX
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace f8::cppsdk {

ShmRegion::~ShmRegion() { close(); }

bool ShmRegion::open_or_create(const std::string& name, std::size_t bytes) {
  close();
  if (name.empty() || bytes == 0) return false;
  name_ = name;
  size_ = bytes;

#if defined(_WIN32)
  const std::wstring wname(name.begin(), name.end());
  HANDLE hmap =
      CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, static_cast<DWORD>((bytes >> 32) & 0xFFFFFFFF),
                         static_cast<DWORD>(bytes & 0xFFFFFFFF), wname.c_str());
  if (!hmap) {
    spdlog::error("CreateFileMapping failed name={} err={}", name, GetLastError());
    return false;
  }
  void* ptr = MapViewOfFile(hmap, FILE_MAP_ALL_ACCESS, 0, 0, bytes);
  if (!ptr) {
    spdlog::error("MapViewOfFile failed name={} err={}", name, GetLastError());
    CloseHandle(hmap);
    return false;
  }
  mapping_ = hmap;
  data_ = ptr;
  return true;
#else
  std::string shm_name = name;
  if (!shm_name.empty() && shm_name[0] != '/') shm_name = "/" + shm_name;
  posix_name_ = shm_name;
  int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
  if (fd < 0) {
    spdlog::error("shm_open failed name={}", shm_name);
    return false;
  }
  if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
    spdlog::error("ftruncate failed name={}", shm_name);
    ::close(fd);
    return false;
  }
  void* ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    spdlog::error("mmap failed name={}", shm_name);
    ::close(fd);
    return false;
  }
  fd_ = fd;
  data_ = ptr;
  owner_ = true;
  return true;
#endif
}

bool ShmRegion::open_existing_readonly(const std::string& name, std::size_t bytes) {
  return open_existing_impl(name, bytes, false);
}

bool ShmRegion::open_existing_readwrite(const std::string& name, std::size_t bytes) {
  return open_existing_impl(name, bytes, true);
}

bool ShmRegion::open_existing_impl(const std::string& name, std::size_t bytes, bool read_write) {
  close();
  if (name.empty() || bytes == 0) return false;
  name_ = name;
  size_ = bytes;

#if defined(_WIN32)
  const std::wstring wname(name.begin(), name.end());
  const DWORD access = read_write ? FILE_MAP_ALL_ACCESS : FILE_MAP_READ;
  HANDLE hmap = OpenFileMappingW(access, FALSE, wname.c_str());
  if (!hmap) {
    spdlog::error("OpenFileMapping failed name={} err={}", name, GetLastError());
    return false;
  }
  void* ptr = MapViewOfFile(hmap, access, 0, 0, bytes);
  if (!ptr) {
    spdlog::error("MapViewOfFile failed name={} err={}", name, GetLastError());
    CloseHandle(hmap);
    return false;
  }
  mapping_ = hmap;
  data_ = ptr;
  return true;
#else
  std::string shm_name = name;
  if (!shm_name.empty() && shm_name[0] != '/') shm_name = "/" + shm_name;
  posix_name_ = shm_name;
  const int oflag = read_write ? O_RDWR : O_RDONLY;
  int fd = shm_open(shm_name.c_str(), oflag, 0666);
  if (fd < 0) {
    spdlog::error("shm_open(existing) failed name={}", shm_name);
    return false;
  }
  struct stat st;
  if (fstat(fd, &st) != 0) {
    spdlog::error("fstat failed name={}", shm_name);
    ::close(fd);
    return false;
  }
  if (static_cast<std::size_t>(st.st_size) < bytes) {
    spdlog::error("shm size too small name={} want={} actual={}", shm_name, bytes, static_cast<std::size_t>(st.st_size));
    ::close(fd);
    return false;
  }
  const int prot = read_write ? (PROT_READ | PROT_WRITE) : PROT_READ;
  void* ptr = mmap(nullptr, bytes, prot, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    spdlog::error("mmap(existing) failed name={}", shm_name);
    ::close(fd);
    return false;
  }
  fd_ = fd;
  data_ = ptr;
  owner_ = false;
  return true;
#endif
}

void ShmRegion::close() {
#if defined(_WIN32)
  if (data_) {
    UnmapViewOfFile(data_);
    data_ = nullptr;
  }
  if (mapping_) {
    CloseHandle(static_cast<HANDLE>(mapping_));
    mapping_ = nullptr;
  }
#else
  if (data_) {
    munmap(data_, size_);
    data_ = nullptr;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
  if (owner_ && unlink_on_close_ && !posix_name_.empty()) {
    (void)shm_unlink(posix_name_.c_str());
  }
  owner_ = false;
  unlink_on_close_ = false;
  posix_name_.clear();
#endif
  size_ = 0;
  name_.clear();
}

}  // namespace f8::cppsdk
