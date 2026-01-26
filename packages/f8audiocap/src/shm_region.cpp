#include "shm_region.h"

#include <spdlog/spdlog.h>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace f8::audiocap {

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
#endif
  size_ = 0;
  name_.clear();
}

}  // namespace f8::audiocap

