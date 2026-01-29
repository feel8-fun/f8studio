#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace f8::screencap::win32 {

struct RectI {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
};

struct MonitorInfo {
  int id = 0;           // 0..N-1 (stable ordering within process)
  std::string name;     // system display device name if available
  RectI rect;           // virtual desktop coordinates
  RectI work_rect;      // virtual desktop coordinates
  bool primary = false;
  std::uintptr_t handle = 0;  // HMONITOR on Windows
};

RectI virtual_screen_rect();

std::vector<MonitorInfo> enumerate_monitors();
bool try_get_monitor_rect(int display_id, RectI& out, std::string& err);
bool try_get_monitor_handle(int display_id, std::uintptr_t& out_hmonitor, std::string& err);

bool try_parse_csv_rect(const std::string& csv, RectI& out, std::string& err);   // x,y,w,h
bool try_parse_csv_size(const std::string& csv, int& out_w, int& out_h, std::string& err);  // w,h

// window_id is expected like: "win32:hwnd:0x0001234" or "0x0001234"
bool try_parse_window_id(const std::string& window_id, std::uintptr_t& out_hwnd, std::string& err);
bool try_get_window_rect(std::uintptr_t hwnd, RectI& out, std::string& out_title, std::uint32_t& out_pid,
                         std::string& err);

}  // namespace f8::screencap::win32
