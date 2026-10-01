#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace f8::screencap::x11 {

struct RectI {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
};

struct DisplayInfo {
  int id = 0;
  std::string name;
  bool primary = false;
  RectI rect;
  RectI work_rect;
};

bool try_parse_csv_rect(const std::string& csv, RectI& out, std::string& err);
bool try_parse_csv_size(const std::string& csv, int& out_w, int& out_h, std::string& err);

bool try_parse_window_id(const std::string& window_id, std::uint64_t& out_xid, std::string& err);

std::vector<DisplayInfo> enumerate_displays(std::string& err);

bool try_get_window_rect(std::uint64_t xid, RectI& out, std::string& title, std::uint32_t& pid, std::string& err);

}  // namespace f8::screencap::x11

