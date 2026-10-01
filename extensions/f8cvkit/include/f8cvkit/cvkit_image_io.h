#pragma once

#include <string>

#include <opencv2/core.hpp>

namespace f8::cvkit {

struct ImageLoadResult {
  cv::Mat image;
  std::string error;
};

// Load an image from disk with explicit errors.
ImageLoadResult load_image_bgr(const std::string& path);

}  // namespace f8::cvkit

