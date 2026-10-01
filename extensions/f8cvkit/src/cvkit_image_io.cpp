#include "f8cvkit/cvkit_image_io.h"

#include <opencv2/imgcodecs.hpp>

namespace f8::cvkit {

ImageLoadResult load_image_bgr(const std::string& path) {
  ImageLoadResult out;
  out.image = cv::imread(path, cv::IMREAD_COLOR);
  if (!out.image.empty()) {
    return out;
  }
  out.error = "imread failed: " + path;
  return out;
}

}  // namespace f8::cvkit

