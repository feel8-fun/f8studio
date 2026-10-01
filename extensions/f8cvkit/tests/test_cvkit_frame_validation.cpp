#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "f8cppsdk/latest_video_frame_transport.h"
#include "service_runtime_utils.h"

namespace {

TEST(CvkitFrameValidation, AcceptsValidBgraFrame) {
  f8::cppsdk::LatestVideoFrame frame;
  frame.width = 4;
  frame.height = 3;
  frame.pitch = 16;
  frame.format = f8::cppsdk::kVideoFormatBgra32;
  frame.frame_id = 1;
  frame.payload.assign(static_cast<std::size_t>(frame.pitch) * static_cast<std::size_t>(frame.height), std::byte{0});

  const auto validation =
      f8::cvkit::service_runtime::validate_latest_video_frame(frame, f8::cppsdk::kVideoFormatBgra32, 4u);

  EXPECT_TRUE(validation.ok);
  EXPECT_EQ(validation.row_bytes, 16u);
}

TEST(CvkitFrameValidation, RejectsTransientInvalidFrameMetadata) {
  f8::cppsdk::LatestVideoFrame frame;
  frame.width = 0;
  frame.height = 3;
  frame.pitch = 16;
  frame.format = f8::cppsdk::kVideoFormatBgra32;
  frame.frame_id = 1;
  frame.payload.assign(48u, std::byte{0});

  const auto validation =
      f8::cvkit::service_runtime::validate_latest_video_frame(frame, f8::cppsdk::kVideoFormatBgra32, 4u);

  EXPECT_FALSE(validation.ok);
}

TEST(CvkitFrameValidation, RejectsTransientShortPayload) {
  f8::cppsdk::LatestVideoFrame frame;
  frame.width = 4;
  frame.height = 3;
  frame.pitch = 16;
  frame.format = f8::cppsdk::kVideoFormatBgra32;
  frame.frame_id = 1;
  frame.payload.assign(47u, std::byte{0});

  const auto validation =
      f8::cvkit::service_runtime::validate_latest_video_frame(frame, f8::cppsdk::kVideoFormatBgra32, 4u);

  EXPECT_FALSE(validation.ok);
}

TEST(CvkitFrameValidation, RejectsTransientFormatChanges) {
  f8::cppsdk::LatestVideoFrame frame;
  frame.width = 4;
  frame.height = 3;
  frame.pitch = 16;
  frame.format = f8::cppsdk::kVideoFormatFlow2F16;
  frame.frame_id = 1;
  frame.payload.assign(48u, std::byte{0});

  const auto validation =
      f8::cvkit::service_runtime::validate_latest_video_frame(frame, f8::cppsdk::kVideoFormatBgra32, 4u);

  EXPECT_FALSE(validation.ok);
}

}  // namespace
