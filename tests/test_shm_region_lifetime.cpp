#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "f8cppsdk/shm_region.h"

namespace {

std::string unique_name(const char* prefix) {
#if defined(_WIN32)
  return std::string(prefix) + "_win";
#else
  return std::string(prefix) + "_" + std::to_string(static_cast<long long>(::getpid())) + "_x";
#endif
}

}  // namespace

TEST(ShmRegionLifetime, CreatorDoesNotUnlinkByDefault) {
#if defined(_WIN32)
  GTEST_SKIP() << "POSIX shm_unlink semantics only";
#else
  const std::string name = unique_name("f8_test_shmregion_keep");
  const std::size_t bytes = 4096;

  {
    f8::cppsdk::ShmRegion creator;
    ASSERT_TRUE(creator.open_or_create(name, bytes));
    // Default: do not unlink on close.
    creator.close();
  }

  {
    f8::cppsdk::ShmRegion reader;
    ASSERT_TRUE(reader.open_existing_readonly(name, bytes));
    reader.close();
  }

  // Cleanup explicitly.
  {
    f8::cppsdk::ShmRegion cleanup;
    ASSERT_TRUE(cleanup.open_or_create(name, bytes));
    cleanup.set_unlink_on_close(true);
    cleanup.close();
  }
#endif
}

TEST(ShmRegionLifetime, CreatorCanUnlinkWhenEnabled) {
#if defined(_WIN32)
  GTEST_SKIP() << "POSIX shm_unlink semantics only";
#else
  const std::string name = unique_name("f8_test_shmregion_unlink");
  const std::size_t bytes = 4096;

  {
    f8::cppsdk::ShmRegion creator;
    ASSERT_TRUE(creator.open_or_create(name, bytes));
    creator.set_unlink_on_close(true);
    creator.close();
  }

  {
    f8::cppsdk::ShmRegion reader;
    ASSERT_FALSE(reader.open_existing_readonly(name, bytes));
  }
#endif
}

