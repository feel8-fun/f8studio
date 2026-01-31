#include <gtest/gtest.h>

#include "f8cppsdk/main_thread_queue.h"

TEST(MainThreadQueue, DrainRunsAllTasksInOrder) {
  f8::cppsdk::MainThreadQueue q;
  int x = 0;
  q.post([&]() { x = x * 10 + 1; });
  q.post([&]() { x = x * 10 + 2; });
  q.post([&]() { x = x * 10 + 3; });

  const auto ran = q.drain();
  EXPECT_EQ(ran, 3u);
  EXPECT_EQ(x, 123);

  EXPECT_EQ(q.drain(), 0u);
}

TEST(MainThreadQueue, DrainRespectsMaxTasks) {
  f8::cppsdk::MainThreadQueue q;
  int x = 0;
  for (int i = 0; i < 5; ++i) {
    q.post([&]() { ++x; });
  }
  EXPECT_EQ(q.drain(2), 2u);
  EXPECT_EQ(x, 2);
  EXPECT_EQ(q.drain(2), 2u);
  EXPECT_EQ(x, 4);
  EXPECT_EQ(q.drain(2), 1u);
  EXPECT_EQ(x, 5);
}

