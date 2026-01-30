#pragma once

#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <utility>

namespace f8::cppsdk {

// Minimal cross-thread task queue.
//
// - `post(...)` may be called from any thread.
// - `drain(...)` must be called from the desired "main thread" (e.g. service tick thread).
class MainThreadQueue final {
 public:
  using Task = std::function<void()>;

  void post(Task task) {
    if (!task) return;
    std::lock_guard<std::mutex> lock(mu_);
    q_.push_back(std::move(task));
  }

  // Execute up to `max_tasks` tasks. If `max_tasks==0`, drain everything.
  std::size_t drain(std::size_t max_tasks = 0) {
    std::deque<Task> local;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (q_.empty()) return 0;
      if (max_tasks == 0 || max_tasks >= q_.size()) {
        local.swap(q_);
      } else {
        for (std::size_t i = 0; i < max_tasks && !q_.empty(); ++i) {
          local.push_back(std::move(q_.front()));
          q_.pop_front();
        }
      }
    }

    std::size_t ran = 0;
    while (!local.empty()) {
      Task t = std::move(local.front());
      local.pop_front();
      try {
        t();
      } catch (...) {
      }
      ++ran;
    }
    return ran;
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mu_);
    q_.clear();
  }

 private:
  std::mutex mu_;
  std::deque<Task> q_;
};

}  // namespace f8::cppsdk

