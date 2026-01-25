#include "f8cppsdk/time_utils.h"

#include <chrono>

namespace f8::cppsdk {

std::int64_t now_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

}  // namespace f8::cppsdk

