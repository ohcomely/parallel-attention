#include "attn/timers.hpp"

#include <chrono>

namespace attn {

double now() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void Timers::reset() {
  const bool keep_record = record;
  *this = Timers{};
  record = keep_record;
}

void Timers::add_compute(std::string_view label, double t0, double t1) {
  compute += t1 - t0;
  ++compute_calls;
  if (record) events.push_back({block, label, t0 - origin, t1 - origin});
}

void Timers::add_comm(Phase ph, bool idle, double t0, double t1) {
  (idle ? idle_handoff : comm) += t1 - t0;
  phase_time[int(ph)] += t1 - t0;
  ++phase_calls[int(ph)];
  if (record) events.push_back({block, name(ph), t0 - origin, t1 - origin});
}

}  // namespace attn
