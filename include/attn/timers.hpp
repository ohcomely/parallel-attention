#pragma once
#include <array>
#include <string_view>
#include <vector>

#include "attn/problem.hpp"

namespace attn {

double now();

struct Event {
  int block;
  std::string_view label;
  double start, end;  // seconds since Timers::origin
};

struct Timers {
  double compute = 0, comm = 0, idle_handoff = 0, idle_tail = 0, total = 0;
  int compute_calls = 0;
  std::array<double, num_phases> words{};  // logical words received, debug only
  std::array<double, num_phases> phase_time{};
  std::array<int, num_phases> phase_calls{};

  int block = -1;
  bool record = false;
  double origin = 0;
  std::vector<Event> events;

  void reset();
  double other() const { return total - compute - comm - idle_handoff; }
  void add_compute(std::string_view label, double t0, double t1);
  void add_comm(Phase ph, bool idle, double t0, double t1);
};

class Timed {
 public:
  Timed(Timers& t, std::string_view label) : t_(t), label_(label), t0_(now()) {}
  ~Timed() { t_.add_compute(label_, t0_, now()); }
  Timed(const Timed&) = delete;
  Timed& operator=(const Timed&) = delete;

 private:
  Timers& t_;
  std::string_view label_;
  double t0_;
};

}  // namespace attn
