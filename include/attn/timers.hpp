#pragma once
#include <array>
#include <string_view>
#include <vector>

namespace attn {

// Communication phases, used as labels for timers and word counters.
enum class Phase { Q, K, V, S_rs, S_ag, S_handoff, rowstat, O, ring_kv, count };
constexpr int num_phases = int(Phase::count);
constexpr std::array<std::string_view, num_phases> phase_names{
    "Q", "K", "V", "S_rs", "S_ag", "S_handoff", "rowstat", "O", "ring_kv"};
inline std::string_view name(Phase p) { return phase_names[int(p)]; }

double now();  // seconds, monotonic

struct Event {
  int block;
  std::string_view label;
  double start, end;  // seconds since Timers::origin
};

// Per-rank, per-run instrumentation. Every Attention owns one; Comm and Timed write into it.
struct Timers {
  double compute = 0, comm = 0, idle_handoff = 0, idle_tail = 0, total = 0;
  int compute_calls = 0;
  std::array<double, num_phases> words{};  // logical words received (debug, see README conventions)
  std::array<double, num_phases> phase_time{};
  std::array<int, num_phases> phase_calls{};

  int block = -1;       // current pipeline block / ring step, tagged onto events
  bool record = false;  // collect `events` (timeline)
  double origin = 0;
  std::vector<Event> events;

  void reset();  // clears everything except `record`
  double other() const { return total - compute - comm - idle_handoff; }
  void add_compute(std::string_view label, double t0, double t1);
  void add_comm(Phase ph, bool idle, double t0, double t1);
};

// Times a compute scope.
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
