#include "attn/comm.hpp"

#include <utility>

namespace attn {

Comm::Comm(MPI_Comm owned, Timers& t) : comm_(owned), timers_(&t) {
  MPI_Comm_size(comm_, &size_);
  MPI_Comm_rank(comm_, &rank_);
}

Comm& Comm::operator=(Comm&& o) noexcept {
  if (this != &o) {
    if (comm_ != MPI_COMM_NULL) MPI_Comm_free(&comm_);
    comm_ = std::exchange(o.comm_, MPI_COMM_NULL);
    timers_ = o.timers_;
    size_ = o.size_;
    rank_ = o.rank_;
  }
  return *this;
}

Comm::~Comm() {
  if (comm_ != MPI_COMM_NULL) MPI_Comm_free(&comm_);
}

Comm Comm::dup(MPI_Comm comm, Timers& t) {
  MPI_Comm out;
  MPI_Comm_dup(comm, &out);
  return Comm(out, t);
}

void Comm::wait(Request& r) {
  const double t0 = now();
  MPI_Wait(&r.handle, MPI_STATUS_IGNORE);
  finish(r.phase, t0, 0, r.idle);
}

void Comm::finish(Phase ph, double t0, double words, bool idle) {
  timers_->add_comm(ph, idle, t0, now());
  timers_->words[int(ph)] += words;
}

}  // namespace attn
