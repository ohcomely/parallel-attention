#pragma once
#include <mpi.h>

#include "attn/problem.hpp"
#include "attn/timers.hpp"

namespace attn {

template <class T>
MPI_Datatype mpi_type();
template <>
inline MPI_Datatype mpi_type<float>() { return MPI_FLOAT; }
template <>
inline MPI_Datatype mpi_type<double>() { return MPI_DOUBLE; }

enum class Op { max, sum };

struct Request {
  MPI_Request handle = MPI_REQUEST_NULL;
  Phase phase = Phase::S_handoff;
  bool idle = false;  // time in wait() counts as idle_handoff instead of comm
};

// The only place that issues MPI communication inside algorithms. Owns its MPI_Comm.
class Comm {
 public:
  Comm() = default;
  Comm(MPI_Comm owned, Timers& t);
  Comm(Comm&& o) noexcept { *this = std::move(o); }
  Comm& operator=(Comm&& o) noexcept;
  Comm(const Comm&) = delete;
  Comm& operator=(const Comm&) = delete;
  ~Comm();
  static Comm dup(MPI_Comm comm, Timers& t);

  int size() const { return size_; }
  int rank() const { return rank_; }

  template <class T>
  void allgather(const T* send, T* recv, int64_t count, Phase ph) {
    const double t0 = now();
    MPI_Allgather(send, to_int(count), mpi_type<T>(), recv, to_int(count), mpi_type<T>(), comm_);
    finish(ph, t0, double(size_ - 1) * count);
  }

  template <class T>
  void reduce_scatter(const T* send, T* recv, int64_t count, Phase ph) {
    const double t0 = now();
    MPI_Reduce_scatter_block(send, recv, to_int(count), mpi_type<T>(), MPI_SUM, comm_);
    finish(ph, t0, double(size_ - 1) * count);
  }

  template <class T>
  void allreduce(T* buf, int64_t count, Op op, Phase ph) {
    const double t0 = now();
    MPI_Allreduce(MPI_IN_PLACE, buf, to_int(count), mpi_type<T>(), op == Op::max ? MPI_MAX : MPI_SUM, comm_);
    finish(ph, t0, 2.0 * (size_ - 1) / size_ * count);
  }

  template <class T>
  Request isend(const T* buf, int64_t count, int dest, int tag, Phase ph) {
    const double t0 = now();
    Request r{MPI_REQUEST_NULL, ph, false};
    MPI_Isend(buf, to_int(count), mpi_type<T>(), dest, tag, comm_, &r.handle);
    finish(ph, t0, 0);
    return r;
  }

  template <class T>
  Request irecv(T* buf, int64_t count, int src, int tag, Phase ph, bool idle_on_wait) {
    const double t0 = now();
    Request r{MPI_REQUEST_NULL, ph, idle_on_wait};
    MPI_Irecv(buf, to_int(count), mpi_type<T>(), src, tag, comm_, &r.handle);
    finish(ph, t0, double(count));
    return r;
  }

  void wait(Request& r);

  template <class T>
  void sendrecv(const T* send, int64_t send_count, int dest, T* recv, int64_t recv_count, int src, Phase ph) {
    const double t0 = now();
    MPI_Sendrecv(send, to_int(send_count), mpi_type<T>(), dest, 0, recv, to_int(recv_count), mpi_type<T>(), src, 0,
                 comm_, MPI_STATUS_IGNORE);
    finish(ph, t0, double(recv_count));
  }

 private:
  void finish(Phase ph, double t0, double words, bool idle = false);

  MPI_Comm comm_ = MPI_COMM_NULL;
  Timers* timers_ = nullptr;
  int size_ = 0, rank_ = 0;
};

}  // namespace attn
