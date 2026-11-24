#include "bench/report.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

#include "attn/costmodel.hpp"
#include "attn/kernels.hpp"

#ifdef __linux__
#include <sched.h>
#endif

namespace attn::bench {

namespace {

std::string sanitize(std::string s) {
  std::replace(s.begin(), s.end(), ',', ';');
  std::replace(s.begin(), s.end(), '\n', ' ');
  return s;
}

int rank_of(MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

std::string role_columns(const std::vector<Sample>& s, Role r) {
  const int i = slowest(s, r);
  if (i < 0) return "na,na,na";
  std::ostringstream o;
  o << std::setprecision(12) << s[i].compute << ',' << s[i].comm << ',' << s[i].idle_handoff + s[i].idle_tail;
  return o.str();
}

}  // namespace

RunInfo run_info(MPI_Comm comm) {
  RunInfo r;
  const char* job = std::getenv("ATTN_JOB_ID");  // set by experiments/run.py for local runs
  if (!job) job = std::getenv("SLURM_JOB_ID");
  r.job_id = job ? sanitize(job) : "local";
  char stamp[32];
  const std::time_t tt = std::time(nullptr);
  std::strftime(stamp, sizeof stamp, "%Y-%m-%dT%H:%M:%S", std::localtime(&tt));
  r.timestamp = stamp;
  char host[MPI_MAX_PROCESSOR_NAME];
  int len;
  MPI_Get_processor_name(host, &len);
  r.machine = sanitize(host);
  char lib[MPI_MAX_LIBRARY_VERSION_STRING];
  MPI_Get_library_version(lib, &len);
  r.mpi = sanitize(std::string(lib).substr(0, std::string(lib).find(',')));
  r.compiler = sanitize(__VERSION__);

  MPI_Comm node;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
  int node_rank;
  MPI_Comm_rank(node, &node_rank);
  MPI_Comm_size(node, &r.ranks_per_node);
  const int leader = node_rank == 0;
  MPI_Allreduce(&leader, &r.nodes, 1, MPI_INT, MPI_SUM, comm);
  MPI_Comm_free(&node);

  r.blas_threads = blas_threads();
  r.kernel_threads = kernel_threads();
#ifdef __linux__
  cpu_set_t mask;
  int cpus[2] = {-1, 0};
  if (sched_getaffinity(0, sizeof mask, &mask) == 0) cpus[0] = cpus[1] = CPU_COUNT(&mask);
  cpus[1] = -cpus[1];  // one MIN all-reduce gives min and max
  MPI_Allreduce(MPI_IN_PLACE, cpus, 2, MPI_INT, MPI_MIN, comm);
  if (cpus[0] > 0)
    r.cpus_per_rank = cpus[0] == -cpus[1] ? std::to_string(cpus[0])
                                          : std::to_string(cpus[0]) + "-" + std::to_string(-cpus[1]);
#endif
  // Binding policy as the launcher reports it: Slurm's srun, else Open MPI's mpirun (PRRTE), else unset.
  std::string bind;
  for (const char* var : {"SLURM_CPU_BIND_TYPE", "SLURM_HINT", "OMPI_MCA_hwloc_base_binding_policy",
                          "PRTE_MCA_hwloc_default_binding_policy", "OMP_PROC_BIND"})
    if (const char* v = std::getenv(var); v && *v) bind += (bind.empty() ? "" : " ") + std::string(var) + "=" + v;
  r.cpu_bind = bind.empty() ? "unset" : sanitize(bind);
  return r;
}

const char* const csv_header =
    "job_id,timestamp,machine,gpu,nodes,ranks_per_node,mpi,cuda,cublas,compiler,device,blas,alg,n,d1,d2,P,p1,p2,"
    "p3,T,map,dtype,seed,rep,t_total_max,crit_rank,crit_role,crit_compute,crit_comm,crit_idle_handoff,"
    "crit_idle_tail,crit_other,prod_compute,prod_comm,prod_idle,cons_compute,cons_comm,cons_idle,pred_words,"
    "thmA_words,thmB_words,blas_threads,kernel_threads,cpus_per_rank,cpu_bind";

std::string csv_row(const RunInfo& info, const Plan& p, uint64_t seed, int rep, const std::vector<Sample>& s) {
  const int crit = slowest(s);
  const Sample& x = s[crit];
  std::ostringstream r;
  r << std::setprecision(12);
  r << info.job_id << ',' << info.timestamp << ',' << info.machine << ",na," << info.nodes << ','
    << info.ranks_per_node << ',' << info.mpi << ",na,na," << info.compiler << ",cpu," << blas_name() << ','
    << p.alg << ',' << p.shape.n << ',' << p.shape.d1 << ',' << p.shape.d2 << ',' << p.P << ',' << p.grid.p1 << ','
    << p.grid.p2 << ',' << p.grid.p3 << ',' << p.T << ',' << p.map << ",f32," << seed << ',' << rep << ','
    << x.total << ',' << crit << ',' << name(x.role) << ',' << x.compute << ',' << x.comm << ',' << x.idle_handoff
    << ',' << x.idle_tail << ',' << x.other() << ',' << role_columns(s, Role::producer) << ','
    << role_columns(s, Role::consumer) << ',' << p.words << ',' << thm_a_words(p.shape, p.P) << ',';
  if (const double b = thm_b_words(p.shape, p.P); std::isnan(b)) r << "na";
  else r << b;
  r << ',';
  if (info.blas_threads < 0) r << "na";
  else r << info.blas_threads;
  r << ',' << info.kernel_threads << ',' << info.cpus_per_rank << ',' << info.cpu_bind;
  return r.str();
}

CsvWriter::CsvWriter(const std::string& path, MPI_Comm comm) : path_(path), active_(!path.empty() && rank_of(comm) == 0) {
  int same = 1;  // decided on rank 0, then shared so every rank throws together
  if (active_) {
    std::ifstream in(path_);
    std::string header;
    if (!in.good() || !std::getline(in, header)) std::ofstream(path_) << csv_header << '\n';
    else same = header == csv_header;
  }
  if (!path.empty()) MPI_Bcast(&same, 1, MPI_INT, 0, comm);
  require(same, path + " has a different CSV header (older attnbench?): write to a new file");
}

void CsvWriter::write(const std::string& row) {
  if (active_) std::ofstream(path_, std::ios::app) << row << '\n';
}

void write_timeline(const std::string& path, const RunInfo& info, int rep, const Attention& alg, MPI_Comm comm) {
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  std::ostringstream lines;
  lines << std::setprecision(12);
  for (const auto& e : alg.timers().events)
    lines << info.job_id << ',' << rep << ',' << rank << ',' << name(alg.role()) << ',' << e.block << ','
          << e.label << ',' << e.start << ',' << e.end << '\n';
  const std::string mine = lines.str();
  const int len = int(mine.size());
  std::vector<int> lens(size), offsets(size);
  MPI_Gather(&len, 1, MPI_INT, lens.data(), 1, MPI_INT, 0, comm);
  int sum = 0;
  for (int i = 0; i < size; ++i) offsets[i] = std::exchange(sum, sum + lens[i]);
  std::string all(rank == 0 ? sum : 0, '\0');
  MPI_Gatherv(mine.data(), len, MPI_CHAR, all.data(), lens.data(), offsets.data(), MPI_CHAR, 0, comm);
  if (rank == 0) std::ofstream(path) << "job_id,rep,rank,role,block_t,phase,t_start,t_end\n" << all;
}

}  // namespace attn::bench
