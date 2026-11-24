#pragma once
#include <mpi.h>

#include <string>
#include <vector>

#include "attn/attention.hpp"
#include "bench/harness.hpp"

// Output files: results.csv (one row per repetition, all statistics recomputed offline by tools/stats.py) and
// timeline.csv (per-rank stage timestamps of one repetition).
namespace attn::bench {

struct RunInfo {
  std::string job_id, timestamp, machine, mpi, compiler;
  int nodes = 1, ranks_per_node = 1;
  int blas_threads = -1, kernel_threads = 1;  // per rank (rank 0); -1 = unknown
  std::string cpus_per_rank = "na";           // CPUs in each rank's affinity mask: "8", or "min-max" if they differ
  std::string cpu_bind = "na";                // binding policy from the launcher's environment
};

RunInfo run_info(MPI_Comm comm);  // collective

extern const char* const csv_header;
std::string csv_row(const RunInfo& info, const Plan& p, uint64_t seed, int rep, const std::vector<Sample>& s);

// Appends rows to `path`, writing the header if the file is new (rank 0 only). Refuses a file whose header
// differs (written by an older attnbench): use a new output directory.
class CsvWriter {
 public:
  CsvWriter(const std::string& path, MPI_Comm comm);
  void write(const std::string& row);

 private:
  std::string path_;
  bool active_;
};

// Gathers every rank's timeline events of the last run and writes them to `path` on rank 0. Collective.
void write_timeline(const std::string& path, const RunInfo& info, int rep, const Attention& alg, MPI_Comm comm);

}  // namespace attn::bench
