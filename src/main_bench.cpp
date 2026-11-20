#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "attn/bench.hpp"
#include "attn/costmodel.hpp"
#include "attn/kernels.hpp"
#include "attn/stats.hpp"

using namespace attn;

namespace {

struct Options {
  Config cfg;
  bool grid_auto = false;
  int warmup = 5, reps = 20, max_reps = 100;
  double ci_target = 0.05;
  bool check = false;
  std::string out, timeline;
};

const char* usage =
    "attnbench --alg {2d,3d,split,ring} --n N --d D [--d2 D2] [--grid p1xp2xp3|auto] [--T T]\n"
    "          [--warmup 5] [--reps 20] [--max-reps 100] [--ci-target 0.05] [--map kji] [--seed S]\n"
    "          [--check] [--out results.csv] [--timeline timeline.csv]\n";

Options parse(int argc, char** argv) {
  Options o;
  int64_t d2 = 0;
  for (int a = 1; a < argc; ++a) {
    const std::string key = argv[a];
    if (key == "--check") {
      o.check = true;
      continue;
    }
    require(a + 1 < argc, "missing value for " + key);
    const std::string v = argv[++a];
    if (key == "--alg") o.cfg.alg = v;
    else if (key == "--n") o.cfg.prob.n = std::stoll(v);
    else if (key == "--d") o.cfg.prob.d1 = std::stoll(v);
    else if (key == "--d2") d2 = std::stoll(v);
    else if (key == "--grid" && v == "auto") o.grid_auto = true;
    else if (key == "--grid") o.cfg.grid = parse_dims(v);
    else if (key == "--T") o.cfg.T = std::stoi(v);
    else if (key == "--map") o.cfg.map = v;
    else if (key == "--seed") o.cfg.prob.seed = std::stoull(v);
    else if (key == "--warmup") o.warmup = std::stoi(v);
    else if (key == "--reps") o.reps = std::stoi(v);
    else if (key == "--max-reps") o.max_reps = std::stoi(v);
    else if (key == "--ci-target") o.ci_target = std::stod(v);
    else if (key == "--out") o.out = v;
    else if (key == "--timeline") o.timeline = v;
    else throw std::invalid_argument("unknown option " + key + "\n" + usage);
  }
  o.cfg.prob.d2 = d2 ? d2 : o.cfg.prob.d1;
  if (o.cfg.alg != "split") o.cfg.T = 1;
  require(o.reps > 0 && o.warmup >= 0, "reps must be positive");
  o.max_reps = std::max(o.max_reps, o.reps);
  return o;
}

std::string sanitize(std::string s) {
  std::replace(s.begin(), s.end(), ',', ';');
  std::replace(s.begin(), s.end(), '\n', ' ');
  return s;
}

struct RunInfo {
  std::string job_id, timestamp, machine, mpi, compiler;
  int nodes = 1, ranks_per_node = 1;
};

RunInfo run_info() {
  RunInfo r;
  const char* job = std::getenv("SLURM_JOB_ID");
  r.job_id = job ? job : "local";
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
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
  int node_rank;
  MPI_Comm_rank(node, &node_rank);
  MPI_Comm_size(node, &r.ranks_per_node);
  const int leader = node_rank == 0;
  MPI_Allreduce(&leader, &r.nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Comm_free(&node);
  return r;
}

enum Field { total, compute, comm, idle_handoff, idle_tail, role, num_fields };
using Sample = std::array<double, num_fields>;

std::vector<Sample> gather_samples(const Timers& t, Role r) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const Sample mine{t.total, t.compute, t.comm, t.idle_handoff, t.idle_tail, double(int(r))};
  std::vector<Sample> all(rank == 0 ? size : 0);
  MPI_Gather(mine.data(), num_fields, MPI_DOUBLE, all.data(), num_fields, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return all;
}

// Index of the slowest rank among those with the given role (any role if `r` is Role::all).
int slowest(const std::vector<Sample>& s, Role r) {
  int best = -1;
  for (int i = 0; i < int(s.size()); ++i)
    if ((r == Role::all || int(s[i][role]) == int(r)) && (best < 0 || s[i][total] > s[best][total])) best = i;
  return best;
}

std::string role_columns(const std::vector<Sample>& s, Role r) {
  const int i = slowest(s, r);
  if (i < 0) return "na,na,na";
  std::ostringstream o;
  o << s[i][compute] << ',' << s[i][comm] << ',' << s[i][idle_handoff] + s[i][idle_tail];
  return o.str();
}

const char* csv_header =
    "job_id,timestamp,machine,gpu,nodes,ranks_per_node,mpi,cuda,cublas,compiler,device,blas,alg,n,d1,d2,P,p1,p2,"
    "p3,T,map,dtype,seed,rep,t_total_max,crit_rank,crit_role,crit_compute,crit_comm,crit_idle_handoff,"
    "crit_idle_tail,crit_other,prod_compute,prod_comm,prod_idle,cons_compute,cons_comm,cons_idle,pred_words,"
    "thmA_words,thmB_words";

std::string csv_row(const RunInfo& info, const Options& o, int P, int rep, const std::vector<Sample>& s) {
  const auto& c = o.cfg;
  const bool ring = c.alg == "ring";
  const int crit = slowest(s, Role::all);
  const Sample& x = s[crit];
  std::ostringstream r;
  r << std::setprecision(12);
  r << info.job_id << ',' << info.timestamp << ',' << info.machine << ",na," << info.nodes << ','
    << info.ranks_per_node << ',' << info.mpi << ",na,na," << info.compiler << ",cpu," << blas_name() << ','
    << c.alg << ',' << c.prob.n << ',' << c.prob.d1 << ',' << c.prob.d2 << ',' << P << ','
    << (ring ? P : c.grid.p1) << ',' << (ring ? 1 : c.grid.p2) << ',' << (ring ? 1 : c.grid.p3) << ',' << c.T << ','
    << c.map << ",f32," << c.prob.seed << ',' << rep << ',' << x[total] << ',' << crit << ','
    << name(Role(int(x[role]))) << ',' << x[compute] << ',' << x[comm] << ',' << x[idle_handoff] << ','
    << x[idle_tail] << ',' << x[total] - x[compute] - x[comm] - x[idle_handoff] << ','
    << role_columns(s, Role::producer) << ',' << role_columns(s, Role::consumer) << ','
    << predicted_words(c, P) << ',' << thm_a_words(c.prob, P) << ',';
  if (const double b = thm_b_words(c.prob, P); std::isnan(b)) r << "na\n";
  else r << b << '\n';
  return r.str();
}

void write_timeline(const std::string& path, const RunInfo& info, int rep, const Timers& t, Role r) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::ostringstream lines;
  lines << std::setprecision(12);
  for (const auto& e : t.events)
    lines << info.job_id << ',' << rep << ',' << rank << ',' << name(r) << ',' << e.block << ',' << e.label << ','
          << e.start << ',' << e.end << '\n';
  const std::string mine = lines.str();
  const int len = int(mine.size());
  std::vector<int> lens(size), offsets(size);
  MPI_Gather(&len, 1, MPI_INT, lens.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  int sum = 0;
  for (int i = 0; i < size; ++i) offsets[i] = std::exchange(sum, sum + lens[i]);
  std::string all(rank == 0 ? sum : 0, '\0');
  MPI_Gatherv(mine.data(), len, MPI_CHAR, all.data(), lens.data(), offsets.data(), MPI_CHAR, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    std::ofstream f(path);
    f << "job_id,rep,rank,role,block_t,phase,t_start,t_end\n" << all;
  }
}

int bench(Options o) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (o.grid_auto && o.cfg.alg != "ring") {
    const auto g = best_grid(o.cfg.alg, o.cfg.prob, size, o.cfg.T);
    require(g.has_value(), "no valid grid for " + o.cfg.alg + " on " + std::to_string(size) + " ranks");
    o.cfg.grid = *g;
  }
  const RunInfo info = run_info();
  Timers t;
  auto alg = make_algorithm(o.cfg, MPI_COMM_WORLD, t);

  for (int w = 0; w < o.warmup; ++w) run_timed(*alg, t, MPI_COMM_WORLD);

  std::ofstream csv;
  if (rank == 0 && !o.out.empty()) {
    const bool fresh = !std::ifstream(o.out).good();
    csv.open(o.out, std::ios::app);
    if (fresh) csv << csv_header << '\n';
  }
  // At least `reps` repetitions, then continue until the median's 95% CI is within ci_target, up to max_reps.
  std::vector<double> totals;
  for (int rep = 0, more = 1; more; ++rep) {
    t.record = !o.timeline.empty() && rep == 0;
    run_timed(*alg, t, MPI_COMM_WORLD);
    const auto samples = gather_samples(t, alg->role());
    if (rank == 0) {
      totals.push_back(samples[slowest(samples, Role::all)][total]);
      if (csv.is_open()) csv << csv_row(info, o, size, rep, samples);
    }
    if (t.record) write_timeline(o.timeline, info, rep, t, alg->role());
    if (rank == 0)
      more = rep + 1 < o.reps || (rep + 1 < o.max_reps && relative_halfwidth(median_ci(totals)) > o.ci_target);
    MPI_Bcast(&more, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    const MedianCI m = median_ci(totals);
    std::printf("%s n=%lld d1=%lld d2=%lld P=%d grid=%s T=%d map=%s: median t_total_max %.4g s, 95%% CI [%.4g, %.4g]%s, "
                "%zu reps, predicted words %.0f\n",
                o.cfg.alg.c_str(), (long long)o.cfg.prob.n, (long long)o.cfg.prob.d1, (long long)o.cfg.prob.d2, size,
                o.cfg.alg == "ring" ? "-" : o.cfg.grid.str().c_str(), o.cfg.T, o.cfg.map.c_str(), m.median, m.lo, m.hi,
                m.exact ? "" : " (too few reps)", totals.size(), predicted_words(o.cfg, size));
  }

  if (o.check) {
    const double err = max_rel_error(*alg, o.cfg.prob, MPI_COMM_WORLD);
    const bool ok = err < 1e-4;
    if (rank == 0) std::printf("check: max rel err %.3e -> %s\n", err, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rc = 0;
  try {
    rc = bench(parse(argc, argv));
  } catch (const std::exception& e) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) std::fprintf(stderr, "error: %s\n", e.what());
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  MPI_Finalize();
  return rc;
}
