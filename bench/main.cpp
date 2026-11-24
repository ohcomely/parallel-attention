// attnbench: times one or more attention configurations and writes results.csv / timeline.csv.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "attn/costmodel.hpp"
#include "attn/registry.hpp"
#include "bench/harness.hpp"
#include "bench/inputs.hpp"
#include "bench/reference.hpp"
#include "bench/report.hpp"

using namespace attn;
using namespace attn::bench;

namespace {

const char* const usage = R"(usage:
  attnbench --alg ALG --n N --d D [options]       time one configuration
  attnbench --batch FILE [options]                time every line of FILE (one set of options per line)
  attnbench --alg ALG --n N --d D --plan [--P P]  print grid and predicted words, do not run
  attnbench --list                                list algorithms

problem:     --alg ALG  --n N  --d D1  [--d2 D2]  [--seed 1]
algorithm:   [--grid auto|p1xp2xp3]  [--T 1]  [--map kji]
repetitions: [--warmup 5]  [--reps 20]  [--max-reps 100]  [--ci-target 0.05]
output:      [--out results.csv]  [--timeline timeline.csv]  [--check]

In a batch file, '#' starts a comment and each line's options override those given on the command line.
)";

struct Args {
  std::string alg;
  Shape shape;
  Options opts;
  uint64_t seed = 1;
  Repetitions reps;
  bool check = false;
  std::string out, timeline;
};

struct Command {
  Args args;
  std::string batch;
  bool plan_only = false, list = false, help = false;
  int plan_P = 0;
};

// Applies `tokens` on top of `cmd`. Flags without a value: --check, --plan, --list, --help.
void parse_into(const std::vector<std::string>& tokens, Command& cmd) {
  Args& a = cmd.args;
  for (size_t i = 0; i < tokens.size(); ++i) {
    const std::string& key = tokens[i];
    if (key == "--check") { a.check = true; continue; }
    if (key == "--plan") { cmd.plan_only = true; continue; }
    if (key == "--list") { cmd.list = true; continue; }
    if (key == "--help" || key == "-h") { cmd.help = true; continue; }
    require(i + 1 < tokens.size(), "missing value for " + key);
    const std::string& v = tokens[++i];
    if (key == "--alg") a.alg = v;
    else if (key == "--n") a.shape.n = std::stoll(v);
    else if (key == "--d") a.shape.d1 = std::stoll(v);
    else if (key == "--d2") a.shape.d2 = std::stoll(v);
    else if (key == "--seed") a.seed = std::stoull(v);
    else if (key == "--grid") a.opts.grid = v == "auto" ? std::nullopt : std::optional(parse_dims(v));
    else if (key == "--T") a.opts.T = std::stoi(v);
    else if (key == "--map") a.opts.map = v;
    else if (key == "--warmup") a.reps.warmup = std::stoi(v);
    else if (key == "--reps") a.reps.reps = std::stoi(v);
    else if (key == "--max-reps") a.reps.max_reps = std::stoi(v);
    else if (key == "--ci-target") a.reps.ci_target = std::stod(v);
    else if (key == "--out") a.out = v;
    else if (key == "--timeline") a.timeline = v;
    else if (key == "--batch") cmd.batch = v;
    else if (key == "--P") cmd.plan_P = std::stoi(v);
    else throw std::invalid_argument("unknown option " + key + "\n" + usage);
  }
}

std::vector<std::string> split_words(const std::string& line) {
  std::istringstream in(line.substr(0, line.find('#')));
  std::vector<std::string> words;
  for (std::string w; in >> w;) words.push_back(w);
  return words;
}

void print_plan(const Plan& p) {
  std::printf("%s n=%lld d1=%lld d2=%lld P=%d grid=%s T=%d: predicted words %.0f, Theorem A %.0f", p.alg.c_str(),
              (long long)p.shape.n, (long long)p.shape.d1, (long long)p.shape.d2, p.P, p.grid.str().c_str(), p.T,
              p.words, thm_a_words(p.shape, p.P));
  if (const double b = thm_b_words(p.shape, p.P); !std::isnan(b)) std::printf(", Theorem B %.0f", b);
  std::printf("\n");
}

// Times one configuration. Returns false if --check failed.
bool bench(const Args& a, const RunInfo& info, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  auto alg = make(a.alg, comm, a.shape, a.opts);
  fill_generated(*alg, a.seed);
  const Plan& p = alg->plan();

  CsvWriter csv(a.out, comm);
  const Measurement m = measure(*alg, comm, a.reps, !a.timeline.empty(), [&](int rep, const auto& samples) {
    if (rank == 0) csv.write(csv_row(info, p, a.seed, rep, samples));
    if (rep == 0 && !a.timeline.empty()) write_timeline(a.timeline, info, rep, *alg, comm);
  });
  if (rank == 0) {
    const MedianCI& t = m.t_total_max;
    std::printf("%s n=%lld d1=%lld d2=%lld P=%d grid=%s T=%d map=%s: median t_total_max %.4g s, 95%% CI [%.4g, %.4g]%s, "
                "%d reps, predicted words %.0f\n",
                p.alg.c_str(), (long long)p.shape.n, (long long)p.shape.d1, (long long)p.shape.d2, p.P,
                p.grid.str().c_str(), p.T, p.map.c_str(), t.median, t.lo, t.hi, t.exact ? "" : " (too few reps)",
                m.reps, p.words);
  }
  if (!a.check) return true;
  const double err = max_rel_error(*alg, a.seed, comm);
  const bool ok = err < 1e-4;
  if (rank == 0) std::printf("check: max rel err %.3e -> %s\n", err, ok ? "PASS" : "FAIL");
  return ok;
}

int run(int argc, char** argv) {
  const MPI_Comm comm = MPI_COMM_WORLD;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  Command cmd;
  parse_into(std::vector<std::string>(argv + 1, argv + argc), cmd);
  if (cmd.help || argc == 1) {
    if (rank == 0) std::printf("%s", usage);
    return 0;
  }
  if (cmd.list) {
    if (rank == 0)
      for (const AlgorithmSpec* s : algorithms())
        std::printf("%-6s %s\n", std::string(s->name).c_str(), std::string(s->description).c_str());
    return 0;
  }
  if (cmd.plan_only) {
    if (rank == 0) print_plan(plan(cmd.args.alg, cmd.args.shape, cmd.plan_P ? cmd.plan_P : size, cmd.args.opts));
    return 0;
  }

  std::vector<Args> configs;
  if (cmd.batch.empty()) {
    configs.push_back(cmd.args);
  } else {
    std::ifstream in(cmd.batch);
    require(in.good(), "cannot read batch file " + cmd.batch);
    for (std::string line; std::getline(in, line);)
      if (const auto words = split_words(line); !words.empty()) {
        Command c{.args = cmd.args};
        parse_into(words, c);
        require(c.batch.empty() && !c.plan_only && !c.list && !c.help,
                "--batch, --plan, --list and --help are not allowed in a batch file");
        configs.push_back(c.args);
      }
  }

  const RunInfo info = run_info(comm);
  int failures = 0;
  for (const Args& a : configs) {
    try {
      failures += !bench(a, info, comm);
    } catch (const std::invalid_argument& e) {  // invalid configuration: raised identically on every rank
      if (rank == 0) std::fprintf(stderr, "error: %s\n", e.what());
      ++failures;
    }
  }
  return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  std::setvbuf(stdout, nullptr, _IOLBF, 0);  // one line per run, visible in Slurm output files as it happens
  int rc = 0;
  try {
    rc = run(argc, argv);
  } catch (const std::invalid_argument& e) {  // bad options or configuration: raised identically on every rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) std::fprintf(stderr, "error: %s\n", e.what());
    rc = 2;
  } catch (const std::exception& e) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) std::fprintf(stderr, "error: %s\n", e.what());
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  MPI_Finalize();
  return rc;
}
