#include "attn/grid.hpp"

#include <algorithm>
#include <charconv>
#include <vector>

namespace attn {

std::string Dims::str() const {
  return std::to_string(p1) + "x" + std::to_string(p2) + "x" + std::to_string(p3);
}

Dims parse_dims(std::string_view s) {
  std::vector<int> parts;
  while (!s.empty()) {
    const auto x = s.find('x');
    const auto tok = s.substr(0, x);
    int v = 0;
    const auto [ptr, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), v);
    require(ec == std::errc{} && ptr == tok.data() + tok.size() && v > 0, "bad grid: " + std::string(tok));
    parts.push_back(v);
    s = x == std::string_view::npos ? std::string_view{} : s.substr(x + 1);
  }
  require(parts.size() == 2 || parts.size() == 3, "grid must be p1xp2 or p1xp2xp3");
  return {parts[0], parts[1], parts.size() == 3 ? parts[2] : 1};
}

Coords coords_of(int rank, Dims d, std::string_view map) {
  std::string sorted(map);
  std::sort(sorted.begin(), sorted.end());
  require(sorted == "ijk", "map must be a permutation of ijk: " + std::string(map));
  const int extent[3] = {d.p1, d.p2, d.p3};
  int c[3];
  for (char letter : map) {
    const int axis = letter - 'i';
    c[axis] = rank % extent[axis];
    rank /= extent[axis];
  }
  return {c[0], c[1], c[2]};
}

namespace {
Comm split(MPI_Comm comm, int color, int key, Timers& t) {
  MPI_Comm sub;
  MPI_Comm_split(comm, color, key, &sub);
  return Comm(sub, t);
}
}  // namespace

Grid make_grid(MPI_Comm comm, Dims d, std::string_view map, Timers& t) {
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  require(d.size() == size, "grid " + d.str() + " needs " + std::to_string(d.size()) + " ranks, have " +
                                std::to_string(size));
  const Coords at = coords_of(rank, d, map);
  Grid g{d, at, {}, {}, {}};
  g.ci = split(comm, at.j * d.p3 + at.k, at.i, t);
  g.cj = split(comm, at.i * d.p3 + at.k, at.j, t);
  g.ck = split(comm, at.i * d.p2 + at.j, at.k, t);
  return g;
}

}  // namespace attn
