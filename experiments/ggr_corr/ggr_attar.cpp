// gsta's GPU pattern router (ggr.cu) on a placed LEF/DEF design. The gcell grid, the capacity of every
// gcell edge on every layer and the pins' gcells are OpenROAD GlobalRouter's, computed from the files by
// gsta's FastRoute port (gr::build_design): tracks per gcell less obstructions (pins, OBS, special nets),
// then the layer adjustment.
//
// usage: ggr_attar <def> <out_prefix> --tech <tech.lef> --lef <cells.lef> [--lef ...]
//                  [--min 2] [--max 7] [--clk-min 4] [--adj 0.25] [--iters 30] [--maze 0] [--no-layers]
// out: <prefix>.json (grid, stats), <prefix>.cap.i32 [L][ny][nx] (3D capacity, layers M1..ML),
//      <prefix>.use.i32 [L][ny][nx] (3D usage after layer assignment), <prefix>.use2d.i32 [2][ny][nx] (H, V),
//      <prefix>.nets.tsv (net, clock, pins, unit gcell edges).
// Edge (x, y) of a horizontal layer joins gcells (x, y) and (x + 1, y); of a vertical one (x, y) and (x, y + 1).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "ggr.h"
#include "gr/gr.h"
#include "gr/gr_nets.h"
#include "gr/gr_setup.h"
#include "rcest.h"

using namespace gsta;

static int def_dbu(const std::string& path) {
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream ls(line);
    std::string a, b, c;
    int v;
    if (ls >> a >> b >> c >> v && a == "UNITS" && b == "DISTANCE" && c == "MICRONS") return v;
    if (a == "COMPONENTS") break;
  }
  return 1000;
}

template <class T>
static void write_raw(const std::string& path, const std::vector<T>& v) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: ggr_attar <def> <out_prefix> --tech <tech.lef> --lef <cells.lef> ... [options]\n");
    return 2;
  }
  const std::string def = argv[1], out = argv[2];
  std::string tech;
  std::vector<std::string> cell_lefs;
  int minl = 2, maxl = 7, clk_min = 4;
  float adj = 0.25f;
  GrOptions opt;
  for (int i = 3; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
      return argv[++i];
    };
    if (a == "--tech") tech = next();
    else if (a == "--lef") cell_lefs.push_back(next());
    else if (a == "--min") minl = std::stoi(next());
    else if (a == "--max") maxl = std::stoi(next());
    else if (a == "--clk-min") clk_min = std::stoi(next());
    else if (a == "--adj") adj = std::stof(next());
    else if (a == "--iters") opt.iters = std::stoi(next());
    else if (a == "--maze") opt.maze_iters = std::stoi(next());
    else if (a == "--no-layers") opt.layers = false;
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  if (tech.empty()) { std::fprintf(stderr, "--tech is required\n"); return 2; }
  if (minl != 2 || maxl != 7)  // ggr.cu's layer assignment is written for M2-M7, even layers horizontal
    std::fprintf(stderr, "warning: ggr.cu assigns layers M2-M7 only; --min/--max set the capacity range\n");

  auto t0 = std::chrono::steady_clock::now();
  auto lap = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

  // 1. OpenROAD's router input from the files
  gr::NetsParams np;
  np.min_layer = minl, np.max_layer = maxl, np.clock_min_layer = clk_min, np.clock_max_layer = maxl;
  np.dbu_per_micron = def_dbu(def);
  std::vector<std::string> lefs = {tech};
  lefs.insert(lefs.end(), cell_lefs.begin(), cell_lefs.end());
  gr::Design d;
  std::string err;
  if (!gr::build_design(def, lefs, tech, np, adj, d, &err)) {
    std::fprintf(stderr, "build_design: %s\n", err.c_str());
    return 1;
  }
  const double t_setup = lap();
  const int nx = d.nx, ny = d.ny, L = d.layers;
  std::printf("grid %d x %d, tile %d dbu, %d layers, %zu nets, setup %.2f s\n", nx, ny, d.tile, L, d.nets.size(), t_setup);

  // 3D capacity per layer in the gcell indexing (layer l: M(l+1))
  std::vector<int32_t> cap(size_t(L) * nx * ny, 0);
  for (int l = 0; l < L; l++) {
    const bool hz = d.ldir[l] == 'H';
    for (int y = 0; y < ny; y++)
      for (int x = 0; x < nx; x++) {
        int c = 0;
        if (hz && x + 1 < nx && !d.h3[l].empty()) c = d.h3[l][size_t(y) * (nx - 1) + x];
        if (!hz && y + 1 < ny && !d.v3[l].empty()) c = d.v3[l][size_t(y) * nx + x];
        cap[(size_t(l) * ny + y) * nx + x] = c;
      }
  }

  // 2. ggr's grid: capacities per layer M2..M7 (preferred direction), their sums per direction
  RcGrid G;
  G.tile = d.tile, G.x0 = d.xcorner, G.y0 = d.ycorner, G.dbu = float(np.dbu_per_micron);
  G.nx = nx, G.ny = ny;
  G.caph.assign(size_t(nx) * ny, 0.f), G.capv.assign(size_t(nx) * ny, 0.f);
  double sh = 0, sv = 0;
  for (int k = 2; k <= 7 && k <= L; k++) {
    const bool hz = d.ldir[k - 1] == 'H';
    if (hz != (k % 2 == 0)) {
      std::fprintf(stderr, "layer M%d is %c: ggr.cu takes even layers as horizontal\n", k, d.ldir[k - 1]);
      return 1;
    }
    G.capl[k].assign(size_t(nx) * ny, 0.f);
    for (size_t i = 0; i < G.capl[k].size(); i++) {
      const float c = float(cap[size_t(k - 1) * nx * ny + i]);
      G.capl[k][i] = c;
      (hz ? G.caph : G.capv)[i] += c;
      (hz ? sh : sv) += c;
    }
  }
  G.cap[0] = float(sh / (double(nx) * ny)), G.cap[1] = float(sv / (double(nx) * ny));

  // 3. one stage per net: its pins' gcell centers (um), as FastRoute sees them
  std::vector<std::vector<std::pair<float, float>>> stages(d.nets.size());
  std::vector<uint8_t> clk(d.nets.size(), 0);
  for (size_t i = 0; i < d.nets.size(); i++) {
    const gr::Net& n = d.nets[i];
    for (int p = 0; p < n.npins(); p++)
      stages[i].push_back({float((d.xcorner + n.px[p] * d.tile + d.tile / 2) / G.dbu),
                           float((d.ycorner + n.py[p] * d.tile + d.tile / 2) / G.dbu)});
    clk[i] = n.clock;
  }

  // 4. route
  std::vector<std::vector<long>> routes;
  const double t_r0 = lap();
  const GrStats st = global_route(G, stages, routes, opt, &clk);
  const double t_route = lap() - t_r0;

  std::vector<int32_t> use(size_t(L) * nx * ny, 0), use2d(size_t(2) * nx * ny, 0);
  for (const auto& r : routes)
    for (long k : r) {
      const int lay = int(k % 8);
      const long e = k / 8;
      const int dir = int(e / (long(nx) * ny));
      const long xy = e % (long(nx) * ny);
      use2d[size_t(dir) * nx * ny + xy]++;
      if (lay >= 1 && lay <= L) use[size_t(lay - 1) * nx * ny + xy]++;
    }
  long of2 = 0;
  for (int y = 0; y < ny; y++)
    for (int x = 0; x < nx; x++) {
      const size_t i = size_t(y) * nx + x;
      of2 += std::max(0, use2d[i] - int(G.caph[i])) + std::max(0, use2d[size_t(nx) * ny + i] - int(G.capv[i]));
    }
  long of3 = 0;
  for (size_t i = 0; i < use.size(); i++) of3 += std::max(0, use[i] - cap[i]);

  write_raw(out + ".cap.i32", cap);
  write_raw(out + ".use.i32", use);
  write_raw(out + ".use2d.i32", use2d);
  {
    std::ofstream f(out + ".nets.tsv");
    f << "net\tclock\tpins\tedges\n";
    for (size_t i = 0; i < d.nets.size(); i++)
      f << d.nets[i].name << '\t' << int(clk[i]) << '\t' << d.nets[i].npins() << '\t' << routes[i].size() << '\n';
  }
  {
    std::ofstream f(out + ".json");
    std::string ld(d.ldir.begin(), d.ldir.end());
    f << "{\"nx\": " << nx << ", \"ny\": " << ny << ", \"layers\": " << L << ", \"tile\": " << d.tile
      << ", \"x0\": " << d.xcorner << ", \"y0\": " << d.ycorner << ", \"dbu\": " << np.dbu_per_micron
      << ", \"ldir\": \"" << ld << "\", \"nets\": " << d.nets.size() << ", \"branches\": " << st.branches
      << ", \"wirelength_gcells\": " << st.wirelength << ", \"overflow_first\": " << st.overflow0
      << ", \"overflow_pattern\": " << st.overflow_pat << ", \"overflow_2d\": " << of2
      << ", \"overflow_3d\": " << of3 << ", \"la_overflow\": " << st.la_overflow << ", \"iters\": " << st.iters
      << ", \"setup_s\": " << t_setup << ", \"route_s\": " << t_route << ", \"adj\": " << adj
      << ", \"min_layer\": " << minl << ", \"max_layer\": " << maxl << "}\n";
  }
  std::printf("ggr: wirelength %ld gcell edges, 2D overflow %ld, 3D overflow %ld, route %.2f s (setup %.2f s)\n",
              st.wirelength, of2, of3, t_route, t_setup);
  return 0;
}
