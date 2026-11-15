// OpenROAD's estimate-parasitics Steiner tree (stt::SteinerTreeBuilder, alpha 0.3 -> pdr::primDijkstra) for nets of
// any size, and the RC moments of the tree of pi segments OpenSTA reduces it with (attar.sta.tree_elmore), on the CPU.
//
// Exact port of OpenROAD src/stt/src/pdr/src/neighbors.cpp and pd.cpp, including what decides ties:
//   - getNearestNeighbors: the column sweep with a row max-tree;
//   - buildSpanningTree: heap key (weight, parent id, node id), a node's entry replaced when a new weight is <=;
//   - steinerize: lemon::ListGraph incidence order (new arcs first, changeU / changeV move an arc to the front of the
//     new node's list, NodeIt newest first) and boost::heap::d_ary_heap<arity 2, mutable> sift rules, so candidates
//     with equal (gain, x, y) pop in OpenROAD's order;
//   - splitDegree4Nodes adds zero-length edges: electrically nothing, skipped.
// Pins are sorted by (x, y) first (EstimateParasitics::makeSteinerTree; equal locations keep their input order).
// Built by attar/steiner_big.py with g++ -O3 -fopenmp -shared; a C ABI for ctypes.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>

namespace {

using std::vector;

// ---------------------------------------------------------------- neighbors.cpp
class RowMaxTree {
 public:
  explicit RowMaxTree(int num_rows) {
    while (size_ < num_rows) size_ *= 2;
    max_x_.assign(2 * (size_t)size_, std::numeric_limits<int>::min());
  }
  int x(int row) const { return max_x_[size_ + row]; }
  void set(int row, int x) {
    size_t node = size_ + row;
    max_x_[node] = x;
    for (node /= 2; node > 0; node /= 2) max_x_[node] = std::max(max_x_[2 * node], max_x_[2 * node + 1]);
  }
  int first(int lo, int hi, int min_x) const { return lo >= hi ? -1 : first(1, 0, size_, lo, hi, min_x); }
  int last(int lo, int hi, int min_x) const { return lo >= hi ? -1 : last(1, 0, size_, lo, hi, min_x); }

 private:
  int first(int node, int node_lo, int node_hi, int lo, int hi, int min_x) const {
    if (node_hi <= lo || hi <= node_lo || max_x_[node] <= min_x) return -1;
    if (node_hi - node_lo == 1) return node_lo;
    int mid = (node_lo + node_hi) / 2;
    int row = first(2 * node, node_lo, mid, lo, hi, min_x);
    return row >= 0 ? row : first(2 * node + 1, mid, node_hi, lo, hi, min_x);
  }
  int last(int node, int node_lo, int node_hi, int lo, int hi, int min_x) const {
    if (node_hi <= lo || hi <= node_lo || max_x_[node] <= min_x) return -1;
    if (node_hi - node_lo == 1) return node_lo;
    int mid = (node_lo + node_hi) / 2;
    int row = last(2 * node + 1, mid, node_hi, lo, hi, min_x);
    return row >= 0 ? row : last(2 * node, node_lo, mid, lo, hi, min_x);
  }
  int size_ = 1;
  vector<int> max_x_;
};

vector<vector<int>> nearest_neighbors(const vector<int>& X, const vector<int>& Y) {
  const int n = X.size();
  vector<vector<int>> nb(n);
  auto link = [&nb](int a, int b) { nb[a].push_back(b); nb[b].push_back(a); };
  vector<int> sorted(n);
  std::iota(sorted.begin(), sorted.end(), 0);
  std::sort(sorted.begin(), sorted.end(), [&](int i, int j) {
    return std::make_tuple(X[i], Y[i], i) < std::make_tuple(X[j], Y[j], j);
  });
  vector<int> uniq;
  uniq.reserve(n);
  for (int pt : sorted) {
    if (!uniq.empty() && X[uniq.back()] == X[pt] && Y[uniq.back()] == Y[pt]) link(uniq.back(), pt);
    else uniq.push_back(pt);
  }
  vector<int> row_ys;
  for (int pt : uniq) row_ys.push_back(Y[pt]);
  std::sort(row_ys.begin(), row_ys.end());
  row_ys.erase(std::unique(row_ys.begin(), row_ys.end()), row_ys.end());
  const int num_rows = row_ys.size();
  auto row_of = [&](int y) { return (int)(std::lower_bound(row_ys.begin(), row_ys.end(), y) - row_ys.begin()); };
  RowMaxTree swept(num_rows);
  vector<int> row_pt(num_rows, -1), col_rows;
  const int uc = uniq.size();
  for (int cb = 0; cb < uc;) {
    const int col_x = X[uniq[cb]];
    int ce = cb;
    col_rows.clear();
    while (ce < uc && X[uniq[ce]] == col_x) { col_rows.push_back(row_of(Y[uniq[ce]])); ++ce; }
    for (int k = cb; k < ce; ++k) {
      const int pt = uniq[k], row = col_rows[k - cb];
      const bool has_below = k > cb, has_above = k + 1 < ce;
      const int lo_row = has_below ? col_rows[k - cb - 1] + 1 : 0;
      const int hi_row = has_above ? col_rows[k - cb + 1] : num_rows;
      if (has_above) link(pt, uniq[k + 1]);
      int min_x = std::numeric_limits<int>::min();
      for (int r = swept.first(row, hi_row, min_x); r >= 0; r = swept.first(r + 1, hi_row, min_x)) {
        link(pt, row_pt[r]);
        min_x = swept.x(r);
      }
      min_x = swept.x(row);
      for (int r = swept.last(lo_row, row, min_x); r >= 0; r = swept.last(lo_row, r, min_x)) {
        link(pt, row_pt[r]);
        min_x = swept.x(r);
      }
    }
    for (int k = cb; k < ce; ++k) {
      swept.set(col_rows[k - cb], col_x);
      row_pt[col_rows[k - cb]] = uniq[k];
    }
    cb = ce;
  }
  for (auto& v : nb) std::sort(v.begin(), v.end());
  return nb;
}

// ---------------------------------------------------------------- lemon::ListGraph (undirected), as used by pd.cpp
struct Graph {
  struct NodeT { int first_out; };
  struct ArcT { int target, next_out, prev_out; };
  vector<NodeT> nodes;
  vector<ArcT> arcs;
  int addNode() { nodes.push_back({-1}); return (int)nodes.size() - 1; }
  int addEdge(int u, int v) {
    int n = arcs.size();
    arcs.push_back({}); arcs.push_back({});
    arcs[n].target = u; arcs[n | 1].target = v;
    arcs[n].next_out = nodes[v].first_out;
    if (nodes[v].first_out != -1) arcs[nodes[v].first_out].prev_out = n;
    arcs[n].prev_out = -1; nodes[v].first_out = n;
    arcs[n | 1].next_out = nodes[u].first_out;
    if (nodes[u].first_out != -1) arcs[nodes[u].first_out].prev_out = (n | 1);
    arcs[n | 1].prev_out = -1; nodes[u].first_out = (n | 1);
    return n / 2;
  }
  int u(int e) const { return arcs[2 * e].target; }
  int v(int e) const { return arcs[2 * e + 1].target; }
  int opposite(int node, int e) const { return u(e) == node ? v(e) : u(e); }
  void changeV(int e, int n) {
    ArcT& a = arcs[2 * e];
    if (a.next_out != -1) arcs[a.next_out].prev_out = a.prev_out;
    if (a.prev_out != -1) arcs[a.prev_out].next_out = a.next_out;
    else nodes[arcs[(2 * e) | 1].target].first_out = a.next_out;
    if (nodes[n].first_out != -1) arcs[nodes[n].first_out].prev_out = 2 * e;
    arcs[(2 * e) | 1].target = n;
    arcs[2 * e].prev_out = -1;
    arcs[2 * e].next_out = nodes[n].first_out;
    nodes[n].first_out = 2 * e;
  }
  void changeU(int e, int n) {
    ArcT& a = arcs[(2 * e) | 1];
    if (a.next_out != -1) arcs[a.next_out].prev_out = a.prev_out;
    if (a.prev_out != -1) arcs[a.prev_out].next_out = a.next_out;
    else nodes[arcs[2 * e].target].first_out = a.next_out;
    if (nodes[n].first_out != -1) arcs[nodes[n].first_out].prev_out = ((2 * e) | 1);
    arcs[2 * e].target = n;
    arcs[(2 * e) | 1].prev_out = -1;
    arcs[(2 * e) | 1].next_out = nodes[n].first_out;
    nodes[n].first_out = ((2 * e) | 1);
  }
  // IncEdgeIt: arc a of the node's list; its edge a / 2, running node v(e) if a is odd else u(e)
  int running(int a) const { return (a & 1) ? v(a / 2) : u(a / 2); }
};

// ---------------------------------------------------------------- pd.cpp
struct Cand {
  int gain = 0;
  int node = -1, e1 = -1, e2 = -1;
  int sx = 0, sy = 0;   // odb::Point() is (0, 0) for gain-0 candidates
};

Cand best_steiner_for_node(const Graph& g, int node, const vector<int>& PX, const vector<int>& PY) {
  const int nx = PX[node], ny = PY[node];
  Cand best;
  for (int a1 = g.nodes[node].first_out; a1 != -1; a1 = g.arcs[a1].next_out) {
    const int n1 = g.running(a1);
    for (int a2 = g.arcs[a1].next_out; a2 != -1; a2 = g.arcs[a2].next_out) {
      const int n2 = g.running(a2);
      int sx = nx, sy = ny;
      if (std::min(PX[n1], PX[n2]) > nx) sx = std::min(PX[n1], PX[n2]);
      else if (std::max(PX[n1], PX[n2]) < nx) sx = std::max(PX[n1], PX[n2]);
      if (std::min(PY[n1], PY[n2]) > ny) sy = std::min(PY[n1], PY[n2]);
      else if (std::max(PY[n1], PY[n2]) < ny) sy = std::max(PY[n1], PY[n2]);
      const int gain = std::abs(sx - nx) + std::abs(sy - ny);
      if (gain > best.gain) best = {gain, node, a1 / 2, a2 / 2, sx, sy};
    }
  }
  return best;
}

// boost::heap::d_ary_heap<Cand, arity<2>, mutable_<true>, compare<CmpCandidate>>: a max-heap on (gain, x, y)
struct CandHeap {
  vector<Cand> ent;       // entries by handle
  vector<int> q, pos;     // heap of handles; handle -> heap index
  static bool less(const Cand& a, const Cand& b) {
    return std::make_tuple(a.gain, a.sx, a.sy) < std::make_tuple(b.gain, b.sx, b.sy);
  }
  bool lt(int i, int j) const { return less(ent[q[i]], ent[q[j]]); }
  void swp(int i, int j) { std::swap(q[i], q[j]); pos[q[i]] = i; pos[q[j]] = j; }
  void siftup(int i) {
    while (i != 0) {
      int p = (i - 1) / 2;
      if (lt(p, i)) { swp(p, i); i = p; } else return;
    }
  }
  void siftdown(int i) {
    const int n = q.size();
    while (2 * i + 1 < n) {
      int c = 2 * i + 1;                                     // std::max_element: the first largest child
      if (c + 1 < n && lt(c, c + 1)) c = c + 1;
      if (!lt(c, i)) { swp(c, i); i = c; } else return;
    }
  }
  int push(const Cand& c) {
    int h = ent.size();
    ent.push_back(c); pos.push_back(q.size()); q.push_back(h);
    siftup(q.size() - 1);
    return h;
  }
  const Cand& top() const { return ent[q[0]]; }
  void pop() {
    swp(0, q.size() - 1);
    pos[q.back()] = -1; q.pop_back();
    if (q.empty()) return;
    siftdown(0);
  }
  void update(int h) {
    int i = pos[h];
    if (i == 0) { siftdown(0); return; }
    if (lt((i - 1) / 2, i)) siftup(i); else siftdown(i);
  }
  bool empty() const { return q.empty(); }
};

// The tree: graph and node points after buildSpanningTree and steinerize. Pins are nodes 0..n-1.
void pd_tree(const vector<int>& X, const vector<int>& Y, int driver, float alpha, Graph& g, vector<int>& PX,
             vector<int>& PY) {
  const int n = X.size();
  PX = X; PY = Y;
  for (int i = 0; i < n; ++i) g.addNode();
  const auto nn = nearest_neighbors(X, Y);
  // buildSpanningTree: the heap's order is its key (weight, parent, node): unique, so a set reproduces it
  vector<char> visited(n, 0), inheap(n, 0);
  vector<float> w(n, 0.f);
  vector<int> pl(n, 0), par(n, -1);
  std::set<std::tuple<float, int, int>> heap;
  heap.insert({0.f, -1, driver}); inheap[driver] = 1;
  int num_visited = 0;
  while (!heap.empty()) {
    auto [ew, eparent, enode] = *heap.begin();
    heap.erase(heap.begin());
    visited[enode] = 1;
    ++num_visited;
    if (eparent != -1) g.addEdge(eparent, enode);
    if (num_visited == n) break;
    const int epl = pl[enode];
    for (int nb : nn[enode]) {
      if (visited[nb]) continue;
      const int len = std::abs(X[nb] - X[enode]) + std::abs(Y[nb] - Y[enode]);
      const int npl = len + epl;
      const float nw = len + alpha * epl;
      if (!inheap[nb]) {
        inheap[nb] = 1; w[nb] = nw; pl[nb] = npl; par[nb] = enode;
        heap.insert({nw, enode, nb});
      } else if (nw <= w[nb]) {
        heap.erase({w[nb], par[nb], nb});
        w[nb] = nw; pl[nb] = npl; par[nb] = enode;
        heap.insert({nw, enode, nb});
      }
    }
  }
  // steinerize
  CandHeap heap2;
  vector<int> handle(n, -1);
  for (int node = (int)g.nodes.size() - 1; node >= 0; --node)   // lemon NodeIt: newest first
    handle[node] = heap2.push(best_steiner_for_node(g, node, PX, PY));
  while (!heap2.empty()) {
    const Cand best = heap2.top();
    heap2.pop();
    if (best.gain == 0) break;
    const int opp1 = g.opposite(best.node, best.e1), opp2 = g.opposite(best.node, best.e2);
    bool new_node = false;
    int st;
    if (best.sx == PX[opp1] && best.sy == PY[opp1]) st = opp1;
    else if (best.sx == PX[opp2] && best.sy == PY[opp2]) st = opp2;
    else {
      st = g.addNode(); PX.push_back(best.sx); PY.push_back(best.sy); handle.push_back(-1);
      new_node = true;
    }
    if (st != opp1) { if (g.u(best.e1) == best.node) g.changeU(best.e1, st); else g.changeV(best.e1, st); }
    if (st != opp2) { if (g.u(best.e2) == best.node) g.changeU(best.e2, st); else g.changeV(best.e2, st); }
    if (new_node) {
      g.addEdge(st, best.node);
      handle[st] = heap2.push(best_steiner_for_node(g, st, PX, PY));
    }
    handle[best.node] = heap2.push(best_steiner_for_node(g, best.node, PX, PY));
    for (int node : {opp1, opp2}) {
      heap2.ent[handle[node]] = best_steiner_for_node(g, node, PX, PY);
      heap2.update(handle[node]);
    }
  }
}

}  // namespace

extern "C" {

// Per net (CSR ptr over pins): pin locations (dbu), the driver's index in the net, per pin V capacitances and its via
// resistance. Writes per pin its Elmore delay (V), per net the driver's pi model (V x 3: near C, R, far C) and the
// tree's wire length (um). r, c: wire RC per um. Returns 0.
int attar_pd_rc(int n_nets, const int64_t* ptr, const int64_t* drv, const int32_t* x, const int32_t* y, int V,
                const float* cap, const float* rv, double r, double c, double dbu, float alpha, float* elm, float* pi,
                float* len, int threads) {
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
  for (int k = 0; k < n_nets; ++k) {
    const int64_t b = ptr[k];
    const int n = (int)(ptr[k + 1] - b);
    if (n < 2) { len[k] = 0.f; continue; }
    // pins sorted by (x, y), equal locations in input order
    vector<int> ord(n);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int i, int j) {
      return std::make_pair(x[b + i], y[b + i]) < std::make_pair(x[b + j], y[b + j]);
    });
    vector<int> X(n), Y(n);
    int driver = 0;
    for (int i = 0; i < n; ++i) {
      X[i] = x[b + ord[i]]; Y[i] = y[b + ord[i]];
      if (ord[i] == drv[k]) driver = i;
    }
    Graph g;
    vector<int> PX, PY;
    pd_tree(X, Y, driver, alpha, g, PX, PY);
    const int N = g.nodes.size();
    // the tree from the driver: parent and order (BFS)
    vector<int> parent(N, -1), order;
    vector<double> el(N, 0.0);
    vector<char> seen(N, 0);
    order.reserve(N);
    order.push_back(driver); seen[driver] = 1;
    for (size_t h = 0; h < order.size(); ++h) {
      const int u = order[h];
      for (int a = g.nodes[u].first_out; a != -1; a = g.arcs[a].next_out) {
        const int w = g.running(a);
        if (!seen[w]) {
          seen[w] = 1; parent[w] = u; order.push_back(w);
          el[w] = (std::abs(PX[w] - PX[u]) + std::abs(PY[w] - PY[u])) / dbu;
        }
      }
    }
    double total = 0.0;
    vector<double> cw(N, 0.0);
    for (int w = 0; w < N; ++w) {
      total += el[w];
      if (parent[w] >= 0) { cw[w] += c * el[w] / 2; cw[parent[w]] += c * el[w] / 2; }
    }
    len[k] = (float)total;
    const double rd = rv[b + ord[driver]];
    for (int v = 0; v < V; ++v) {
      vector<double> cp(N, 0.0), rp(N, 0.0), cn(N), sub(N), m1(N), m1p(N), qq(N), m2(N);
      for (int i = 0; i < n; ++i) { cp[i] = cap[(b + ord[i]) * V + v]; rp[i] = rv[b + ord[i]]; }
      double y1 = 0.0;
      for (int w = 0; w < N; ++w) { cn[w] = cw[w] + cp[w]; y1 += cn[w]; }
      // subtree sums (reverse BFS), then root-to-node sums of R x subtree (BFS)
      auto moments = [&](const vector<double>& val, double base, vector<double>& out) {
        for (int w = 0; w < N; ++w) sub[w] = val[w];
        for (int h = N - 1; h > 0; --h) { const int w = order[h]; sub[parent[w]] += sub[w]; }
        out[driver] = base;
        for (int h = 1; h < N; ++h) { const int w = order[h]; out[w] = out[parent[w]] + r * el[w] * sub[w]; }
      };
      moments(cn, rd * y1, m1);
      double sq = 0.0;
      for (int w = 0; w < N; ++w) { m1p[w] = m1[w] + rp[w] * cp[w]; qq[w] = cw[w] * m1[w] + cp[w] * m1p[w]; sq += qq[w]; }
      moments(qq, rd * sq, m2);
      double y3 = 0.0;
      for (int w = 0; w < N; ++w) y3 += cw[w] * m2[w] + cp[w] * (m2[w] + rp[w] * cp[w] * m1p[w]);
      const double y2 = -sq;
      const bool ok = (y3 > 1e-12) && (y2 < -1e-12);
      double c2 = ok ? y2 * y2 / y3 : 0.0;
      const double rr = ok ? -y3 * y3 / (y2 * y2 * y2) : 0.0;
      c2 = std::min(c2, y1);
      float* pk = pi + ((int64_t)k * V + v) * 3;
      pk[0] = (float)(y1 - c2); pk[1] = (float)rr; pk[2] = (float)c2;
      for (int i = 0; i < n; ++i) elm[(b + ord[i]) * V + v] = (float)m1p[i];
    }
  }
  return 0;
}

}  // extern "C"
