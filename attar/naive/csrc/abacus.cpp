// Abacus legalization (Spindler, Schlichtmann, Johannes, ISPD 2008) of one height class on its bands, for
// attar.naive.lg (method='abacus'). A band is a row (or k rows for k-row cells) cut into free segments by obstacles.
// Cells go in order of their target x; each tries the bands outward from its target y and, in each band, the
// segments outward from its target x, and goes where its own displacement (|dx| + |dy|, um) after the trial
// placement is least. Inside a segment the cells keep their order and form clusters at the position that minimizes
// their squared displacement (the cluster collapse), clamped to the segment; at the end clusters are rounded to
// sites without overlap. A search stops in a direction once its lower bound reaches the best cost found.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

struct Cluster {
    double x, e, q, w;   // position, weight, sum of e * (target - offset), width
    int first;           // its first cell in the segment's list
};

struct Segment {
    double a, b;         // [a, b) in band sites
    double used = 0.0;
    std::vector<Cluster> cl;
    std::vector<int> cells;
};

inline double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

// the cell's x after appending it (target t, width w) to the segment, without changing the segment
double trial(const Segment& s, double t, double w) {
    double e = 1.0, q = t, cw = w;
    double x = clampd(q / e, s.a, s.b - cw);
    for (int i = (int)s.cl.size() - 1; i >= 0; --i) {
        const Cluster& c = s.cl[i];
        if (c.x + c.w <= x) break;
        q = c.q + q - e * c.w;
        e += c.e;
        cw += c.w;
        x = clampd(q / e, s.a, s.b - cw);
    }
    return x + cw - w;
}

void commit(Segment& s, int cell, double t, double w) {
    Cluster c{0.0, 1.0, t, w, (int)s.cells.size()};
    s.cells.push_back(cell);
    s.used += w;
    c.x = clampd(c.q / c.e, s.a, s.b - c.w);
    while (!s.cl.empty()) {
        Cluster& p = s.cl.back();
        if (p.x + p.w <= c.x) break;
        Cluster m{0.0, p.e + c.e, p.q + c.q - c.e * p.w, p.w + c.w, p.first};
        m.x = clampd(m.q / m.e, s.a, s.b - m.w);
        s.cl.pop_back();
        c = m;
    }
    s.cl.push_back(c);
}

}  // namespace

extern "C" int attar_abacus(int n, const double* gx, const double* gy, const double* w, int nb, const double* band_y,
                            const double* roff, int ns, const int64_t* seg_band, const double* s0, const double* s1,
                            double sw, int64_t* out_band, double* out_x, uint8_t* unfit) {
    // segments per band, by start
    std::vector<Segment> seg(ns);
    std::vector<std::vector<int>> by_band(nb);
    for (int i = 0; i < ns; ++i) {
        seg[i].a = s0[i];
        seg[i].b = s1[i];
        if (seg_band[i] >= 0 && seg_band[i] < nb && s1[i] > s0[i]) by_band[seg_band[i]].push_back(i);
    }
    std::vector<std::vector<double>> starts(nb);
    for (int b = 0; b < nb; ++b) {
        std::sort(by_band[b].begin(), by_band[b].end(), [&](int i, int j) { return seg[i].a < seg[j].a; });
        for (int i : by_band[b]) starts[b].push_back(seg[i].a);
    }
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int i, int j) { return gx[i] < gx[j]; });
    std::vector<int> where(n, -1);
    int failed = 0;
    for (int ci : order) {
        double best = INFINITY;
        int best_s = -1;
        const double wc = w[ci];
        int up = (int)(std::lower_bound(band_y, band_y + nb, gy[ci]) - band_y);
        int dn = up - 1;
        while (up < nb || dn >= 0) {
            // the nearer band next
            int b;
            double dy_up = up < nb ? std::fabs(band_y[up] - gy[ci]) : INFINITY;
            double dy_dn = dn >= 0 ? std::fabs(band_y[dn] - gy[ci]) : INFINITY;
            if (dy_up <= dy_dn) { b = up++; } else { b = dn--; }
            double dy = std::fabs(band_y[b] - gy[ci]);
            if (dy >= best) break;               // every band left is at least this far
            const std::vector<int>& L = by_band[b];
            if (L.empty()) continue;
            double t = gx[ci] - roff[b];         // target in band sites
            int j = (int)(std::upper_bound(starts[b].begin(), starts[b].end(), t) - starts[b].begin()) - 1;
            int l = j, r = j + 1;
            while (l >= 0 || r < (int)L.size()) {
                // the segment nearer to t next (lower bound: the distance to its usable span)
                double dl = INFINITY, dr = INFINITY;
                if (l >= 0) { const Segment& s = seg[L[l]]; dl = std::max(0.0, t - (s.b - wc)); }
                if (r < (int)L.size()) { const Segment& s = seg[L[r]]; dr = std::max(0.0, s.a - t); }
                int k;
                double lb;
                if (dl <= dr) { k = L[l--]; lb = dl; } else { k = L[r++]; lb = dr; }
                if (dy + lb * sw >= best) break;  // every segment left on this side is farther: both sides are
                Segment& s = seg[k];             // ordered by distance, so the next one is too
                if (s.b - s.a - s.used < wc - 1e-9) continue;
                double x = trial(s, t, wc);
                double cost = dy + std::fabs(x - t) * sw;
                if (cost < best) { best = cost; best_s = k; }
            }
        }
        if (best_s < 0) {
            unfit[ci] = 1;
            ++failed;
            continue;
        }
        unfit[ci] = 0;
        where[ci] = best_s;
        commit(seg[best_s], ci, gx[ci] - roff[seg_band[best_s]], wc);
    }
    // clusters to sites, cells packed inside them
    for (int k = 0; k < ns; ++k) {
        Segment& s = seg[k];
        int m = (int)s.cl.size();
        std::vector<double> xs(m);
        double cur = std::ceil(s.a - 1e-9);
        for (int i = 0; i < m; ++i) {
            xs[i] = std::max(std::round(s.cl[i].x), cur);
            cur = xs[i] + s.cl[i].w;
        }
        double lim = std::floor(s.b + 1e-9);
        for (int i = m - 1; i >= 0; --i) {
            xs[i] = std::min(xs[i], lim - s.cl[i].w);
            lim = xs[i];
        }
        for (int i = 0; i < m; ++i) {
            int end = i + 1 < m ? s.cl[i + 1].first : (int)s.cells.size();
            double x = xs[i];
            for (int c = s.cl[i].first; c < end; ++c) {
                int ci = s.cells[c];
                out_band[ci] = seg_band[k];
                out_x[ci] = x;
                x += w[ci];
            }
        }
    }
    return failed;
}
