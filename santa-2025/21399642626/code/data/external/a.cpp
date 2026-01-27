// Tree Packer v18 - PARALLEL + AGGRESSIVE BACK PROPAGATION
// + Free-area & Protrusion removal & reinsertion heuristic
// + Edge-based slide compaction (outline-aware)
// Compile example:
//   OMP_NUM_THREADS=32 g++ -fopenmp -O3 -march=native -std=c++17 -o a.exe a.cpp

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

constexpr int MAX_N = 200;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;

alignas(64) const double TX[NV] = {
    0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,
    -0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125
};
alignas(64) const double TY[NV] = {
    0.8,0.5,0.5,0.25,0.25,0,0,-0.2,
    -0.2,0,0,0.25,0.25,0.5,0.5
};

struct FastRNG {
    uint64_t s[2];
    FastRNG(uint64_t seed = 42) {
        s[0] = seed ^ 0x853c49e6748fea9bULL;
        s[1] = (seed * 0x9e3779b97f4a7c15ULL) ^ 0xc4ceb9fe1a85ec53ULL;
    }
    inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        uint64_t s0 = s[0], s1 = s[1], r = s0 + s1;
        s1 ^= s0;
        s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
        s[1] = rotl(s1, 37);
        return r;
    }
    inline double rf() { return (next() >> 11) * 0x1.0p-53; }
    inline double rf2() { return rf() * 2.0 - 1.0; }
    inline int ri(int n) { return (int)(next() % (uint64_t)n); }
    inline double gaussian() {
        double u1 = rf() + 1e-10, u2 = rf();
        return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    }
};

struct Poly {
    double px[NV], py[NV];
    double x0, y0, x1, y1;
};

inline void getPoly(double cx, double cy, double deg, Poly& q) {
    double rad = deg * (PI / 180.0);
    double s = sin(rad), c = cos(rad);
    double minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
    for (int i = 0; i < NV; i++) {
        double x = TX[i] * c - TY[i] * s + cx;
        double y = TX[i] * s + TY[i] * c + cy;
        q.px[i] = x;
        q.py[i] = y;
        if (x < minx) minx = x;
        if (x > maxx) maxx = x;
        if (y < miny) miny = y;
        if (y > maxy) maxy = y;
    }
    q.x0 = minx; q.y0 = miny; q.x1 = maxx; q.y1 = maxy;
}

inline bool pip(double px, double py, const Poly& q) {
    bool in = false;
    int j = NV - 1;
    for (int i = 0; i < NV; i++) {
        if ((q.py[i] > py) != (q.py[j] > py) &&
            px < (q.px[j] - q.px[i]) * (py - q.py[i]) / (q.py[j] - q.py[i]) + q.px[i])
            in = !in;
        j = i;
    }
    return in;
}

inline bool segInt(double ax, double ay, double bx, double by,
                   double cx, double cy, double dx, double dy) {
    double d1 = (dx-cx)*(ay-cy) - (dy-cy)*(ax-cx);
    double d2 = (dx-cx)*(by-cy) - (dy-cy)*(bx-cx);
    double d3 = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
    double d4 = (bx-ax)*(dy-ay) - (by-ay)*(dx-ax);
    return ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0));
}

inline bool overlap(const Poly& a, const Poly& b) {
    if (a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0) return false;
    for (int i = 0; i < NV; i++) {
        if (pip(a.px[i], a.py[i], b)) return true;
        if (pip(b.px[i], b.py[i], a)) return true;
    }
    for (int i = 0; i < NV; i++) {
        int ni = (i + 1) % NV;
        for (int j = 0; j < NV; j++) {
            int nj = (j + 1) % NV;
            if (segInt(a.px[i], a.py[i], a.px[ni], a.py[ni],
                      b.px[j], b.py[j], b.px[nj], b.py[nj])) return true;
        }
    }
    return false;
}

struct Cfg {
    int n;
    double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];
    double gx0, gy0, gx1, gy1;

    inline void upd(int i) { getPoly(x[i], y[i], a[i], pl[i]); }
    inline void updAll() { for (int i = 0; i < n; i++) upd(i); updGlobal(); }

    inline void updGlobal() {
        gx0 = gy0 = 1e9;
        gx1 = gy1 = -1e9;
        for (int i = 0; i < n; i++) {
            if (pl[i].x0 < gx0) gx0 = pl[i].x0;
            if (pl[i].x1 > gx1) gx1 = pl[i].x1;
            if (pl[i].y0 < gy0) gy0 = pl[i].y0;
            if (pl[i].y1 > gy1) gy1 = pl[i].y1;
        }
    }

    inline bool hasOvl(int i) const {
        for (int j = 0; j < n; j++)
            if (i != j && overlap(pl[i], pl[j])) return true;
        return false;
    }

    inline bool anyOvl() const {
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (overlap(pl[i], pl[j])) return true;
        return false;
    }

    inline double side() const {
        return max(gx1 - gx0, gy1 - gy0);
    }
    inline double score() const {
        double s = side();
        return s * s / n;
    }

    void getBoundary(vector<int>& b) const {
        b.clear();
        double eps = 0.01;
        for (int i = 0; i < n; i++) {
            if (pl[i].x0 - gx0 < eps || gx1 - pl[i].x1 < eps ||
                pl[i].y0 - gy0 < eps || gy1 - pl[i].y1 < eps)
                b.push_back(i);
        }
    }

    // Remove tree at index, shift others down
    Cfg removeTree(int removeIdx) const {
        Cfg c;
        c.n = n - 1;
        int j = 0;
        for (int i = 0; i < n; i++) {
            if (i != removeIdx) {
                c.x[j] = x[i];
                c.y[j] = y[i];
                c.a[j] = a[i];
                j++;
            }
        }
        c.updAll();
        return c;
    }
};

// ========== Core local transforms ==========

Cfg squeeze(Cfg c) {
    double cx = (c.gx0 + c.gx1) / 2.0;
    double cy = (c.gy0 + c.gy1) / 2.0;
    for (double scale = 0.9995; scale >= 0.98; scale -= 0.0005) {
        Cfg trial = c;
        for (int i = 0; i < c.n; i++) {
            trial.x[i] = cx + (c.x[i] - cx) * scale;
            trial.y[i] = cy + (c.y[i] - cy) * scale;
        }
        trial.updAll();
        if (!trial.anyOvl()) c = trial;
        else break;
    }
    return c;
}

Cfg compaction(Cfg c, int iters) {
    double bs = c.side();
    for (int it = 0; it < iters; it++) {
        double cx = (c.gx0 + c.gx1) / 2.0;
        double cy = (c.gy0 + c.gy1) / 2.0;
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            double ox = c.x[i], oy = c.y[i];
            double dx = cx - c.x[i];
            double dy = cy - c.y[i];
            double d = sqrt(dx*dx + dy*dy);
            if (d < 1e-6) continue;
            const double steps[] = {0.02, 0.008, 0.003, 0.001, 0.0004};
            for (double step : steps) {
                c.x[i] = ox + dx/d * step;
                c.y[i] = oy + dy/d * step;
                c.upd(i);
                if (!c.hasOvl(i)) {
                    c.updGlobal();
                    if (c.side() < bs - 1e-12) {
                        bs = c.side();
                        improved = true;
                        ox = c.x[i];
                        oy = c.y[i];
                    } else {
                        c.x[i] = ox;
                        c.y[i] = oy;
                        c.upd(i);
                    }
                } else {
                    c.x[i] = ox;
                    c.y[i] = oy;
                    c.upd(i);
                }
            }
        }
        c.updGlobal();
        if (!improved) break;
    }
    return c;
}

Cfg localSearch(Cfg c, int maxIter) {
    double bs = c.side();
    const double steps[] = {0.01, 0.004, 0.002, 0.001, 0.0005, 0.00025, 0.0001};
    const double rots[]  = {4.0, 2.0, 1.0, 0.5, 0.25, 0.125};
    const int dxs[] = {1,-1,0,0,1,1,-1,-1};
    const int dys[] = {0,0,1,-1,1,-1,1,-1};

    for (int iter = 0; iter < maxIter; iter++) {
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            double cx = (c.gx0 + c.gx1) / 2.0;
            double cy = (c.gy0 + c.gy1) / 2.0;
            double ddx = cx - c.x[i];
            double ddy = cy - c.y[i];
            double dist = sqrt(ddx*ddx + ddy*ddy);
            if (dist > 1e-6) {
                for (double st : steps) {
                    double ox = c.x[i], oy = c.y[i];
                    c.x[i] += ddx/dist * st;
                    c.y[i] += ddy/dist * st;
                    c.upd(i);
                    if (!c.hasOvl(i)) {
                        c.updGlobal();
                        if (c.side() < bs - 1e-12) {
                            bs = c.side();
                            improved = true;
                        } else {
                            c.x[i] = ox;
                            c.y[i] = oy;
                            c.upd(i);
                            c.updGlobal();
                        }
                    } else {
                        c.x[i] = ox;
                        c.y[i] = oy;
                        c.upd(i);
                    }
                }
            }
            for (double st : steps) {
                for (int d = 0; d < 8; d++) {
                    double ox = c.x[i], oy = c.y[i];
                    c.x[i] += dxs[d]*st;
                    c.y[i] += dys[d]*st;
                    c.upd(i);
                    if (!c.hasOvl(i)) {
                        c.updGlobal();
                        if (c.side() < bs - 1e-12) {
                            bs = c.side();
                            improved = true;
                        } else {
                            c.x[i] = ox;
                            c.y[i] = oy;
                            c.upd(i);
                            c.updGlobal();
                        }
                    } else {
                        c.x[i] = ox;
                        c.y[i] = oy;
                        c.upd(i);
                    }
                }
            }
            for (double rt : rots) {
                for (double da : {rt, -rt}) {
                    double oa = c.a[i];
                    c.a[i] += da;
                    while (c.a[i] < 0)   c.a[i] += 360;
                    while (c.a[i] >= 360) c.a[i] -= 360;
                    c.upd(i);
                    if (!c.hasOvl(i)) {
                        c.updGlobal();
                        if (c.side() < bs - 1e-12) {
                            bs = c.side();
                            improved = true;
                        } else {
                            c.a[i] = oa;
                            c.upd(i);
                            c.updGlobal();
                        }
                    } else {
                        c.a[i] = oa;
                        c.upd(i);
                    }
                }
            }
        }
        if (!improved) break;
    }
    return c;
}

// ========= Edge-based slide compaction (outline-aware) =========
//
// 각 트리를 여러 방향으로 "충돌 직전까지" 이분탐색으로 슬라이드
// → 외곽선 기준으로 벽/이웃에 딱 붙는 효과
Cfg edgeSlideCompaction(Cfg c, int outerIter) {
    double bestSide = c.side();

    for (int it = 0; it < outerIter; ++it) {
        bool improved = false;

        for (int i = 0; i < c.n; ++i) {
            double gcx = (c.gx0 + c.gx1) * 0.5;
            double gcy = (c.gy0 + c.gy1) * 0.5;

            double dirs[5][2] = {
                {gcx - c.x[i], gcy - c.y[i]}, // bbox 중심 방향
                { 1.0,  0.0},
                {-1.0,  0.0},
                { 0.0,  1.0},
                { 0.0, -1.0},
            };

            for (int d = 0; d < 5; ++d) {
                double dx = dirs[d][0];
                double dy = dirs[d][1];
                double len = sqrt(dx*dx + dy*dy);
                if (len < 1e-9) continue;
                dx /= len;
                dy /= len;

                double maxStep = 0.30;
                double lo = 0.0, hi = maxStep;
                double bestStep = 0.0;

                double ox = c.x[i];
                double oy = c.y[i];

                for (int it2 = 0; it2 < 20; ++it2) {
                    double mid = 0.5 * (lo + hi);

                    c.x[i] = ox + dx * mid;
                    c.y[i] = oy + dy * mid;
                    c.upd(i);
                    c.updGlobal();

                    bool okOverlap = !c.hasOvl(i);
                    bool okSide    = (c.side() <= bestSide + 1e-9);

                    if (okOverlap && okSide) {
                        bestStep = mid;
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }

                if (bestStep > 1e-6) {
                    c.x[i] = ox + dx * bestStep;
                    c.y[i] = oy + dy * bestStep;
                    c.upd(i);
                    c.updGlobal();

                    double ns = c.side();
                    if (ns < bestSide - 1e-12) {
                        bestSide = ns;
                        improved = true;
                    }
                } else {
                    c.x[i] = ox;
                    c.y[i] = oy;
                    c.upd(i);
                    c.updGlobal();
                }
            }
        }

        if (!improved) break;
    }

    return c;
}

// ========== SA + perturb + parallel optimize ==========

Cfg sa_opt(Cfg c, int iter, double T0, double Tm, uint64_t seed) {
    FastRNG rng(seed);
    Cfg best = c, cur = c;
    double bs = best.side(), cs = bs, T = T0;
    double alpha = pow(Tm / T0, 1.0 / iter);
    int noImp = 0;

    for (int it = 0; it < iter; it++) {
        int mt = rng.ri(10);
        double sc = T / T0;
        bool valid = true;

        if (mt == 0) {
            int i = rng.ri(c.n);
            double ox = cur.x[i], oy = cur.y[i];
            cur.x[i] += rng.gaussian() * 0.5 * sc;
            cur.y[i] += rng.gaussian() * 0.5 * sc;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }
        else if (mt == 1) {
            int i = rng.ri(c.n);
            double ox = cur.x[i], oy = cur.y[i];
            double bcx = (cur.gx0+cur.gx1)/2.0;
            double bcy = (cur.gy0+cur.gy1)/2.0;
            double dx = bcx - cur.x[i];
            double dy = bcy - cur.y[i];
            double d  = sqrt(dx*dx + dy*dy);
            if (d > 1e-6) {
                cur.x[i] += dx/d * rng.rf() * 0.6 * sc;
                cur.y[i] += dy/d * rng.rf() * 0.6 * sc;
            }
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }
        else if (mt == 2) {
            int i = rng.ri(c.n);
            double oa = cur.a[i];
            cur.a[i] += rng.gaussian() * 80 * sc;
            while (cur.a[i] < 0)   cur.a[i] += 360;
            while (cur.a[i] >= 360) cur.a[i] -= 360;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.a[i]=oa; cur.upd(i); valid=false; }
        }
        else if (mt == 3) {
            int i = rng.ri(c.n);
            double ox=cur.x[i], oy=cur.y[i], oa=cur.a[i];
            cur.x[i] += rng.rf2() * 0.5 * sc;
            cur.y[i] += rng.rf2() * 0.5 * sc;
            cur.a[i] += rng.rf2() * 60 * sc;
            while (cur.a[i] < 0)   cur.a[i] += 360;
            while (cur.a[i] >= 360) cur.a[i] -= 360;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.a[i]=oa; cur.upd(i); valid=false; }
        }
        else if (mt == 4) {
            vector<int> boundary;
            cur.getBoundary(boundary);
            if (!boundary.empty()) {
                int i = boundary[rng.ri((int)boundary.size())];
                double ox=cur.x[i], oy=cur.y[i], oa=cur.a[i];
                double bcx = (cur.gx0+cur.gx1)/2.0;
                double bcy = (cur.gy0+cur.gy1)/2.0;
                double dx = bcx - cur.x[i];
                double dy = bcy - cur.y[i];
                double d  = sqrt(dx*dx + dy*dy);
                if (d > 1e-6) {
                    cur.x[i] += dx/d * rng.rf() * 0.7 * sc;
                    cur.y[i] += dy/d * rng.rf() * 0.7 * sc;
                }
                cur.a[i] += rng.rf2() * 50 * sc;
                while (cur.a[i] < 0)   cur.a[i] += 360;
                while (cur.a[i] >= 360) cur.a[i] -= 360;
                cur.upd(i);
                if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.a[i]=oa; cur.upd(i); valid=false; }
            } else valid = false;
        }
        else if (mt == 5) {
            double factor = 1.0 - rng.rf() * 0.004 * sc;
            double cx = (cur.gx0 + cur.gx1) / 2.0;
            double cy = (cur.gy0 + cur.gy1) / 2.0;
            Cfg trial = cur;
            for (int i = 0; i < c.n; i++) {
                trial.x[i] = cx + (cur.x[i] - cx) * factor;
                trial.y[i] = cy + (cur.y[i] - cy) * factor;
            }
            trial.updAll();
            if (!trial.anyOvl()) cur = trial;
            else valid = false;
        }
        else if (mt == 6) {
            int i = rng.ri(c.n);
            double ox=cur.x[i], oy=cur.y[i];
            double levy = pow(rng.rf() + 0.001, -1.3) * 0.008;
            cur.x[i] += rng.rf2() * levy;
            cur.y[i] += rng.rf2() * levy;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }
        else if (mt == 7 && c.n > 1) {
            int i = rng.ri(c.n), j = (i + 1) % c.n;
            double oxi=cur.x[i], oyi=cur.y[i];
            double oxj=cur.x[j], oyj=cur.y[j];
            double dx = rng.rf2() * 0.3 * sc;
            double dy = rng.rf2() * 0.3 * sc;
            cur.x[i]+=dx; cur.y[i]+=dy;
            cur.x[j]+=dx; cur.y[j]+=dy;
            cur.upd(i); cur.upd(j);
            if (cur.hasOvl(i) || cur.hasOvl(j)) {
                cur.x[i]=oxi; cur.y[i]=oyi;
                cur.x[j]=oxj; cur.y[j]=oyj;
                cur.upd(i); cur.upd(j);
                valid=false;
            }
        }
        else {
            int i = rng.ri(c.n);
            double ox=cur.x[i], oy=cur.y[i];
            cur.x[i] += rng.rf2() * 0.002;
            cur.y[i] += rng.rf2() * 0.002;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }

        if (!valid) {
            noImp++;
            T *= alpha;
            if (T < Tm) T = Tm;
            continue;
        }

        cur.updGlobal();
        double ns = cur.side();
        double delta = ns - cs;

        if (delta < 0 || rng.rf() < exp(-delta / T)) {
            cs = ns;
            if (ns < bs) {
                bs = ns;
                best = cur;
                noImp = 0;
            } else noImp++;
        } else {
            cur = best;
            cs  = bs;
            noImp++;
        }

        if (noImp > 200) {
            T = min(T * 5.0, T0);
            noImp = 0;
        }
        T *= alpha;
        if (T < Tm) T = Tm;
    }
    return best;
}

Cfg perturb(Cfg c, double str, FastRNG& rng) {
    Cfg original = c;
    int np = max(1, (int)(c.n * 0.08 + str * 3));
    for (int k = 0; k < np; k++) {
        int i = rng.ri(c.n);
        c.x[i] += rng.gaussian() * str * 0.5;
        c.y[i] += rng.gaussian() * str * 0.5;
        c.a[i] += rng.gaussian() * 30.0;
        while (c.a[i] < 0)   c.a[i] += 360;
        while (c.a[i] >= 360) c.a[i] -= 360;
    }
    c.updAll();
    for (int iter = 0; iter < 150; iter++) {
        bool fixed = true;
        for (int i = 0; i < c.n; i++) {
            if (c.hasOvl(i)) {
                fixed = false;
                double cx = (c.gx0+c.gx1)/2.0;
                double cy = (c.gy0+c.gy1)/2.0;
                double dx = c.x[i] - cx;
                double dy = c.y[i] - cy;
                double d  = sqrt(dx*dx + dy*dy);
                if (d > 1e-6) {
                    c.x[i] += dx/d*0.02;
                    c.y[i] += dy/d*0.02;
                }
                c.a[i] += rng.rf2() * 15.0;
                while (c.a[i] < 0)   c.a[i] += 360;
                while (c.a[i] >= 360) c.a[i] -= 360;
                c.upd(i);
            }
        }
        if (fixed) break;
    }
    c.updGlobal();
    if (c.anyOvl()) return original;
    return c;
}

Cfg optimizeParallel(Cfg c, int iters, int restarts) {
    Cfg globalBest = c;
    double globalBestSide = c.side();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        FastRNG rng(42 + tid * 1000 + c.n);
        Cfg localBest = c;
        double localBestSide = c.side();

        #pragma omp for schedule(dynamic)
        for (int r = 0; r < restarts; r++) {
            Cfg start;
            if (r == 0) {
                start = c;
            } else {
                start = perturb(c, 0.02 + 0.02 * (r % 8), rng);
                if (start.anyOvl()) continue;
            }

            uint64_t seed = 42 + r * 1000 + tid * 100000 + c.n;
            Cfg o = sa_opt(start, iters, 2.5, 0.0000005, seed);
            o = squeeze(o);
            o = compaction(o, 50);
            o = edgeSlideCompaction(o, 10);
            o = localSearch(o, 80);

            if (!o.anyOvl() && o.side() < localBestSide) {
                localBestSide = o.side();
                localBest = o;
            }
        }

        #pragma omp critical
        {
            if (!localBest.anyOvl() && localBestSide < globalBestSide) {
                globalBestSide = localBestSide;
                globalBest = localBest;
            }
        }
    }

    globalBest = squeeze(globalBest);
    globalBest = compaction(globalBest, 80);
    globalBest = edgeSlideCompaction(globalBest, 12);
    globalBest = localSearch(globalBest, 150);

    if (globalBest.anyOvl()) return c;
    return globalBest;
}

// ========== Free-area & protrusion removal & reinsertion heuristic ==========

struct TreeState {
    double x, y, a;
};

void computeFreeArea(const Cfg& c, vector<double>& freeArea) {
    int n = c.n;
    freeArea.assign(n, 0.0);
    vector<double> area(n), overlapSum(n, 0.0);

    for (int i = 0; i < n; ++i) {
        double w = max(0.0, c.pl[i].x1 - c.pl[i].x0);
        double h = max(0.0, c.pl[i].y1 - c.pl[i].y0);
        area[i] = w * h;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            double ix0 = max(c.pl[i].x0, c.pl[j].x0);
            double iy0 = max(c.pl[i].y0, c.pl[j].y0);
            double ix1 = min(c.pl[i].x1, c.pl[j].x1);
            double iy1 = min(c.pl[i].y1, c.pl[j].y1);
            double dx = ix1 - ix0;
            double dy = iy1 - iy0;
            if (dx > 0.0 && dy > 0.0) {
                overlapSum[i] += dx * dy;
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        double occ = overlapSum[i];
        if (occ > area[i]) occ = area[i];
        freeArea[i] = max(0.0, area[i] - occ);  // 0 = 매우 답답, area[i] = 완전 여유
    }
}

// "튀어나와있는" 정도: 전체 bbox의 경계에 거의 붙어 있으면서
// 중심에서 멀리 떨어진 트리에 높은 점수 부여
void computeProtrudeScore(const Cfg& c, vector<double>& protrudeScore) {
    int n = c.n;
    protrudeScore.assign(n, 0.0);
    double cx = (c.gx0 + c.gx1) * 0.5;
    double cy = (c.gy0 + c.gy1) * 0.5;
    double side = c.side();
    double eps = side * 0.02;  // 2% 이내면 경계에 있다고 봄

    for (int i = 0; i < n; ++i) {
        bool onBoundary =
            (c.pl[i].x0 - c.gx0 < eps) ||
            (c.gx1 - c.pl[i].x1 < eps) ||
            (c.pl[i].y0 - c.gy0 < eps) ||
            (c.gy1 - c.pl[i].y1 < eps);

        if (!onBoundary) {
            protrudeScore[i] = 0.0;
            continue;
        }

        double tx = 0.5 * (c.pl[i].x0 + c.pl[i].x1);
        double ty = 0.5 * (c.pl[i].y0 + c.pl[i].y1);
        double d  = sqrt((tx - cx)*(tx - cx) + (ty - cy)*(ty - cy));

        // 거리 자체를 score로 사용 (멀수록 더 튀어나왔다고 가정)
        protrudeScore[i] = d;
    }
}

Cfg reinsertTrees(const Cfg& base, const vector<TreeState>& removed, uint64_t seed) {
    Cfg cur = base;
    FastRNG rng(seed);

    for (const auto& t : removed) {
        if (cur.n >= MAX_N) return base; // 안전장치

        int idx = cur.n;
        cur.n++;
        cur.x[idx] = t.x;
        cur.y[idx] = t.y;
        cur.a[idx] = t.a;
        cur.upd(idx);
        cur.updGlobal();

        bool placed = false;
        for (int attempt = 0; attempt < 200; ++attempt) {
            if (!cur.hasOvl(idx)) { placed = true; break; }

            double cx = (cur.gx0 + cur.gx1) * 0.5;
            double cy = (cur.gy0 + cur.gy1) * 0.5;
            double radius = 0.1 + 0.6 * rng.rf();
            double ang    = 2.0 * PI * rng.rf();

            cur.x[idx] = cx + radius * cos(ang);
            cur.y[idx] = cy + radius * sin(ang);
            cur.a[idx] = fmod(t.a + rng.rf2() * 120.0 + 360.0, 360.0);

            cur.upd(idx);
            cur.updGlobal();
        }

        if (!placed) {
            cur.n--;
            return base;
        }
    }

    if (cur.anyOvl()) return base;
    return cur;
}

Cfg freeAreaHeuristic(const Cfg& c, double removeRatio, uint64_t seed) {
    Cfg best = c;
    int n = c.n;
    if (n <= 5) return best;   // 너무 작은 n은 스킵

    int k = (int)floor(n * removeRatio + 1e-9);
    if (k < 1) k = 1;
    if (k >= n) k = n - 1;

    vector<double> freeArea;
    vector<double> protrudeScore;
    computeFreeArea(c, freeArea);
    computeProtrudeScore(c, protrudeScore);

    // freeArea 큰 순 (여유 많은 애들)
    vector<pair<double,int>> freeList;
    freeList.reserve(n);
    for (int i = 0; i < n; ++i)
        freeList.emplace_back(freeArea[i], i);
    sort(freeList.begin(), freeList.end(),
         [](const pair<double,int>& a, const pair<double,int>& b){
             if (a.first != b.first) return a.first > b.first;
             return a.second < b.second;
         });

    // protrudeScore 큰 순 (튀어나온 애들, 경계+원점에서 먼)
    vector<pair<double,int>> protList;
    protList.reserve(n);
    for (int i = 0; i < n; ++i)
        if (protrudeScore[i] > 0.0)
            protList.emplace_back(protrudeScore[i], i);
    sort(protList.begin(), protList.end(),
         [](const pair<double,int>& a, const pair<double,int>& b){
             if (a.first != b.first) return a.first > b.first;
             return a.second < b.second;
         });

    // 제거할 개수: 대략 2/3는 튀어나온 애들, 1/3는 여유 많은 애들
    int kProt = min((int)protList.size(), (int)(k * 2 / 3));
    int kFree = k - kProt;
    if (kFree < 0) kFree = 0;

    vector<bool> removeFlag(n, false);
    vector<TreeState> removed;
    removed.reserve(k);

    // 1) 튀어나온 애들부터 제거
    int removedCnt = 0;
    for (int i = 0; i < (int)protList.size() && removedCnt < kProt; ++i) {
        int idx = protList[i].second;
        if (removeFlag[idx]) continue;
        removeFlag[idx] = true;
        removed.push_back(TreeState{c.x[idx], c.y[idx], c.a[idx]});
        removedCnt++;
    }

    // 2) 남은 수만큼 freeArea 큰 애들 제거
    for (int i = 0; i < (int)freeList.size() && removedCnt < k; ++i) {
        int idx = freeList[i].second;
        if (removeFlag[idx]) continue;
        removeFlag[idx] = true;
        removed.push_back(TreeState{c.x[idx], c.y[idx], c.a[idx]});
        removedCnt++;
    }

    if (removed.empty()) return best;

    Cfg reduced;
    reduced.n = n - (int)removed.size();
    int ptr = 0;
    for (int i = 0; i < n; ++i) {
        if (!removeFlag[i]) {
            reduced.x[ptr] = c.x[i];
            reduced.y[ptr] = c.y[i];
            reduced.a[ptr] = c.a[i];
            ptr++;
        }
    }
    reduced.updAll();
    if (reduced.anyOvl()) return best;

    // 남은 subset 다시 최적화 (살짝 강하게)
    Cfg reducedOpt = optimizeParallel(reduced, max(2000, 8000), 8);

    // 제거한 트리 재삽입
    Cfg withInserted = reinsertTrees(reducedOpt, removed, seed);
    if (withInserted.n != n || withInserted.anyOvl()) return best;

    // 한 번 더 조이기 (+ edge slide)
    withInserted = squeeze(withInserted);
    withInserted = compaction(withInserted, 40);
    withInserted = edgeSlideCompaction(withInserted, 10);
    withInserted = localSearch(withInserted, 80);

    if (!withInserted.anyOvl() && withInserted.side() < best.side() - 1e-12) {
        return withInserted;
    }
    return best;
}

// ========== IO & main ==========

map<int, Cfg> loadCSV(const string& fn) {
    map<int, Cfg> cfg;
    ifstream f(fn);
    if (!f) return cfg;
    string ln; getline(f, ln);
    map<int, vector<tuple<int,double,double,double>>> data;
    while (getline(f, ln)) {
        size_t p1=ln.find(','), p2=ln.find(',',p1+1), p3=ln.find(',',p2+1);
        string id = ln.substr(0,p1);
        string xs = ln.substr(p1+1,p2-p1-1);
        string ys = ln.substr(p2+1,p3-p2-1);
        string ds = ln.substr(p3+1);
        if(!xs.empty() && xs[0]=='s') xs=xs.substr(1);
        if(!ys.empty() && ys[0]=='s') ys=ys.substr(1);
        if(!ds.empty() && ds[0]=='s') ds=ds.substr(1);
        int n   = stoi(id.substr(0,3));
        int idx = stoi(id.substr(4));
        data[n].push_back({idx, stod(xs), stod(ys), stod(ds)});
    }
    for (auto& kv : data) {
        int n = kv.first;
        auto& v = kv.second;
        Cfg c;
        c.n = n;
        for (auto& tup : v) {
            int i; double x, y, d;
            tie(i,x,y,d) = tup;
            if (i < n) {
                c.x[i] = x;
                c.y[i] = y;
                c.a[i] = d;
            }
        }
        c.updAll();
        cfg[n] = c;
    }
    return cfg;
}

void saveCSV(const string& fn, const map<int, Cfg>& cfg) {
    ofstream f(fn);
    f << fixed << setprecision(15);
    f << "id,x,y,deg\n";
    for (int n = 1; n <= 200; n++) {
        if (cfg.count(n)) {
            const Cfg& c = cfg.at(n);
            for (int i = 0; i < n; i++) {
                f << setfill('0') << setw(3) << n
                  << "_" << i << ",s" << c.x[i]
                  << ",s" << c.y[i]
                  << ",s" << c.a[i] << "\n";
            }
        }
    }
}

int main(int argc, char** argv) {
    string in="submission.csv", out="submission_v18.csv";
    int iters=15000, restarts=16;

    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a=="-i" && i+1<argc) in=argv[++i];
        else if (a=="-o" && i+1<argc) out=argv[++i];
        else if (a=="-n" && i+1<argc) iters=stoi(argv[++i]);
        else if (a=="-r" && i+1<argc) restarts=stoi(argv[++i]);
    }

    int numThreads = omp_get_max_threads();
    printf("Tree Packer v18 (PARALLEL + BACK PROP + free-area & protrusion removal, %d threads)\n", numThreads);
    printf("Iterations: %d, Restarts: %d\n", iters, restarts);
    printf("Loading %s...\n", in.c_str());

    auto cfg = loadCSV(in);
    if (cfg.empty()) {
        printf("No data!\n");
        return 1;
    }
    printf("Loaded %d configs\n", (int)cfg.size());

    double init = 0.0;
    for (auto& kv : cfg) init += kv.second.score();
    printf("Initial: %.6f\n\nPhase 1: Parallel optimization...\n\n", init);

    auto t0 = chrono::high_resolution_clock::now();
    map<int, Cfg> res;
    int totalImproved = 0;

    // Phase 1: main optimization + per-n free-area & protrusion heuristic
    for (int n = 200; n >= 1; n--) {
        if (!cfg.count(n)) continue;
        Cfg c = cfg[n];
        double os = c.score();

        int it = iters, r = restarts;
        if (n <= 10)      { it = (int)(iters * 2.5); r = restarts * 2; }
        else if (n <= 30) { it = (int)(iters * 1.8); r = (int)(restarts * 1.5); }
        else if (n <= 60) { it = (int)(iters * 1.3); r = restarts; }
        else if (n > 150) { it = (int)(iters * 0.7); r = (int)(restarts * 0.8); }

        Cfg o = optimizeParallel(c, it, max(4, r));

        // Simple backward propagation from n+1, n+2
        for (auto& kv : res) {
            int m   = kv.first;
            Cfg& pc = kv.second;
            if (m > n && m <= n + 2) {
                Cfg ad;
                ad.n = n;
                for (int i = 0; i < n; i++) {
                    ad.x[i] = pc.x[i];
                    ad.y[i] = pc.y[i];
                    ad.a[i] = pc.a[i];
                }
                ad.updAll();
                if (!ad.anyOvl()) {
                    ad = compaction(ad, 40);
                    ad = edgeSlideCompaction(ad, 8);
                    ad = localSearch(ad, 60);
                    if (!ad.anyOvl() && ad.side() < o.side()) o = ad;
                }
            }
        }

        if (o.anyOvl() || o.side() > c.side() + 1e-14) {
            o = c;
        }

        // free-area & protrusion heuristic (remove 70% 정도, 재배치)
        if (!o.anyOvl() && n >= 10) {
            Cfg oh = freeAreaHeuristic(o, 0.70, 1234567ULL + (uint64_t)n * 101ULL);
            if (!oh.anyOvl() && oh.side() < o.side() - 1e-12) {
                o = oh;
            }
        }

        res[n] = o;
        double ns = o.score();
        if (ns < os - 1e-10) {
            printf("n=%3d: %.6f -> %.6f (%.4f%%)\n", n, os, ns, (os-ns)/os*100.0);
            fflush(stdout);
            totalImproved++;
        }
    }

    // Phase 2: aggressive back propagation (removing trees)
    printf("\nPhase 2: Aggressive back propagation (removing trees)...\n\n");

    int backPropImproved = 0;
    bool changed = true;
    int passNum = 0;

    while (changed && passNum < 10) {
        changed = false;
        passNum++;

        // k vs (k-1)
        for (int k = 200; k >= 2; k--) {
            if (!res.count(k) || !res.count(k-1)) continue;

            double sideK  = res[k].side();
            double sideK1 = res[k-1].side();

            if (sideK < sideK1 - 1e-12) {
                Cfg& cfgK = res[k];
                double bestSide = sideK1;
                Cfg bestCfg     = res[k-1];

                #pragma omp parallel
                {
                    double localBestSide = bestSide;
                    Cfg localBestCfg     = bestCfg;

                    #pragma omp for schedule(dynamic)
                    for (int removeIdx = 0; removeIdx < k; removeIdx++) {
                        Cfg reduced = cfgK.removeTree(removeIdx);

                        if (!reduced.anyOvl()) {
                            reduced = squeeze(reduced);
                            reduced = compaction(reduced, 60);
                            reduced = edgeSlideCompaction(reduced, 10);
                            reduced = localSearch(reduced, 100);

                            if (!reduced.anyOvl() && reduced.side() < localBestSide) {
                                localBestSide = reduced.side();
                                localBestCfg  = reduced;
                            }
                        }
                    }

                    #pragma omp critical
                    {
                        if (localBestSide < bestSide) {
                            bestSide = localBestSide;
                            bestCfg  = localBestCfg;
                        }
                    }
                }

                if (bestSide < sideK1 - 1e-12) {
                    double oldScore = res[k-1].score();
                    double newScore = bestCfg.score();
                    res[k-1] = bestCfg;
                    printf("n=%3d: %.6f -> %.6f (from n=%d removal, %.4f%%)\n",
                           k-1, oldScore, newScore, k, (oldScore-newScore)/oldScore*100.0);
                    fflush(stdout);
                    backPropImproved++;
                    changed = true;
                }
            }
        }

        // k vs src>k (k+1..k+5)
        for (int k = 200; k >= 3; k--) {
            for (int src = k + 1; src <= min(200, k + 5); src++) {
                if (!res.count(src) || !res.count(k)) continue;

                double sideSrc = res[src].side();
                double sideK   = res[k].side();

                if (sideSrc < sideK - 1e-12) {
                    int toRemove = src - k;
                    Cfg cfgSrc   = res[src];

                    vector<vector<int>> combos;
                    if (toRemove == 1) {
                        for (int i = 0; i < src; i++) combos.push_back({i});
                    } else if (toRemove == 2 && src <= 50) {
                        for (int i = 0; i < src; i++)
                            for (int j = i+1; j < src; j++)
                                combos.push_back({i,j});
                    } else {
                        FastRNG rng((uint64_t)k * 1000ULL + (uint64_t)src);
                        for (int t = 0; t < min(200, src * 3); t++) {
                            vector<int> combo;
                            unordered_set<int> used;
                            for (int r = 0; r < toRemove; r++) {
                                int idx;
                                do { idx = rng.ri(src); } while (used.count(idx));
                                used.insert(idx);
                                combo.push_back(idx);
                            }
                            sort(combo.begin(), combo.end());
                            combos.push_back(combo);
                        }
                    }

                    double bestSide = sideK;
                    Cfg bestCfg     = res[k];

                    #pragma omp parallel
                    {
                        double localBestSide = bestSide;
                        Cfg localBestCfg     = bestCfg;

                        #pragma omp for schedule(dynamic)
                        for (int ci = 0; ci < (int)combos.size(); ci++) {
                            Cfg reduced = cfgSrc;
                            vector<int> toRem = combos[ci];
                            sort(toRem.rbegin(), toRem.rend());
                            for (int idx : toRem) {
                                reduced = reduced.removeTree(idx);
                            }

                            if (!reduced.anyOvl()) {
                                reduced = squeeze(reduced);
                                reduced = compaction(reduced, 50);
                                reduced = edgeSlideCompaction(reduced, 10);
                                reduced = localSearch(reduced, 80);

                                if (!reduced.anyOvl() && reduced.side() < localBestSide) {
                                    localBestSide = reduced.side();
                                    localBestCfg  = reduced;
                                }
                            }
                        }

                        #pragma omp critical
                        {
                            if (localBestSide < bestSide) {
                                bestSide = localBestSide;
                                bestCfg  = localBestCfg;
                            }
                        }
                    }

                    if (bestSide < sideK - 1e-12) {
                        double oldScore = res[k].score();
                        double newScore = bestCfg.score();
                        res[k] = bestCfg;
                        printf("n=%3d: %.6f -> %.6f (from n=%d removal, %.4f%%)\n",
                               k, oldScore, newScore, src, (oldScore-newScore)/oldScore*100.0);
                        fflush(stdout);
                        backPropImproved++;
                        changed = true;
                    }
                }
            }
        }

        if (changed) {
            printf("Pass %d complete, continuing...\n", passNum);
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    double el = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count() / 1000.0;

    double fin = 0.0;
    for (auto& kv : res) fin += kv.second.score();

    printf("\n========================================\n");
    printf("Initial: %.6f\nFinal:   %.6f\n", init, fin);
    printf("Improve: %.6f (%.4f%%)\n", init-fin, (init-fin)/init*100.0);
    printf("Phase 1 improved: %d configs\n", totalImproved);
    printf("Phase 2 back-prop improved: %d configs\n", backPropImproved);
    printf("Time:    %.1fs (with %d threads)\n", el, numThreads);
    printf("========================================\n");

    saveCSV(out, res);
    printf("Saved %s\n", out.c_str());
    return 0;
}
