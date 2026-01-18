// Aggressive Simulated Annealing with Higher Temperature
// Focuses on escaping local optima with more aggressive exploration
// Compile: g++ -O3 -march=native -std=c++17 -fopenmp -o aggressive_sa aggressive_sa.cpp

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

constexpr int MAX_N = 200;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;

const long double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
const long double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

struct FastRNG {
    uint64_t s[2];
    FastRNG(uint64_t seed = 42) {
        s[0] = seed ^ 0x853c49e6748fea9bULL;
        s[1] = (seed * 0x9e3779b97f4a7c15ULL) ^ 0xc4ceb9fe1a85ec53ULL;
    }
    inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        uint64_t s0 = s[0], s1 = s[1], r = s0 + s1;
        s1 ^= s0; s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); s[1] = rotl(s1, 37);
        return r;
    }
    inline long double rf() { return (next() >> 11) * 0x1.0p-53L; }
    inline long double rf2() { return rf() * 2.0L - 1.0L; }
    inline int ri(int n) { return next() % n; }
    inline long double gaussian() {
        long double u1 = rf() + 1e-10L, u2 = rf();
        return sqrtl(-2.0L * logl(u1)) * cosl(2.0L * PI * u2);
    }
};

struct Poly {
    long double px[NV], py[NV];
    long double x0, y0, x1, y1;
};

inline void getPoly(long double cx, long double cy, long double deg, Poly& q) {
    long double rad = deg * (PI / 180.0L);
    long double s = sinl(rad), c = cosl(rad);
    long double minx = 1e9L, miny = 1e9L, maxx = -1e9L, maxy = -1e9L;
    for (int i = 0; i < NV; i++) {
        long double x = TX[i] * c - TY[i] * s + cx;
        long double y = TX[i] * s + TY[i] * c + cy;
        q.px[i] = x; q.py[i] = y;
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
    }
    q.x0 = minx; q.y0 = miny; q.x1 = maxx; q.y1 = maxy;
}

inline bool pip(long double px, long double py, const Poly& q) {
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

inline bool segInt(long double ax, long double ay, long double bx, long double by,
                   long double cx, long double cy, long double dx, long double dy) {
    long double d1 = (dx-cx)*(ay-cy) - (dy-cy)*(ax-cx);
    long double d2 = (dx-cx)*(by-cy) - (dy-cy)*(bx-cx);
    long double d3 = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
    long double d4 = (bx-ax)*(dy-ay) - (by-ay)*(dx-ax);
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
    long double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];
    long double gx0, gy0, gx1, gy1;

    void upd(int i) { getPoly(x[i], y[i], a[i], pl[i]); }
    void updAll() { for (int i = 0; i < n; i++) upd(i); updGlobal(); }

    void updGlobal() {
        gx0 = gy0 = 1e9L; gx1 = gy1 = -1e9L;
        for (int i = 0; i < n; i++) {
            if (pl[i].x0 < gx0) gx0 = pl[i].x0;
            if (pl[i].x1 > gx1) gx1 = pl[i].x1;
            if (pl[i].y0 < gy0) gy0 = pl[i].y0;
            if (pl[i].y1 > gy1) gy1 = pl[i].y1;
        }
    }

    bool hasOvl(int i) const {
        for (int j = 0; j < n; j++)
            if (i != j && overlap(pl[i], pl[j])) return true;
        return false;
    }

    bool anyOvl() const {
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (overlap(pl[i], pl[j])) return true;
        return false;
    }

    long double side() const { return max(gx1 - gx0, gy1 - gy0); }
};

Cfg configs[MAX_N + 1];
long double best_sides[MAX_N + 1];

void parse_csv(const string& fn) {
    ifstream f(fn);
    string line;
    getline(f, line);
    
    map<int, vector<tuple<int, long double, long double, long double>>> data;
    
    while (getline(f, line)) {
        size_t p1 = line.find(',');
        size_t p2 = line.find(',', p1+1);
        size_t p3 = line.find(',', p2+1);
        
        string id = line.substr(0, p1);
        string xs = line.substr(p1+2, p2-p1-2);
        string ys = line.substr(p2+2, p3-p2-2);
        string ds = line.substr(p3+2);
        
        size_t us = id.find('_');
        int n = stoi(id.substr(0, us));
        int idx = stoi(id.substr(us+1));
        
        data[n].push_back({idx, stold(xs), stold(ys), stold(ds)});
    }
    
    for (auto& [n, trees] : data) {
        sort(trees.begin(), trees.end());
        configs[n].n = n;
        for (int i = 0; i < n; i++) {
            configs[n].x[i] = get<1>(trees[i]);
            configs[n].y[i] = get<2>(trees[i]);
            configs[n].a[i] = get<3>(trees[i]);
        }
        configs[n].updAll();
        best_sides[n] = configs[n].side();
    }
}

void save_csv(const string& fn) {
    ofstream f(fn);
    f << fixed << setprecision(15);
    f << "id,x,y,deg\n";
    for (int n = 1; n <= MAX_N; n++) {
        for (int i = 0; i < n; i++) {
            f << setfill('0') << setw(3) << n << "_" << i << ",s" 
              << configs[n].x[i] << ",s" << configs[n].y[i] << ",s" << configs[n].a[i] << "\n";
        }
    }
}

long double calc_total_score() {
    long double total = 0;
    for (int n = 1; n <= MAX_N; n++) {
        long double s = best_sides[n];
        total += s * s / n;
    }
    return total;
}

// Aggressive SA with very high temperature and many restarts
Cfg aggressiveSA(Cfg c, int iters, FastRNG& rng, long double T0 = 0.5L) {
    long double bs = c.side();
    Cfg best = c;
    long double T = T0;
    long double alpha = powl(0.00001L / T0, 1.0L / iters);
    
    for (int it = 0; it < iters; it++) {
        int moveType = rng.ri(5);
        
        if (moveType == 0) { // Large translation
            int i = rng.ri(c.n);
            long double ox = c.x[i], oy = c.y[i];
            long double step = T * 2.0L;
            c.x[i] += rng.gaussian() * step;
            c.y[i] += rng.gaussian() * step;
            c.upd(i);
            if (c.hasOvl(i)) { c.x[i] = ox; c.y[i] = oy; c.upd(i); }
            else {
                c.updGlobal();
                long double ns = c.side();
                if (ns < bs) { bs = ns; best = c; }
                else if (rng.rf() > expl(-(ns - bs) / T)) { c.x[i] = ox; c.y[i] = oy; c.upd(i); c.updGlobal(); }
            }
        } else if (moveType == 1) { // Large rotation
            int i = rng.ri(c.n);
            long double oa = c.a[i];
            c.a[i] += rng.gaussian() * T * 90.0L;
            while (c.a[i] < 0) c.a[i] += 360; while (c.a[i] >= 360) c.a[i] -= 360;
            c.upd(i);
            if (c.hasOvl(i)) { c.a[i] = oa; c.upd(i); }
            else {
                c.updGlobal();
                long double ns = c.side();
                if (ns < bs) { bs = ns; best = c; }
                else if (rng.rf() > expl(-(ns - bs) / T)) { c.a[i] = oa; c.upd(i); c.updGlobal(); }
            }
        } else if (moveType == 2 && c.n >= 2) { // Swap positions
            int i = rng.ri(c.n), j = rng.ri(c.n);
            if (i != j) {
                swap(c.x[i], c.x[j]); swap(c.y[i], c.y[j]);
                c.upd(i); c.upd(j);
                if (c.hasOvl(i) || c.hasOvl(j)) {
                    swap(c.x[i], c.x[j]); swap(c.y[i], c.y[j]);
                    c.upd(i); c.upd(j);
                } else {
                    c.updGlobal();
                    long double ns = c.side();
                    if (ns < bs) { bs = ns; best = c; }
                    else if (rng.rf() > expl(-(ns - bs) / T)) {
                        swap(c.x[i], c.x[j]); swap(c.y[i], c.y[j]);
                        c.upd(i); c.upd(j); c.updGlobal();
                    }
                }
            }
        } else if (moveType == 3 && c.n >= 2) { // Swap angles
            int i = rng.ri(c.n), j = rng.ri(c.n);
            if (i != j) {
                swap(c.a[i], c.a[j]);
                c.upd(i); c.upd(j);
                if (c.hasOvl(i) || c.hasOvl(j)) {
                    swap(c.a[i], c.a[j]);
                    c.upd(i); c.upd(j);
                } else {
                    c.updGlobal();
                    long double ns = c.side();
                    if (ns < bs) { bs = ns; best = c; }
                    else if (rng.rf() > expl(-(ns - bs) / T)) {
                        swap(c.a[i], c.a[j]);
                        c.upd(i); c.upd(j); c.updGlobal();
                    }
                }
            }
        } else { // Move towards center
            int i = rng.ri(c.n);
            long double cx = (c.gx0 + c.gx1) / 2.0L, cy = (c.gy0 + c.gy1) / 2.0L;
            long double ox = c.x[i], oy = c.y[i];
            long double dx = cx - c.x[i], dy = cy - c.y[i];
            long double d = sqrtl(dx*dx + dy*dy);
            if (d > 1e-6L) {
                long double step = T * 1.0L;
                c.x[i] += dx/d * step; c.y[i] += dy/d * step;
                c.upd(i);
                if (c.hasOvl(i)) { c.x[i] = ox; c.y[i] = oy; c.upd(i); }
                else {
                    c.updGlobal();
                    long double ns = c.side();
                    if (ns < bs) { bs = ns; best = c; }
                    else if (rng.rf() > expl(-(ns - bs) / T)) { c.x[i] = ox; c.y[i] = oy; c.upd(i); c.updGlobal(); }
                }
            }
        }
        T *= alpha;
    }
    return best;
}

// Local search with very fine steps
Cfg fineLocalSearch(Cfg c, int maxIter) {
    long double bs = c.side();
    const long double steps[] = {0.001L, 0.0005L, 0.0002L, 0.0001L, 0.00005L, 0.00002L, 0.00001L};
    const long double rots[] = {1.0L, 0.5L, 0.2L, 0.1L, 0.05L, 0.02L, 0.01L};
    const int dx[] = {1,-1,0,0,1,1,-1,-1};
    const int dy[] = {0,0,1,-1,1,-1,1,-1};

    for (int iter = 0; iter < maxIter; iter++) {
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            // Move towards center
            long double cx = (c.gx0 + c.gx1) / 2.0L, cy = (c.gy0 + c.gy1) / 2.0L;
            long double ddx = cx - c.x[i], ddy = cy - c.y[i];
            long double dist = sqrtl(ddx*ddx + ddy*ddy);
            if (dist > 1e-6L) {
                for (long double st : steps) {
                    long double ox = c.x[i], oy = c.y[i];
                    c.x[i] += ddx/dist * st; c.y[i] += ddy/dist * st; c.upd(i);
                    if (!c.hasOvl(i)) { c.updGlobal(); if (c.side() < bs - 1e-15L) { bs = c.side(); improved = true; }
                        else { c.x[i]=ox; c.y[i]=oy; c.upd(i); c.updGlobal(); } }
                    else { c.x[i]=ox; c.y[i]=oy; c.upd(i); }
                }
            }
            // 8-directional moves
            for (long double st : steps) {
                for (int d = 0; d < 8; d++) {
                    long double ox=c.x[i], oy=c.y[i];
                    c.x[i] += dx[d]*st; c.y[i] += dy[d]*st; c.upd(i);
                    if (!c.hasOvl(i)) { c.updGlobal(); if (c.side() < bs - 1e-15L) { bs = c.side(); improved = true; }
                        else { c.x[i]=ox; c.y[i]=oy; c.upd(i); c.updGlobal(); } }
                    else { c.x[i]=ox; c.y[i]=oy; c.upd(i); }
                }
            }
            // Rotations
            for (long double rt : rots) {
                for (long double da : {rt, -rt}) {
                    long double oa = c.a[i]; c.a[i] += da;
                    while (c.a[i] < 0) c.a[i] += 360; while (c.a[i] >= 360) c.a[i] -= 360;
                    c.upd(i);
                    if (!c.hasOvl(i)) { c.updGlobal(); if (c.side() < bs - 1e-15L) { bs = c.side(); improved = true; }
                        else { c.a[i]=oa; c.upd(i); c.updGlobal(); } }
                    else { c.a[i]=oa; c.upd(i); }
                }
            }
        }
        if (!improved) break;
    }
    return c;
}

int main(int argc, char** argv) {
    int n_iters = 50000;
    int restarts = 50;
    
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "-n" && i+1 < argc) n_iters = stoi(argv[++i]);
        if (string(argv[i]) == "-r" && i+1 < argc) restarts = stoi(argv[++i]);
    }
    
    parse_csv("submission.csv");
    
    cout << fixed << setprecision(10);
    cout << "Aggressive SA Optimizer\n";
    cout << "=======================\n";
    cout << "Iterations: " << n_iters << ", Restarts: " << restarts << "\n";
    cout << "Initial: " << calc_total_score() << "\n";
    
    int total_improvements = 0;
    
    #pragma omp parallel for schedule(dynamic) reduction(+:total_improvements)
    for (int n = 2; n <= MAX_N; n++) {
        FastRNG rng(n * 98765 + 43210);
        Cfg best = configs[n];
        long double bs = best.side();
        
        for (int r = 0; r < restarts; r++) {
            Cfg c = configs[n];
            
            // Random rotation restart
            if (r > 0) {
                long double angle = rng.rf() * 360.0L;
                long double cx = (c.gx0 + c.gx1) / 2.0L, cy = (c.gy0 + c.gy1) / 2.0L;
                long double rad = angle * PI / 180.0L;
                long double cs = cosl(rad), sn = sinl(rad);
                for (int i = 0; i < c.n; i++) {
                    long double dx = c.x[i] - cx, dy = c.y[i] - cy;
                    c.x[i] = cx + dx * cs - dy * sn;
                    c.y[i] = cy + dx * sn + dy * cs;
                    c.a[i] += angle;
                    while (c.a[i] >= 360) c.a[i] -= 360;
                }
                c.updAll();
            }
            
            // High temperature SA
            c = aggressiveSA(c, n_iters, rng, 0.5L);
            
            // Fine local search
            c = fineLocalSearch(c, 50);
            
            if (c.side() < bs) {
                bs = c.side();
                best = c;
                total_improvements++;
            }
        }
        
        #pragma omp critical
        {
            if (bs < best_sides[n]) {
                configs[n] = best;
                best_sides[n] = bs;
            }
        }
    }
    
    cout << "Total improvements: " << total_improvements << "\n";
    cout << "Final:   " << calc_total_score() << "\n";
    save_csv("submission_aggressive.csv");
    
    return 0;
}
