// Small N Optimizer - Generate fresh configurations for N=1 to 30
// Uses exhaustive search for very small N and aggressive SA for larger
// Compile: g++ -O3 -march=native -std=c++17 -fopenmp -o small_n_opt small_n_optimizer.cpp

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

// Optimize N=1: Find optimal single tree rotation
void optimize_n1() {
    Cfg& c = configs[1];
    long double best_side = 1e9L;
    long double best_angle = 0;
    
    // Try many angles
    for (int a = 0; a < 3600; a++) {
        long double angle = a * 0.1L;
        c.a[0] = angle;
        c.x[0] = 0;
        c.y[0] = 0;
        c.upd(0);
        c.updGlobal();
        long double s = c.side();
        if (s < best_side) {
            best_side = s;
            best_angle = angle;
        }
    }
    
    c.a[0] = best_angle;
    c.x[0] = 0;
    c.y[0] = 0;
    c.updAll();
    
    if (best_side < best_sides[1]) {
        best_sides[1] = best_side;
        cout << "N=1 improved: " << best_side << " (angle=" << best_angle << ")\n";
    }
}

// Generate random valid configuration
bool generateRandom(Cfg& c, int n, FastRNG& rng, long double radius) {
    c.n = n;
    
    for (int i = 0; i < n; i++) {
        bool placed = false;
        for (int attempt = 0; attempt < 1000; attempt++) {
            c.x[i] = rng.gaussian() * radius;
            c.y[i] = rng.gaussian() * radius;
            c.a[i] = rng.rf() * 360.0L;
            c.upd(i);
            
            bool valid = true;
            for (int j = 0; j < i; j++) {
                if (overlap(c.pl[i], c.pl[j])) {
                    valid = false;
                    break;
                }
            }
            
            if (valid) {
                placed = true;
                break;
            }
        }
        
        if (!placed) return false;
    }
    
    c.updGlobal();
    return true;
}

// Aggressive SA for small N
Cfg smallNSA(Cfg c, int iters, FastRNG& rng) {
    long double bs = c.side();
    Cfg best = c;
    long double T = 1.0L;
    long double alpha = powl(0.00001L / 1.0L, 1.0L / iters);
    
    for (int it = 0; it < iters; it++) {
        int moveType = rng.ri(4);
        
        if (moveType == 0) { // Translation
            int i = rng.ri(c.n);
            long double ox = c.x[i], oy = c.y[i];
            c.x[i] += rng.gaussian() * T * 0.5L;
            c.y[i] += rng.gaussian() * T * 0.5L;
            c.upd(i);
            if (c.hasOvl(i)) { c.x[i] = ox; c.y[i] = oy; c.upd(i); }
            else {
                c.updGlobal();
                long double ns = c.side();
                if (ns < bs) { bs = ns; best = c; }
                else if (rng.rf() > expl(-(ns - bs) / T)) { c.x[i] = ox; c.y[i] = oy; c.upd(i); c.updGlobal(); }
            }
        } else if (moveType == 1) { // Rotation
            int i = rng.ri(c.n);
            long double oa = c.a[i];
            c.a[i] += rng.gaussian() * T * 45.0L;
            while (c.a[i] < 0) c.a[i] += 360; while (c.a[i] >= 360) c.a[i] -= 360;
            c.upd(i);
            if (c.hasOvl(i)) { c.a[i] = oa; c.upd(i); }
            else {
                c.updGlobal();
                long double ns = c.side();
                if (ns < bs) { bs = ns; best = c; }
                else if (rng.rf() > expl(-(ns - bs) / T)) { c.a[i] = oa; c.upd(i); c.updGlobal(); }
            }
        } else if (moveType == 2 && c.n >= 2) { // Swap
            int i = rng.ri(c.n), j = rng.ri(c.n);
            if (i != j) {
                swap(c.x[i], c.x[j]); swap(c.y[i], c.y[j]); swap(c.a[i], c.a[j]);
                c.upd(i); c.upd(j);
                if (c.hasOvl(i) || c.hasOvl(j)) {
                    swap(c.x[i], c.x[j]); swap(c.y[i], c.y[j]); swap(c.a[i], c.a[j]);
                    c.upd(i); c.upd(j);
                } else {
                    c.updGlobal();
                    long double ns = c.side();
                    if (ns < bs) { bs = ns; best = c; }
                    else if (rng.rf() > expl(-(ns - bs) / T)) {
                        swap(c.x[i], c.x[j]); swap(c.y[i], c.y[j]); swap(c.a[i], c.a[j]);
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
                c.x[i] += dx/d * T * 0.2L; c.y[i] += dy/d * T * 0.2L;
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

int main(int argc, char** argv) {
    int max_small_n = 30;
    int restarts = 200;
    int iters = 100000;
    
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "-m" && i+1 < argc) max_small_n = stoi(argv[++i]);
        if (string(argv[i]) == "-r" && i+1 < argc) restarts = stoi(argv[++i]);
        if (string(argv[i]) == "-n" && i+1 < argc) iters = stoi(argv[++i]);
    }
    
    parse_csv("submission.csv");
    
    cout << fixed << setprecision(10);
    cout << "Small N Optimizer\n";
    cout << "=================\n";
    cout << "Max N: " << max_small_n << ", Restarts: " << restarts << ", Iters: " << iters << "\n";
    cout << "Initial: " << calc_total_score() << "\n";
    
    // Optimize N=1
    optimize_n1();
    
    // Optimize small N values
    #pragma omp parallel for schedule(dynamic)
    for (int n = 2; n <= max_small_n; n++) {
        FastRNG rng(n * 11111 + 22222);
        Cfg best = configs[n];
        long double bs = best.side();
        
        for (int r = 0; r < restarts; r++) {
            Cfg c;
            
            // Try generating from scratch
            if (r % 2 == 0) {
                long double radius = sqrtl((long double)n) * 0.5L;
                if (!generateRandom(c, n, rng, radius)) {
                    c = configs[n];
                }
            } else {
                c = configs[n];
                // Random rotation
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
            
            c = smallNSA(c, iters, rng);
            
            if (c.side() < bs) {
                bs = c.side();
                best = c;
            }
        }
        
        #pragma omp critical
        {
            if (bs < best_sides[n]) {
                cout << "N=" << n << " improved: " << best_sides[n] << " -> " << bs << "\n";
                configs[n] = best;
                best_sides[n] = bs;
            }
        }
    }
    
    cout << "Final:   " << calc_total_score() << "\n";
    save_csv("submission_small_n.csv");
    
    return 0;
}
