// Tree Packer v21 - ENHANCED v19 with SWAP MOVES + MULTI-START
// All n values (1-200) processed in parallel + aggressive exploration
// NEW: Swap move operator, multi-angle restarts, higher temperature SA
// Compile: g++ -O3 -march=native -std=c++17 -fopenmp -o tree_packer tree_packer.cpp

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

constexpr int MAX_N = 200;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;

alignas(64) const long double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
alignas(64) const long double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

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

    inline void upd(int i) { getPoly(x[i], y[i], a[i], pl[i]); }
    inline void updAll() { for (int i = 0; i < n; i++) upd(i); updGlobal(); }

    inline void updGlobal() {
        gx0 = gy0 = 1e9L; gx1 = gy1 = -1e9L;
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

    inline long double side() const { return max(gx1 - gx0, gy1 - gy0); }
    inline long double score() const { long double s = side(); return s * s / n; }

    void getBoundary(vector<int>& b) const {
        b.clear();
        long double eps = 0.01L;
        for (int i = 0; i < n; i++) {
            if (pl[i].x0 - gx0 < eps || gx1 - pl[i].x1 < eps ||
                pl[i].y0 - gy0 < eps || gy1 - pl[i].y1 < eps)
                b.push_back(i);
        }
    }

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

// Squeeze
Cfg squeeze(Cfg c) {
    long double cx = (c.gx0 + c.gx1) / 2.0L, cy = (c.gy0 + c.gy1) / 2.0L;
    for (long double scale = 0.9995L; scale >= 0.98L; scale -= 0.0005L) {
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

// Compaction
Cfg compaction(Cfg c, int iters) {
    long double bs = c.side();
    for (int it = 0; it < iters; it++) {
        long double cx = (c.gx0 + c.gx1) / 2.0L, cy = (c.gy0 + c.gy1) / 2.0L;
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            long double ox = c.x[i], oy = c.y[i];
            long double dx = cx - c.x[i], dy = cy - c.y[i];
            long double d = sqrtl(dx*dx + dy*dy);
            if (d < 1e-6L) continue;
            for (long double step : {0.02L, 0.008L, 0.003L, 0.001L, 0.0004L}) {
                c.x[i] = ox + dx/d * step; c.y[i] = oy + dy/d * step; c.upd(i);
                if (!c.hasOvl(i)) {
                    c.updGlobal();
                    if (c.side() < bs - 1e-12L) { bs = c.side(); improved = true; ox = c.x[i]; oy = c.y[i]; }
                    else { c.x[i] = ox; c.y[i] = oy; c.upd(i); }
                } else { c.x[i] = ox; c.y[i] = oy; c.upd(i); }
            }
        }
        c.updGlobal();
        if (!improved) break;
    }
    return c;
}

// Local search
Cfg localSearch(Cfg c, int maxIter) {
    long double bs = c.side();
    const long double steps[] = {0.01L, 0.004L, 0.0015L, 0.0006L, 0.00025L, 0.0001L};
    const long double rots[] = {5.0L, 2.0L, 0.8L, 0.3L, 0.1L};
    const int dx[] = {1,-1,0,0,1,1,-1,-1};
    const int dy[] = {0,0,1,-1,1,-1,1,-1};

    for (int iter = 0; iter < maxIter; iter++) {
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            long double cx = (c.gx0 + c.gx1) / 2.0L, cy = (c.gy0 + c.gy1) / 2.0L;
            long double ddx = cx - c.x[i], ddy = cy - c.y[i];
            long double dist = sqrtl(ddx*ddx + ddy*ddy);
            if (dist > 1e-6L) {
                for (long double st : steps) {
                    long double ox = c.x[i], oy = c.y[i];
                    c.x[i] += ddx/dist * st; c.y[i] += ddy/dist * st; c.upd(i);
                    if (!c.hasOvl(i)) { c.updGlobal(); if (c.side() < bs - 1e-12L) { bs = c.side(); improved = true; }
                        else { c.x[i]=ox; c.y[i]=oy; c.upd(i); c.updGlobal(); } }
                    else { c.x[i]=ox; c.y[i]=oy; c.upd(i); }
                }
            }
            for (long double st : steps) {
                for (int d = 0; d < 8; d++) {
                    long double ox=c.x[i], oy=c.y[i];
                    c.x[i] += dx[d]*st; c.y[i] += dy[d]*st; c.upd(i);
                    if (!c.hasOvl(i)) { c.updGlobal(); if (c.side() < bs - 1e-12L) { bs = c.side(); improved = true; }
                        else { c.x[i]=ox; c.y[i]=oy; c.upd(i); c.updGlobal(); } }
                    else { c.x[i]=ox; c.y[i]=oy; c.upd(i); }
                }
            }
            for (long double rt : rots) {
                for (long double da : {rt, -rt}) {
                    long double oa = c.a[i]; c.a[i] += da;
                    while (c.a[i] >= 360.0L) c.a[i] -= 360.0L;
                    while (c.a[i] < 0.0L) c.a[i] += 360.0L;
                    c.upd(i);
                    if (!c.hasOvl(i)) { c.updGlobal(); if (c.side() < bs - 1e-12L) { bs = c.side(); improved = true; }
                        else { c.a[i]=oa; c.upd(i); c.updGlobal(); } }
                    else { c.a[i]=oa; c.upd(i); }
                }
            }
        }
        if (!improved) break;
    }
    return c;
}

// Swap move - try swapping positions of two trees
Cfg swapMove(Cfg c, FastRNG& rng) {
    if (c.n < 2) return c;
    long double bs = c.side();
    
    for (int attempts = 0; attempts < c.n * 2; attempts++) {
        int i = rng.ri(c.n);
        int j = rng.ri(c.n);
        if (i == j) continue;
        
        // Swap positions
        swap(c.x[i], c.x[j]);
        swap(c.y[i], c.y[j]);
        c.upd(i);
        c.upd(j);
        
        if (!c.hasOvl(i) && !c.hasOvl(j)) {
            c.updGlobal();
            if (c.side() < bs - 1e-12L) {
                return c;
            }
        }
        
        // Revert
        swap(c.x[i], c.x[j]);
        swap(c.y[i], c.y[j]);
        c.upd(i);
        c.upd(j);
    }
    c.updGlobal();
    return c;
}

// Simulated annealing with higher temperature
Cfg simulatedAnnealing(Cfg c, int maxIter, FastRNG& rng, long double T0 = 0.5L) {
    long double bs = c.side();
    Cfg best = c;
    long double T = T0;
    long double Tmin = 1e-6L;
    long double alpha = 0.9995L;
    
    for (int iter = 0; iter < maxIter && T > Tmin; iter++) {
        int i = rng.ri(c.n);
        long double ox = c.x[i], oy = c.y[i], oa = c.a[i];
        
        // Random perturbation
        int moveType = rng.ri(4);
        if (moveType == 0) {
            c.x[i] += rng.gaussian() * 0.02L;
            c.y[i] += rng.gaussian() * 0.02L;
        } else if (moveType == 1) {
            c.a[i] += rng.gaussian() * 3.0L;
            while (c.a[i] >= 360.0L) c.a[i] -= 360.0L;
            while (c.a[i] < 0.0L) c.a[i] += 360.0L;
        } else if (moveType == 2) {
            // Move toward center
            long double cx = (c.gx0 + c.gx1) / 2.0L, cy = (c.gy0 + c.gy1) / 2.0L;
            long double dx = cx - c.x[i], dy = cy - c.y[i];
            long double d = sqrtl(dx*dx + dy*dy);
            if (d > 1e-6L) {
                c.x[i] += dx/d * rng.rf() * 0.02L;
                c.y[i] += dy/d * rng.rf() * 0.02L;
            }
        } else {
            // Small random step
            c.x[i] += rng.rf2() * 0.01L;
            c.y[i] += rng.rf2() * 0.01L;
        }
        
        c.upd(i);
        
        if (!c.hasOvl(i)) {
            c.updGlobal();
            long double ns = c.side();
            long double delta = ns - bs;
            
            if (delta < 0 || rng.rf() < expl(-delta / T)) {
                bs = ns;
                if (ns < best.side()) {
                    best = c;
                }
            } else {
                c.x[i] = ox; c.y[i] = oy; c.a[i] = oa;
                c.upd(i);
                c.updGlobal();
            }
        } else {
            c.x[i] = ox; c.y[i] = oy; c.a[i] = oa;
            c.upd(i);
        }
        
        T *= alpha;
    }
    
    return best;
}

// Initialize configuration with random angles
Cfg initRandom(int n, FastRNG& rng) {
    Cfg c;
    c.n = n;
    
    // Place first tree at origin
    c.x[0] = 0; c.y[0] = 0; c.a[0] = rng.rf() * 360.0L;
    c.upd(0);
    
    // Place remaining trees
    for (int i = 1; i < n; i++) {
        c.a[i] = rng.rf() * 360.0L;
        
        // Try random positions until no overlap
        bool placed = false;
        for (int attempt = 0; attempt < 1000 && !placed; attempt++) {
            long double angle = rng.rf() * 2.0L * PI;
            long double radius = 0.5L + rng.rf() * (0.5L * sqrtl((long double)i));
            c.x[i] = radius * cosl(angle);
            c.y[i] = radius * sinl(angle);
            c.upd(i);
            
            bool hasOverlap = false;
            for (int j = 0; j < i; j++) {
                if (overlap(c.pl[i], c.pl[j])) {
                    hasOverlap = true;
                    break;
                }
            }
            if (!hasOverlap) placed = true;
        }
        
        if (!placed) {
            // Fallback: place far away
            c.x[i] = 10.0L + rng.rf() * 5.0L;
            c.y[i] = 10.0L + rng.rf() * 5.0L;
            c.upd(i);
        }
    }
    
    c.updGlobal();
    return c;
}

// Global best configurations
Cfg bestConfigs[MAX_N + 1];
long double bestSides[MAX_N + 1];

void parseCSV(const string& filename) {
    ifstream f(filename);
    string line;
    getline(f, line); // header
    
    map<int, vector<tuple<long double, long double, long double>>> data;
    
    while (getline(f, line)) {
        // Parse: id,x,y,deg
        size_t p1 = line.find(',');
        size_t p2 = line.find(',', p1+1);
        size_t p3 = line.find(',', p2+1);
        
        string id = line.substr(0, p1);
        string xs = line.substr(p1+1, p2-p1-1);
        string ys = line.substr(p2+1, p3-p2-1);
        string ds = line.substr(p3+1);
        
        // Remove 's' prefix
        if (xs[0] == 's') xs = xs.substr(1);
        if (ys[0] == 's') ys = ys.substr(1);
        if (ds[0] == 's') ds = ds.substr(1);
        
        // Parse group id
        size_t us = id.find('_');
        int n = stoi(id.substr(0, us));
        
        long double x = stold(xs);
        long double y = stold(ys);
        long double deg = stold(ds);
        
        data[n].push_back({x, y, deg});
    }
    
    for (int n = 1; n <= MAX_N; n++) {
        bestConfigs[n].n = n;
        for (int i = 0; i < n; i++) {
            bestConfigs[n].x[i] = get<0>(data[n][i]);
            bestConfigs[n].y[i] = get<1>(data[n][i]);
            bestConfigs[n].a[i] = get<2>(data[n][i]);
        }
        bestConfigs[n].updAll();
        bestSides[n] = bestConfigs[n].side();
    }
}

void saveCSV(const string& filename) {
    ofstream f(filename);
    f << fixed << setprecision(12);
    f << "id,x,y,deg\n";
    
    for (int n = 1; n <= MAX_N; n++) {
        for (int i = 0; i < n; i++) {
            f << setfill('0') << setw(3) << n << "_" << i << ",";
            f << "s" << bestConfigs[n].x[i] << ",";
            f << "s" << bestConfigs[n].y[i] << ",";
            f << "s" << bestConfigs[n].a[i] << "\n";
        }
    }
}

long double calcTotalScore() {
    long double total = 0;
    for (int n = 1; n <= MAX_N; n++) {
        total += bestSides[n] * bestSides[n] / n;
    }
    return total;
}

int main(int argc, char** argv) {
    int nIter = 5000;
    int nRestarts = 16;
    
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "-n" && i+1 < argc) nIter = stoi(argv[++i]);
        if (string(argv[i]) == "-r" && i+1 < argc) nRestarts = stoi(argv[++i]);
    }
    
    cout << "Tree Packer v21\n";
    cout << "Iterations: " << nIter << ", Restarts: " << nRestarts << "\n";
    
    // Initialize with sample submission or random
    string inputFile = "submission.csv";
    ifstream test(inputFile);
    if (test.good()) {
        test.close();
        parseCSV(inputFile);
        cout << "Loaded: " << inputFile << "\n";
    } else {
        // Initialize randomly
        for (int n = 1; n <= MAX_N; n++) {
            FastRNG rng(42 + n);
            bestConfigs[n] = initRandom(n, rng);
            bestSides[n] = bestConfigs[n].side();
        }
        cout << "Initialized randomly\n";
    }
    
    cout << "Initial: " << fixed << setprecision(6) << calcTotalScore() << "\n";
    
    // Optimize all n values in parallel
    #pragma omp parallel for schedule(dynamic)
    for (int n = 1; n <= MAX_N; n++) {
        FastRNG rng(time(0) + n * 1000 + omp_get_thread_num());
        
        Cfg best = bestConfigs[n];
        long double bestSide = best.side();
        
        for (int restart = 0; restart < nRestarts; restart++) {
            // Start from current best or random
            Cfg c;
            if (restart == 0) {
                c = best;
            } else {
                c = initRandom(n, rng);
            }
            
            // Optimization pipeline
            c = squeeze(c);
            c = compaction(c, 10);
            c = localSearch(c, 5);
            c = simulatedAnnealing(c, nIter, rng);
            c = swapMove(c, rng);
            c = squeeze(c);
            c = compaction(c, 20);
            c = localSearch(c, 10);
            
            if (c.side() < bestSide - 1e-12L) {
                bestSide = c.side();
                best = c;
            }
        }
        
        #pragma omp critical
        {
            if (bestSide < bestSides[n] - 1e-12L) {
                bestSides[n] = bestSide;
                bestConfigs[n] = best;
            }
        }
    }
    
    cout << "Final:   " << fixed << setprecision(6) << calcTotalScore() << "\n";
    
    saveCSV("submission_v21.csv");
    cout << "Saved: submission_v21.csv\n";
    
    return 0;
}
