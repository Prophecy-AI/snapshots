// Periodic Structure Simulated Annealing
// Based on egortrushin's approach - treats large N as tiled unit cells
// Compile: g++ -O3 -march=native -std=c++17 -fopenmp -o periodic_sa periodic_sa.cpp

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

// Periodic structure configuration
struct PeriodicConfig {
    int rows, cols;      // Grid dimensions
    long double tx, ty;  // Translation vector between cells
    long double angle;   // Base rotation angle for all trees
    long double base_x, base_y;  // Position of first tree
    bool append_y;       // Whether to add extra row
    
    // Generate trees from periodic config
    void generateTrees(Cfg& c) const {
        int idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int col = 0; col < cols; col++) {
                if (idx >= c.n) break;
                c.x[idx] = base_x + col * tx;
                c.y[idx] = base_y + r * ty;
                c.a[idx] = angle;
                idx++;
            }
        }
        // Handle remaining trees if N != rows * cols
        while (idx < c.n) {
            c.x[idx] = base_x + (idx % cols) * tx;
            c.y[idx] = base_y + rows * ty;
            c.a[idx] = angle;
            idx++;
        }
        c.updAll();
    }
};

// Find good factorizations for N
vector<pair<int,int>> getFactorizations(int n) {
    vector<pair<int,int>> result;
    for (int i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            result.push_back({i, n/i});
            if (i != n/i) {
                result.push_back({n/i, i});
            }
        }
    }
    // Also try factorizations that are close (for append_y mode)
    for (int i = 1; i * i <= n + 10; i++) {
        for (int j = i; i * j <= n + 10 && i * j >= n - 10; j++) {
            if (i * j != n && i * j >= n - j && i * j <= n + j) {
                result.push_back({i, j});
                if (i != j) result.push_back({j, i});
            }
        }
    }
    return result;
}

// Periodic SA optimizer
Cfg periodicSA(int n, int iters, FastRNG& rng) {
    Cfg best;
    best.n = n;
    long double bs = 1e9L;
    
    auto factors = getFactorizations(n);
    
    for (auto [rows, cols] : factors) {
        if (rows * cols < n - cols || rows * cols > n + cols) continue;
        
        // Try multiple starting configurations
        for (int restart = 0; restart < 20; restart++) {
            PeriodicConfig pc;
            pc.rows = rows;
            pc.cols = cols;
            pc.tx = 0.5L + rng.rf() * 0.5L;  // Initial translation x
            pc.ty = 0.5L + rng.rf() * 0.5L;  // Initial translation y
            pc.angle = rng.rf() * 360.0L;
            pc.base_x = 0;
            pc.base_y = 0;
            
            Cfg c;
            c.n = n;
            pc.generateTrees(c);
            
            // Skip if overlapping
            if (c.anyOvl()) continue;
            
            long double T = 0.5L;
            long double alpha = powl(0.00001L / 0.5L, 1.0L / iters);
            long double cs = c.side();
            Cfg cbest = c;
            long double cbs = cs;
            
            for (int it = 0; it < iters; it++) {
                int moveType = rng.ri(5);
                
                PeriodicConfig pc_new = pc;
                
                if (moveType == 0) {
                    // Adjust tx
                    pc_new.tx += rng.gaussian() * T * 0.1L;
                } else if (moveType == 1) {
                    // Adjust ty
                    pc_new.ty += rng.gaussian() * T * 0.1L;
                } else if (moveType == 2) {
                    // Adjust angle
                    pc_new.angle += rng.gaussian() * T * 30.0L;
                    while (pc_new.angle < 0) pc_new.angle += 360;
                    while (pc_new.angle >= 360) pc_new.angle -= 360;
                } else if (moveType == 3) {
                    // Adjust base position
                    pc_new.base_x += rng.gaussian() * T * 0.1L;
                    pc_new.base_y += rng.gaussian() * T * 0.1L;
                } else {
                    // Combined move
                    pc_new.tx += rng.gaussian() * T * 0.05L;
                    pc_new.ty += rng.gaussian() * T * 0.05L;
                    pc_new.angle += rng.gaussian() * T * 15.0L;
                    while (pc_new.angle < 0) pc_new.angle += 360;
                    while (pc_new.angle >= 360) pc_new.angle -= 360;
                }
                
                Cfg c_new;
                c_new.n = n;
                pc_new.generateTrees(c_new);
                
                if (!c_new.anyOvl()) {
                    long double ns = c_new.side();
                    if (ns < cbs) {
                        cbs = ns;
                        cbest = c_new;
                        pc = pc_new;
                    } else if (rng.rf() < expl(-(ns - cs) / T)) {
                        cs = ns;
                        c = c_new;
                        pc = pc_new;
                    }
                }
                
                T *= alpha;
            }
            
            if (cbs < bs) {
                bs = cbs;
                best = cbest;
            }
        }
    }
    
    return best;
}

// Standard SA for comparison
Cfg standardSA(Cfg c, int iters, FastRNG& rng) {
    long double bs = c.side();
    Cfg best = c;
    long double T = 0.5L;
    long double alpha = powl(0.00001L / 0.5L, 1.0L / iters);
    
    for (int it = 0; it < iters; it++) {
        int moveType = rng.ri(4);
        
        if (moveType == 0) {
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
        } else if (moveType == 1) {
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
        } else if (moveType == 2 && c.n >= 2) {
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
        } else {
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

int main(int argc, char** argv) {
    int iters = 100000;
    int min_n = 20;  // Start periodic optimization from this N
    
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "-n" && i+1 < argc) iters = stoi(argv[++i]);
        if (string(argv[i]) == "-m" && i+1 < argc) min_n = stoi(argv[++i]);
    }
    
    parse_csv("submission.csv");
    
    cout << fixed << setprecision(10);
    cout << "Periodic Structure SA Optimizer\n";
    cout << "================================\n";
    cout << "Iterations: " << iters << ", Min N for periodic: " << min_n << "\n";
    cout << "Initial: " << calc_total_score() << "\n";
    
    int improvements = 0;
    
    // Process N values in parallel
    #pragma omp parallel for schedule(dynamic) reduction(+:improvements)
    for (int n = min_n; n <= MAX_N; n++) {
        FastRNG rng(n * 54321 + 98765);
        
        // Try periodic structure optimization
        Cfg periodic_best = periodicSA(n, iters, rng);
        
        // Also try standard SA on current config
        Cfg standard_best = standardSA(configs[n], iters / 2, rng);
        
        // Keep the best
        long double periodic_side = periodic_best.n > 0 ? periodic_best.side() : 1e9L;
        long double standard_side = standard_best.side();
        long double current_side = best_sides[n];
        
        #pragma omp critical
        {
            if (periodic_side < current_side && periodic_side < standard_side) {
                if (!periodic_best.anyOvl()) {
                    cout << "N=" << n << " improved by periodic: " << current_side << " -> " << periodic_side << "\n";
                    configs[n] = periodic_best;
                    best_sides[n] = periodic_side;
                    improvements++;
                }
            } else if (standard_side < current_side) {
                if (!standard_best.anyOvl()) {
                    cout << "N=" << n << " improved by standard: " << current_side << " -> " << standard_side << "\n";
                    configs[n] = standard_best;
                    best_sides[n] = standard_side;
                    improvements++;
                }
            }
        }
    }
    
    cout << "\nTotal improvements: " << improvements << "\n";
    cout << "Final: " << calc_total_score() << "\n";
    save_csv("submission_periodic.csv");
    
    return 0;
}
