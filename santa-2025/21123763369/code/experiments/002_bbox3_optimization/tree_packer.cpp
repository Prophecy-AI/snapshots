// Tree Packer - Simulated Annealing optimizer for Christmas tree packing
// Simplified version based on santa-claude kernel

#include <bits/stdc++.h>
using namespace std;

constexpr int MAX_N = 200;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;

const double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
const double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

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
    inline double rf() { return (next() >> 11) * 0x1.0p-53; }
    inline double rf2() { return rf() * 2.0 - 1.0; }
    inline int ri(int n) { return next() % n; }
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
        q.px[i] = x; q.py[i] = y;
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
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
        gx0 = gy0 = 1e9; gx1 = gy1 = -1e9;
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

    inline double side() const { return max(gx1 - gx0, gy1 - gy0); }
    inline double score() const { double s = side(); return s * s / n; }
};

// Local search
Cfg localSearch(Cfg c, int maxIter, FastRNG& rng) {
    double bs = c.side();
    const double steps[] = {0.01, 0.004, 0.0015, 0.0006, 0.00025, 0.0001};
    const double rots[] = {5.0, 2.0, 0.8, 0.3, 0.1};
    const int dx[] = {1,-1,0,0,1,1,-1,-1};
    const int dy[] = {0,0,1,-1,1,-1,1,-1};

    for (int iter = 0; iter < maxIter; iter++) {
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            // Move towards center
            double cx = (c.gx0 + c.gx1) / 2.0, cy = (c.gy0 + c.gy1) / 2.0;
            double ddx = cx - c.x[i], ddy = cy - c.y[i];
            double dist = sqrt(ddx*ddx + ddy*ddy);
            if (dist > 1e-6) {
                for (double st : steps) {
                    double ox = c.x[i], oy = c.y[i];
                    c.x[i] += ddx/dist * st; c.y[i] += ddy/dist * st; c.upd(i);
                    if (!c.hasOvl(i)) { 
                        c.updGlobal(); 
                        if (c.side() < bs - 1e-12) { bs = c.side(); improved = true; }
                        else { c.x[i]=ox; c.y[i]=oy; c.upd(i); c.updGlobal(); } 
                    }
                    else { c.x[i]=ox; c.y[i]=oy; c.upd(i); }
                }
            }
            // 8-directional moves
            for (double st : steps) {
                for (int d = 0; d < 8; d++) {
                    double ox=c.x[i], oy=c.y[i];
                    c.x[i] += dx[d]*st; c.y[i] += dy[d]*st; c.upd(i);
                    if (!c.hasOvl(i)) { 
                        c.updGlobal(); 
                        if (c.side() < bs - 1e-12) { bs = c.side(); improved = true; }
                        else { c.x[i]=ox; c.y[i]=oy; c.upd(i); c.updGlobal(); } 
                    }
                    else { c.x[i]=ox; c.y[i]=oy; c.upd(i); }
                }
            }
            // Rotation adjustments
            for (double rt : rots) {
                for (double da : {rt, -rt}) {
                    double oa = c.a[i]; c.a[i] += da;
                    while (c.a[i] >= 360) c.a[i] -= 360;
                    while (c.a[i] < 0) c.a[i] += 360;
                    c.upd(i);
                    if (!c.hasOvl(i)) { 
                        c.updGlobal(); 
                        if (c.side() < bs - 1e-12) { bs = c.side(); improved = true; }
                        else { c.a[i]=oa; c.upd(i); c.updGlobal(); } 
                    }
                    else { c.a[i]=oa; c.upd(i); }
                }
            }
        }
        if (!improved) break;
    }
    return c;
}

// Squeeze towards center
Cfg squeeze(Cfg c) {
    double cx = (c.gx0 + c.gx1) / 2.0, cy = (c.gy0 + c.gy1) / 2.0;
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

map<int, Cfg> configs;
map<int, double> best_sides;

void parse_csv(const string& filename) {
    ifstream file(filename);
    string line;
    getline(file, line); // header
    
    while (getline(file, line)) {
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
        
        // Parse group number
        int n = stoi(id.substr(0, 3));
        int idx = stoi(id.substr(4));
        
        if (configs.find(n) == configs.end()) {
            configs[n].n = 0;
        }
        
        configs[n].x[idx] = stod(xs);
        configs[n].y[idx] = stod(ys);
        configs[n].a[idx] = stod(ds);
        if (idx >= configs[n].n) configs[n].n = idx + 1;
    }
    
    // Update all polygons
    for (auto& [n, cfg] : configs) {
        cfg.updAll();
        best_sides[n] = cfg.side();
    }
}

void save_csv(const string& filename) {
    ofstream file(filename);
    file << fixed << setprecision(17);
    file << "id,x,y,deg\n";
    
    for (int n = 1; n <= MAX_N; n++) {
        if (configs.find(n) == configs.end()) continue;
        const Cfg& cfg = configs[n];
        for (int i = 0; i < cfg.n; i++) {
            file << setfill('0') << setw(3) << n << "_" << i << ",";
            file << "s" << cfg.x[i] << ",s" << cfg.y[i] << ",s" << cfg.a[i] << "\n";
        }
    }
}

double calc_total_score() {
    double total = 0;
    for (auto& [n, side] : best_sides) {
        total += side * side / n;
    }
    return total;
}

int main(int argc, char** argv) {
    int max_iters = 100;
    int radius = 30;
    
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "-n" && i+1 < argc) max_iters = stoi(argv[++i]);
        if (string(argv[i]) == "-r" && i+1 < argc) radius = stoi(argv[++i]);
    }
    
    cout << "Tree Packer - Local Search Optimizer\n";
    cout << "Max iterations: " << max_iters << ", Radius: " << radius << "\n";
    
    parse_csv("submission.csv");
    cout << "Loaded " << configs.size() << " configurations\n";
    cout << fixed << setprecision(8) << "Initial score: " << calc_total_score() << "\n";
    
    FastRNG rng(42);
    int improved_count = 0;
    
    for (int n = 1; n <= MAX_N; n++) {
        if (configs.find(n) == configs.end()) continue;
        
        Cfg& cfg = configs[n];
        double old_side = cfg.side();
        
        // Apply squeeze
        cfg = squeeze(cfg);
        
        // Apply local search
        cfg = localSearch(cfg, max_iters, rng);
        
        double new_side = cfg.side();
        if (new_side < old_side - 1e-10) {
            best_sides[n] = new_side;
            improved_count++;
            if (n <= 20 || old_side - new_side > 0.001) {
                cout << "n=" << n << ": " << old_side << " -> " << new_side 
                     << " (improved by " << (old_side - new_side) << ")\n";
            }
        }
    }
    
    cout << "\nImproved " << improved_count << " configurations\n";
    cout << "Final score: " << calc_total_score() << "\n";
    
    save_csv("submission.csv");
    cout << "Saved to submission.csv\n";
    
    return 0;
}
