// Random Start Optimizer - Generate random initial configurations and optimize
// This explores different basins of the solution space
// Compile: g++ -O3 -march=native -std=c++17 -fopenmp -o random_start random_start_optimizer.cpp

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

constexpr int MN = 200;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;

double TX[NV] = {0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125};
double TY[NV] = {0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5};

struct Poly {
    double px[NV], py[NV];
    double x0, y0, x1, y1;
};

inline void getPoly(double cx, double cy, double deg, Poly& q) {
    double rad = deg * PI / 180.0;
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
    double d1 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx);
    double d2 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx);
    double d3 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    double d4 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax);
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
    double x[MN], y[MN], a[MN];
    Poly pl[MN];
    double gx0, gy0, gx1, gy1;

    void upd(int i) { getPoly(x[i], y[i], a[i], pl[i]); }
    void updAll() { for (int i = 0; i < n; i++) upd(i); updGlobal(); }
    void updGlobal() {
        gx0 = gy0 = 1e9; gx1 = gy1 = -1e9;
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
    double side() const { return max(gx1 - gx0, gy1 - gy0); }
    double score() const { double s = side(); return s * s / n; }
};

thread_local mt19937_64 rng_engine;
inline double rf() { return uniform_real_distribution<double>(0, 1)(rng_engine); }
inline double rf2() { return rf() * 2 - 1; }
inline int ri(int n) { return uniform_int_distribution<int>(0, n - 1)(rng_engine); }

// Generate a random valid configuration
bool generateRandom(Cfg& c, int n, double radius) {
    c.n = n;
    for (int i = 0; i < n; i++) {
        bool placed = false;
        for (int attempt = 0; attempt < 2000; attempt++) {
            c.x[i] = rf2() * radius;
            c.y[i] = rf2() * radius;
            c.a[i] = rf() * 360.0;
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

// Simulated Annealing
Cfg sa(Cfg c, int iters, double T0, double T1, int seed) {
    rng_engine.seed(seed);
    Cfg best = c;
    double bs = best.side();
    double T = T0;
    double alpha = pow(T1 / T0, 1.0 / iters);

    for (int it = 0; it < iters; it++) {
        int i = ri(c.n);
        int mt = ri(4);
        double ox = c.x[i], oy = c.y[i], oa = c.a[i];

        if (mt == 0) {
            c.x[i] += rf2() * 0.25 * T;
            c.y[i] += rf2() * 0.25 * T;
        } else if (mt == 1) {
            c.a[i] = fmod(c.a[i] + rf2() * 70.0 * T + 360, 360.0);
        } else if (mt == 2 && c.n >= 2) {
            int j = ri(c.n);
            if (j != i) {
                swap(c.x[i], c.x[j]); swap(c.y[i], c.y[j]);
                c.upd(i); c.upd(j);
                if (c.hasOvl(i) || c.hasOvl(j)) {
                    swap(c.x[i], c.x[j]); swap(c.y[i], c.y[j]);
                    c.upd(i); c.upd(j);
                } else {
                    c.updGlobal();
                    double ns = c.side();
                    if (ns < bs) { bs = ns; best = c; }
                    else if (rf() > exp(-(ns - bs) / T)) {
                        swap(c.x[i], c.x[j]); swap(c.y[i], c.y[j]);
                        c.upd(i); c.upd(j); c.updGlobal();
                    }
                }
                T *= alpha;
                continue;
            }
        } else {
            double cx = 0, cy = 0;
            for (int j = 0; j < c.n; j++) { cx += c.x[j]; cy += c.y[j]; }
            cx /= c.n; cy /= c.n;
            double dx = cx - c.x[i], dy = cy - c.y[i];
            double d = sqrt(dx * dx + dy * dy);
            if (d > 1e-6) {
                c.x[i] += dx / d * 0.125 * T;
                c.y[i] += dy / d * 0.125 * T;
            }
        }

        c.upd(i);
        if (c.hasOvl(i)) {
            c.x[i] = ox; c.y[i] = oy; c.a[i] = oa; c.upd(i);
        } else {
            c.updGlobal();
            double ns = c.side();
            if (ns < bs) { bs = ns; best = c; }
            else if (rf() > exp(-(ns - bs) / T)) {
                c.x[i] = ox; c.y[i] = oy; c.a[i] = oa; c.upd(i); c.updGlobal();
            }
        }
        T *= alpha;
    }
    return best;
}

// Local Search
Cfg ls(Cfg c, int max_iter) {
    Cfg best = c;
    double bs = best.side();
    double steps[] = {0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001};
    double rots[] = {5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05};
    double dx[] = {0, 0, 1, -1, 1, 1, -1, -1};
    double dy[] = {1, -1, 0, 0, 1, -1, 1, -1};

    for (int iter = 0; iter < max_iter; iter++) {
        bool improved = false;
        for (int i = 0; i < best.n; i++) {
            for (double step : steps) {
                for (int d = 0; d < 8; d++) {
                    double ox = best.x[i], oy = best.y[i];
                    best.x[i] += dx[d] * step;
                    best.y[i] += dy[d] * step;
                    best.upd(i);
                    if (!best.hasOvl(i)) {
                        best.updGlobal();
                        double ns = best.side();
                        if (ns < bs - 1e-15) {
                            bs = ns;
                            improved = true;
                        } else {
                            best.x[i] = ox; best.y[i] = oy; best.upd(i);
                        }
                    } else {
                        best.x[i] = ox; best.y[i] = oy; best.upd(i);
                    }
                }
            }
            for (double rot : rots) {
                for (double da : {rot, -rot}) {
                    double oa = best.a[i];
                    best.a[i] = fmod(best.a[i] + da + 360, 360.0);
                    best.upd(i);
                    if (!best.hasOvl(i)) {
                        best.updGlobal();
                        double ns = best.side();
                        if (ns < bs - 1e-15) {
                            bs = ns;
                            improved = true;
                        } else {
                            best.a[i] = oa; best.upd(i);
                        }
                    } else {
                        best.a[i] = oa; best.upd(i);
                    }
                }
            }
        }
        if (!improved) break;
    }
    return best;
}

map<int, Cfg> loadCSV(const string& fn) {
    map<int, Cfg> cfg;
    ifstream f(fn);
    if (!f) { cerr << "Cannot open " << fn << endl; return cfg; }
    string ln; getline(f, ln);
    map<int, vector<tuple<int, double, double, double>>> data;
    while (getline(f, ln)) {
        auto p1 = ln.find(','), p2 = ln.find(',', p1 + 1), p3 = ln.find(',', p2 + 1);
        string id = ln.substr(0, p1);
        string xs = ln.substr(p1 + 1, p2 - p1 - 1);
        string ys = ln.substr(p2 + 1, p3 - p2 - 1);
        string ds = ln.substr(p3 + 1);
        if (xs[0] == 's') xs = xs.substr(1);
        if (ys[0] == 's') ys = ys.substr(1);
        if (ds[0] == 's') ds = ds.substr(1);
        int n = stoi(id.substr(0, 3)), idx = stoi(id.substr(4));
        data[n].push_back({idx, stod(xs), stod(ys), stod(ds)});
    }
    for (auto& [n, v] : data) {
        Cfg c; c.n = n;
        for (auto& [i, x, y, d] : v) if (i < n) { c.x[i] = x; c.y[i] = y; c.a[i] = d; }
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
            for (int i = 0; i < n; i++)
                f << setfill('0') << setw(3) << n << "_" << i
                  << ",s" << c.x[i] << ",s" << c.y[i] << ",s" << c.a[i] << "\n";
        }
    }
}

int main(int argc, char** argv) {
    string in = "submission.csv";
    string out = "submission_random.csv";
    int restarts = 50;
    int iters = 50000;

    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "-i" && i + 1 < argc) in = argv[++i];
        else if (a == "-o" && i + 1 < argc) out = argv[++i];
        else if (a == "-n" && i + 1 < argc) iters = stoi(argv[++i]);
        else if (a == "-r" && i + 1 < argc) restarts = stoi(argv[++i]);
    }

    int threads = omp_get_max_threads();
    omp_set_num_threads(threads);
    cout << "Using " << threads << " threads\n";
    cout << "Iterations: " << iters << ", Restarts: " << restarts << "\n";

    auto cfg = loadCSV(in);
    if (cfg.empty()) { cerr << "No data!\n"; return 1; }

    map<int, Cfg> best_so_far = cfg;
    double global_best_score = 0;
    for (const auto& [n, c] : best_so_far) global_best_score += c.score();

    cout << fixed << setprecision(10);
    cout << "Starting score: " << global_best_score << "\n\n";

    int total_improvements = 0;

    #pragma omp parallel for schedule(dynamic, 1) reduction(+:total_improvements)
    for (int n = 1; n <= 200; n++) {
        if (!cfg.count(n)) continue;

        rng_engine.seed(n * 12345 + 67890);
        
        Cfg best = cfg[n];
        double bs = best.side();

        // Try random starts
        for (int r = 0; r < restarts; r++) {
            Cfg c;
            double radius = sqrt((double)n) * 0.6;
            
            if (r == 0) {
                // First restart: use existing config
                c = cfg[n];
            } else if (r % 3 == 0) {
                // Generate random config
                if (!generateRandom(c, n, radius)) {
                    c = cfg[n];
                }
            } else {
                // Rotate existing config
                c = cfg[n];
                double angle = rf() * 360.0;
                double cx = (c.gx0 + c.gx1) / 2.0, cy = (c.gy0 + c.gy1) / 2.0;
                double rad = angle * PI / 180.0;
                double cs = cos(rad), sn = sin(rad);
                for (int i = 0; i < c.n; i++) {
                    double dx = c.x[i] - cx, dy = c.y[i] - cy;
                    c.x[i] = cx + dx * cs - dy * sn;
                    c.y[i] = cy + dx * sn + dy * cs;
                    c.a[i] = fmod(c.a[i] + angle + 360, 360.0);
                }
                c.updAll();
            }

            // Optimize
            c = sa(c, iters, 1.0, 0.000001, r * 1000 + n);
            c = ls(c, 200);

            if (c.side() < bs) {
                bs = c.side();
                best = c;
            }
        }

        #pragma omp critical
        {
            if (bs < best_so_far[n].side()) {
                double old_score = best_so_far[n].score();
                double new_score = best.score();
                cout << "N=" << n << " improved: " << old_score << " -> " << new_score << endl;
                best_so_far[n] = best;
                total_improvements++;
            }
        }
    }

    double final_score = 0;
    for (const auto& [n, c] : best_so_far) final_score += c.score();

    cout << "\nTotal improvements: " << total_improvements << endl;
    cout << "Final score: " << final_score << endl;
    
    saveCSV(out, best_so_far);
    cout << "Saved to: " << out << endl;

    return 0;
}
