// sa_v1_parallel.cpp - Based on jonathanchan's optimizer
// Includes: SA, local search, fractional translation, population-based optimization
// Compile: g++ -O3 -march=native -std=c++17 -fopenmp -o sa_v1_parallel sa_v1_parallel.cpp

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

// Random number generator
thread_local mt19937_64 rng_engine;
inline double rf() { return uniform_real_distribution<double>(0, 1)(rng_engine); }
inline double rf2() { return rf() * 2 - 1; }
inline int ri(int n) { return uniform_int_distribution<int>(0, n - 1)(rng_engine); }

// Simulated Annealing
Cfg sa_v3(Cfg c, int iters, double T0, double T1, double move_scale, double rot_scale, int seed) {
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
            c.x[i] += rf2() * move_scale * T;
            c.y[i] += rf2() * move_scale * T;
        } else if (mt == 1) {
            c.a[i] = fmod(c.a[i] + rf2() * rot_scale * T + 360, 360.0);
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
                c.x[i] += dx / d * move_scale * T * 0.5;
                c.y[i] += dy / d * move_scale * T * 0.5;
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
Cfg ls_v3(Cfg c, int max_iter) {
    Cfg best = c;
    double bs = best.side();
    double steps[] = {0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001};
    double rots[] = {5.0, 2.0, 1.0, 0.5, 0.2, 0.1};
    double dx[] = {0, 0, 1, -1, 1, 1, -1, -1};
    double dy[] = {1, -1, 0, 0, 1, -1, 1, -1};

    for (int iter = 0; iter < max_iter; iter++) {
        bool improved = false;
        for (int i = 0; i < best.n; i++) {
            // Translations
            for (double step : steps) {
                for (int d = 0; d < 8; d++) {
                    double ox = best.x[i], oy = best.y[i];
                    best.x[i] += dx[d] * step;
                    best.y[i] += dy[d] * step;
                    best.upd(i);
                    if (!best.hasOvl(i)) {
                        best.updGlobal();
                        double ns = best.side();
                        if (ns < bs - 1e-12) {
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
            // Rotations
            for (double rot : rots) {
                for (double da : {rot, -rot}) {
                    double oa = best.a[i];
                    best.a[i] = fmod(best.a[i] + da + 360, 360.0);
                    best.upd(i);
                    if (!best.hasOvl(i)) {
                        best.updGlobal();
                        double ns = best.side();
                        if (ns < bs - 1e-12) {
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

// Perturbation
Cfg perturb(Cfg c, double scale, int seed) {
    rng_engine.seed(seed);
    for (int iter = 0; iter < 50; iter++) {
        bool fixed = true;
        for (int i = 0; i < c.n; i++) {
            if (c.hasOvl(i)) {
                fixed = false;
                double cx = 0, cy = 0;
                for (int j = 0; j < c.n; j++) { cx += c.x[j]; cy += c.y[j]; }
                cx /= c.n; cy /= c.n;
                double dx = cx - c.x[i], dy = cy - c.y[i];
                double d = sqrt(dx * dx + dy * dy);
                if (d > 1e-6) {
                    c.x[i] -= dx / d * 0.02;
                    c.y[i] -= dy / d * 0.02;
                }
                c.a[i] = fmod(c.a[i] + rf() * 20 - 10 + 360, 360.0);
                c.upd(i);
            }
        }
        if (fixed) break;
    }
    return c;
}

// Fractional Translation
Cfg fractional_translation(Cfg c, int max_iter = 200) {
    Cfg best = c;
    double bs = best.side();
    double frac_steps[] = {0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001};
    double dx[] = {0, 0, 1, -1, 1, 1, -1, -1};
    double dy[] = {1, -1, 0, 0, 1, -1, 1, -1};
    for (int iter = 0; iter < max_iter; iter++) {
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            for (double step : frac_steps) {
                for (int d = 0; d < 8; d++) {
                    double ox = best.x[i], oy = best.y[i];
                    best.x[i] += dx[d] * step;
                    best.y[i] += dy[d] * step;
                    best.upd(i);
                    if (!best.hasOvl(i)) {
                        best.updGlobal();
                        double ns = best.side();
                        if (ns < bs - 1e-12) {
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
        }
        if (!improved) break;
    }
    return best;
}

// Population-based optimization
Cfg opt_v3(Cfg c, int nr, int si) {
    Cfg best = c;
    double bs = best.side();
    vector<pair<double, Cfg>> pop;
    pop.push_back({bs, c});
    for (int r = 0; r < nr; r++) {
        Cfg start;
        if (r == 0) {
            start = c;
        } else if (r < (int)pop.size()) {
            start = pop[r % pop.size()].second;
        } else {
            start = perturb(pop[0].second, 0.1 + 0.05 * (r % 3), 42 + r * 1000 + c.n);
        }
        Cfg o = sa_v3(start, si, 1.0, 0.000005, 0.25, 70.0, 42 + r * 1000 + c.n);
        o = ls_v3(o, 300);
        o = fractional_translation(o, 150);
        double s = o.side();
        pop.push_back({s, o});
        sort(pop.begin(), pop.end(), [](const pair<double, Cfg>& a, const pair<double, Cfg>& b) {
            return a.first < b.first;
        });
        if (pop.size() > 3) pop.resize(3);
        if (s < bs) {
            bs = s;
            best = o;
        }
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
    string out = "submission_opt.csv";
    int si = 20000, nr = 80;
    int max_generations = 3;

    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "-i" && i + 1 < argc) in = argv[++i];
        else if (a == "-o" && i + 1 < argc) out = argv[++i];
        else if (a == "-n" && i + 1 < argc) si = stoi(argv[++i]);
        else if (a == "-r" && i + 1 < argc) nr = stoi(argv[++i]);
        else if (a == "-g" && i + 1 < argc) max_generations = stoi(argv[++i]);
    }

    int threads = omp_get_max_threads();
    omp_set_num_threads(threads);
    cout << "Using " << threads << " threads\n";
    cout << "Iterations: " << si << ", Restarts: " << nr << ", Generations: " << max_generations << "\n";

    auto cfg = loadCSV(in);
    if (cfg.empty()) { cerr << "No data!\n"; return 1; }

    map<int, Cfg> best_so_far = cfg;
    double global_best_score = 0;
    for (const auto& [n, c] : best_so_far) global_best_score += c.score();

    cout << fixed << setprecision(10);
    cout << "Starting score: " << global_best_score << "\n\n";

    for (int generation = 1; generation <= max_generations; generation++) {
        cout << "\n=== Generation " << generation << " ===" << endl;

        map<int, Cfg> current = best_so_far;
        map<int, Cfg> local;

        #pragma omp parallel for schedule(dynamic, 1)
        for (int n = 1; n <= 200; n++) {
            if (!current.count(n)) continue;

            Cfg c = current[n];

            int it = si, r = nr;
            if (n <= 20) { r = max(6, nr); it = int(si * 1.5); }
            else if (n <= 50) { r = max(5, nr); it = int(si * 1.3); }
            else if (n > 150) { r = max(4, nr); it = int(si * 0.8); }

            Cfg candidate = opt_v3(c, r, it);
            candidate = fractional_translation(candidate, 120);

            #pragma omp critical
            {
                local[n] = candidate;
            }
        }

        int improvements = 0;
        for (auto& p : local) {
            int n = p.first;
            Cfg& cand = p.second;
            double old_n_score = current[n].score();
            double new_n_score = cand.score();

            if (new_n_score < old_n_score - 1e-9) {
                current[n] = cand;
                double improvement = (old_n_score - new_n_score) / old_n_score * 100.0;
                cout << "n=" << setw(3) << n << "  "
                     << old_n_score << " -> " << new_n_score
                     << "  (+" << fixed << setprecision(4) << improvement << "%)" << endl;
                improvements++;
            }
        }

        double new_total = 0;
        for (const auto& [n, c] : current) new_total += c.score();

        bool improved = (new_total < global_best_score - 1e-8);

        if (improved) {
            global_best_score = new_total;
            best_so_far = current;
            cout << "\nNEW GLOBAL BEST! -> " << global_best_score << endl;
        } else {
            cout << "Generation " << generation << " finished - no global improvement ("
                 << new_total << ")" << endl;
        }

        cout << "Improvements this generation: " << improvements << endl;
    }

    saveCSV(out, best_so_far);
    cout << "\nFinal score: " << global_best_score << endl;
    cout << "Saved to: " << out << endl;

    return 0;
}
