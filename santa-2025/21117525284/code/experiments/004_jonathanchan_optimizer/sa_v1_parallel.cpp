// SA V1 Parallel - from jonathanchan kernel
// Features: opt_v3 (multi-start SA), fractional_translation, ls_v3

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <tuple>
#include <iomanip>
#include <chrono>
#include <random>
#include <numeric>
#include <omp.h>
#include <thread>

using namespace std;

constexpr int MAX_N = 200;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;

const double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
const double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

thread_local mt19937_64 rng(42);
thread_local uniform_real_distribution<double> U(0, 1);

inline double rf() { return U(rng); }
inline int ri(int n) { return rng() % n; }

struct Poly {
    double px[NV], py[NV];
    double x0, y0, x1, y1;
    void bbox() {
        x0 = x1 = px[0]; y0 = y1 = py[0];
        for (int i = 1; i < NV; i++) {
            x0 = min(x0, px[i]); x1 = max(x1, px[i]);
            y0 = min(y0, py[i]); y1 = max(y1, py[i]);
        }
    }
};

Poly getPoly(double cx, double cy, double deg) {
    Poly q;
    double r = deg * PI / 180;
    double c = cos(r), s = sin(r);
    for (int i = 0; i < NV; i++) {
        q.px[i] = c * TX[i] - s * TY[i] + cx;
        q.py[i] = s * TX[i] + c * TY[i] + cy;
    }
    q.bbox();
    return q;
}

bool pip(double px, double py, const Poly& q) {
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

bool segInt(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy) {
    auto ccw = [](double px, double py, double qx, double qy, double rx, double ry) {
        return (ry - py) * (qx - px) > (qy - py) * (rx - px);
    };
    return ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy) &&
           ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy);
}

bool overlap(const Poly& a, const Poly& b) {
    if (a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0) return false;
    for (int i = 0; i < NV; i++) {
        if (pip(a.px[i], a.py[i], b)) return true;
        if (pip(b.px[i], b.py[i], a)) return true;
    }
    for (int i = 0; i < NV; i++)
        for (int j = 0; j < NV; j++)
            if (segInt(a.px[i], a.py[i], a.px[(i+1)%NV], a.py[(i+1)%NV],
                       b.px[j], b.py[j], b.px[(j+1)%NV], b.py[(j+1)%NV])) return true;
    return false;
}

struct Cfg {
    int n;
    double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];

    void upd(int i) { pl[i] = getPoly(x[i], y[i], a[i]); }
    void updAll() { for (int i = 0; i < n; i++) upd(i); }

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

    double side() const {
        if (!n) return 0;
        double x0 = pl[0].x0, x1 = pl[0].x1, y0 = pl[0].y0, y1 = pl[0].y1;
        for (int i = 1; i < n; i++) {
            x0 = min(x0, pl[i].x0); x1 = max(x1, pl[i].x1);
            y0 = min(y0, pl[i].y0); y1 = max(y1, pl[i].y1);
        }
        return max(x1 - x0, y1 - y0);
    }

    double score() const { double s = side(); return s * s / n; }
};

Cfg perturb(Cfg c, double strength, uint64_t seed) {
    rng.seed(seed);
    for (int i = 0; i < c.n; i++) {
        c.x[i] += (rf() - 0.5) * strength;
        c.y[i] += (rf() - 0.5) * strength;
        c.a[i] = fmod(c.a[i] + rf() * 30 - 15 + 360, 360.0);
    }
    c.updAll();
    return c;
}

Cfg sa_v3(Cfg c, int si, double T0, double Tmin, double alpha, double rot_range, uint64_t seed) {
    rng.seed(seed);
    Cfg best = c;
    double bs = best.side();
    double T = T0;
    
    for (int iter = 0; iter < si; iter++) {
        int i = ri(c.n);
        double ox = c.x[i], oy = c.y[i], oa = c.a[i];
        
        int move = ri(3);
        if (move == 0) {
            c.x[i] += (rf() - 0.5) * 0.1 * T / T0;
            c.y[i] += (rf() - 0.5) * 0.1 * T / T0;
        } else if (move == 1) {
            c.a[i] = fmod(c.a[i] + (rf() - 0.5) * rot_range * T / T0 + 360, 360.0);
        } else {
            int j = ri(c.n);
            if (i != j) {
                swap(c.x[i], c.x[j]);
                swap(c.y[i], c.y[j]);
                swap(c.a[i], c.a[j]);
                c.upd(j);
            }
        }
        c.upd(i);
        
        if (!c.hasOvl(i)) {
            double ns = c.side();
            double delta = ns - bs;
            if (delta < 0 || rf() < exp(-delta / T)) {
                if (ns < bs) {
                    bs = ns;
                    best = c;
                }
            } else {
                c.x[i] = ox; c.y[i] = oy; c.a[i] = oa;
                c.upd(i);
            }
        } else {
            c.x[i] = ox; c.y[i] = oy; c.a[i] = oa;
            c.upd(i);
        }
        
        T = max(Tmin, T * alpha);
    }
    return best;
}

Cfg ls_v3(Cfg c, int max_iter) {
    Cfg best = c;
    double bs = best.side();
    double steps[] = {0.01, 0.005, 0.002, 0.001};
    double angles[] = {5.0, 2.0, 1.0, 0.5};
    
    for (int iter = 0; iter < max_iter; iter++) {
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            for (double step : steps) {
                double dx[] = {step, -step, 0, 0};
                double dy[] = {0, 0, step, -step};
                for (int d = 0; d < 4; d++) {
                    double ox = best.x[i], oy = best.y[i];
                    best.x[i] += dx[d];
                    best.y[i] += dy[d];
                    best.upd(i);
                    if (!best.hasOvl(i)) {
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
            for (double ang : angles) {
                double oa = best.a[i];
                for (int d = -1; d <= 1; d += 2) {
                    best.a[i] = fmod(oa + d * ang + 360, 360.0);
                    best.upd(i);
                    if (!best.hasOvl(i)) {
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
    string out = "submission_out.csv";
    int si = 15000, nr = 5;

    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "-i" && i + 1 < argc) in = argv[++i];
        else if (a == "-o" && i + 1 < argc) out = argv[++i];
        else if (a == "-n" && i + 1 < argc) si = stoi(argv[++i]);
        else if (a == "-r" && i + 1 < argc) nr = stoi(argv[++i]);
    }

    int threads = omp_get_max_threads();
    omp_set_num_threads(threads);
    cout << "Using " << threads << " threads\n";

    auto cfg = loadCSV(in);
    if (cfg.empty()) { cerr << "No data!\n"; return 1; }

    double initial_score = 0;
    for (const auto& [n, c] : cfg) initial_score += c.score();
    cout << fixed << setprecision(6);
    cout << "Initial score: " << initial_score << "\n";

    map<int, Cfg> best_cfg = cfg;
    double best_total = initial_score;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int n = 1; n <= 200; n++) {
        if (!cfg.count(n)) continue;

        Cfg c = cfg[n];
        int it = si, r = nr;
        if (n <= 20) { r = max(6, nr); it = int(si * 1.5); }
        else if (n <= 50) { r = max(5, nr); it = int(si * 1.3); }
        else if (n > 150) { r = max(4, nr); it = int(si * 0.8); }

        Cfg candidate = opt_v3(c, r, it);
        candidate = fractional_translation(candidate, 120);

        #pragma omp critical
        {
            double old_score = cfg[n].score();
            double new_score = candidate.score();
            if (new_score < old_score - 1e-9) {
                best_cfg[n] = candidate;
                cout << "n=" << setw(3) << n << "  "
                     << old_score << " -> " << new_score
                     << "  (+" << fixed << setprecision(4) 
                     << (old_score - new_score) / old_score * 100.0 << "%)" << endl;
            }
        }
    }

    double final_score = 0;
    for (const auto& [n, c] : best_cfg) final_score += c.score();
    cout << "\nFinal score: " << final_score << "\n";
    cout << "Improvement: " << initial_score - final_score << "\n";

    saveCSV(out, best_cfg);
    cout << "Saved to " << out << "\n";

    return 0;
}
