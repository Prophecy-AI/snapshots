// Compile: g++ -O3 -march=native -std=c++17 -fopenmp -o sa_parallel sa_parallel.cpp
// Run: ./sa_parallel -i input.csv -o output.csv -n 15000 -r 5

#include <bits/stdc++.h>
using namespace std;
using namespace chrono;
#include <omp.h>

constexpr int MAX_N = 200;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;
const double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
const double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

mt19937_64 rng(42);
uniform_real_distribution<double> U(0, 1);
inline double rf() { return U(rng); }
inline int ri(int n) { return rng() % n; }

struct Pt { double x, y; };

struct Poly {
    Pt p[NV];
    double x0, y0, x1, y1;
    void bbox() {
        x0 = x1 = p[0].x; y0 = y1 = p[0].y;
        for (int i = 1; i < NV; i++) {
            x0 = min(x0, p[i].x); x1 = max(x1, p[i].x);
            y0 = min(y0, p[i].y); y1 = max(y1, p[i].y);
        }
    }
};

Poly getPoly(double cx, double cy, double deg) {
    Poly q;
    double r = deg * PI / 180, c = cos(r), s = sin(r);
    for (int i = 0; i < NV; i++) {
        q.p[i].x = TX[i] * c - TY[i] * s + cx;
        q.p[i].y = TX[i] * s + TY[i] * c + cy;
    }
    q.bbox();
    return q;
}

bool pip(double px, double py, const Poly& q) {
    bool in = false;
    int j = NV - 1;
    for (int i = 0; i < NV; i++) {
        if ((q.p[i].y > py) != (q.p[j].y > py) &&
            px < (q.p[j].x - q.p[i].x) * (py - q.p[i].y) / (q.p[j].y - q.p[i].y) + q.p[i].x)
            in = !in;
        j = i;
    }
    return in;
}

bool segInt(Pt a, Pt b, Pt c, Pt d) {
    auto ccw = [](Pt p, Pt q, Pt r) { return (r.y - p.y) * (q.x - p.x) > (q.y - p.y) * (r.x - p.x); };
    return ccw(a, c, d) != ccw(b, c, d) && ccw(a, b, c) != ccw(a, b, d);
}

bool overlap(const Poly& a, const Poly& b) {
    if (a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0) return false;
    for (int i = 0; i < NV; i++) {
        if (pip(a.p[i].x, a.p[i].y, b)) return true;
        if (pip(b.p[i].x, b.p[i].y, a)) return true;
    }
    for (int i = 0; i < NV; i++)
        for (int j = 0; j < NV; j++)
            if (segInt(a.p[i], a.p[(i + 1) % NV], b.p[j], b.p[(j + 1) % NV])) return true;
    return false;
}

struct Cfg {
    int n;
    double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];
    void upd(int i) { pl[i] = getPoly(x[i], y[i], a[i]); }
    void updAll() { for (int i = 0; i < n; i++) upd(i); }
    bool hasOvl(int i) const {
        for (int j = 0; j < n; j++) if (i != j && overlap(pl[i], pl[j])) return true;
        return false;
    }
    bool hasOvlPair(int i, int j) const {
        if (overlap(pl[i], pl[j])) return true;
        return false;
    }
    bool valid() const {
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (overlap(pl[i], pl[j])) return false;
        return true;
    }
    double side() const {
        double mnx = 1e300, mny = 1e300, mxx = -1e300, mxy = -1e300;
        for (int i = 0; i < n; i++) {
            mnx = min(mnx, pl[i].x0); mxx = max(mxx, pl[i].x1);
            mny = min(mny, pl[i].y0); mxy = max(mxy, pl[i].y1);
        }
        return max(mxx - mnx, mxy - mny);
    }
    double score() const { double s = side(); return s * s / n; }
};

Cfg sa_v3(Cfg c, int si, double T0, double Tf, double alpha, double beta, int seed) {
    mt19937_64 rng(seed);
    uniform_real_distribution<double> U(0, 1);
    auto rf = [&]() { return U(rng); };
    auto ri = [&](int n) { return rng() % n; };

    double T = T0;
    double bs = c.side();
    Cfg best = c;

    for (int it = 0; it < si; it++) {
        int i = ri(c.n);
        double ox = c.x[i], oy = c.y[i], oa = c.a[i];
        double dx = (rf() - 0.5) * alpha * T;
        double dy = (rf() - 0.5) * alpha * T;
        double da = (rf() - 0.5) * beta * T;
        c.x[i] += dx; c.y[i] += dy; c.a[i] += da;
        c.upd(i);

        bool ok = !c.hasOvl(i);
        double ns = ok ? c.side() : 1e300;

        if (ok && (ns < bs || rf() < exp((bs - ns) / T))) {
            if (ns < bs) { bs = ns; best = c; }
        } else {
            c.x[i] = ox; c.y[i] = oy; c.a[i] = oa;
            c.upd(i);
        }

        T *= (1.0 - 1.0 / si);
        if (T < Tf) T = Tf;
    }
    return best;
}

Cfg ls_v3(Cfg c, int iters) {
    double bs = c.side();
    double steps[] = {0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001};
    double asteps[] = {10, 5, 2, 1, 0.5, 0.2, 0.1};
    int dirs[][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{-1,1},{1,-1},{-1,-1}};

    for (int it = 0; it < iters; it++) {
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            for (double step : steps) {
                for (auto& d : dirs) {
                    double ox = c.x[i], oy = c.y[i];
                    c.x[i] += d[0] * step;
                    c.y[i] += d[1] * step;
                    c.upd(i);
                    if (!c.hasOvl(i) && c.side() < bs - 1e-9) {
                        bs = c.side();
                        improved = true;
                    } else {
                        c.x[i] = ox; c.y[i] = oy;
                        c.upd(i);
                    }
                }
            }
            for (double astep : asteps) {
                for (int dir = -1; dir <= 1; dir += 2) {
                    double oa = c.a[i];
                    c.a[i] += dir * astep;
                    c.upd(i);
                    if (!c.hasOvl(i) && c.side() < bs - 1e-9) {
                        bs = c.side();
                        improved = true;
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

Cfg fractional_translation(Cfg c, int iters) {
    double bs = c.side();
    double steps[] = {0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001};
    int dirs[][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{-1,1},{1,-1},{-1,-1}};

    for (int it = 0; it < iters; it++) {
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            for (double step : steps) {
                for (auto& d : dirs) {
                    double ox = c.x[i], oy = c.y[i];
                    c.x[i] += d[0] * step;
                    c.y[i] += d[1] * step;
                    c.upd(i);
                    if (!c.hasOvl(i) && c.side() < bs - 1e-9) {
                        bs = c.side();
                        improved = true;
                    } else {
                        c.x[i] = ox; c.y[i] = oy;
                        c.upd(i);
                    }
                }
            }
        }
        if (!improved) break;
    }
    return c;
}

Cfg perturb(Cfg c, double mag, int seed) {
    mt19937_64 rng(seed);
    uniform_real_distribution<double> U(-mag, mag);
    for (int i = 0; i < c.n; i++) {
        c.x[i] += U(rng);
        c.y[i] += U(rng);
        c.a[i] += U(rng) * 30;
    }
    c.updAll();
    return c;
}

Cfg opt_v3(Cfg c, int nr, int si) {
    double bs = c.side();
    Cfg best = c;

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
    string in = "./submission.csv";
    string out = "./submission_out.csv";
    int si = 20000, nr = 5;

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

    map<int, Cfg> best_so_far = cfg;
    double global_best_score = 0;
    for (const auto& [n, c] : best_so_far) global_best_score += c.score();

    cout << fixed << setprecision(6);
    cout << "Starting score: " << global_best_score << "\n\n";

    // Single generation optimization
    map<int, Cfg> current = best_so_far;

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
            double old_n_score = current[n].score();
            double new_n_score = candidate.score();

            if (new_n_score < old_n_score - 1e-9) {
                current[n] = candidate;
                double improvement = (old_n_score - new_n_score) / old_n_score * 100.0;
                cout << "n=" << setw(3) << n << "  "
                     << old_n_score << " -> " << new_n_score
                     << "  (+" << fixed << setprecision(4) << improvement << "%)" << endl;
            }
        }
    }

    double new_total = 0;
    for (const auto& [n, c] : current) new_total += c.score();

    cout << "\nFinal score: " << new_total << endl;
    cout << "Improvement: " << (global_best_score - new_total) << endl;

    saveCSV(out, current);
    cout << "Saved to " << out << endl;

    return 0;
}
