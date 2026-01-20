// Compile: g++ -O3 -march=native -std=c++17 -fopenmp -o sa_v1_parallel sa_v1_parallel.cpp
// Run: ./sa_v1_parallel -i /kaggle/working/submission_ensemble.csv -o /kaggle/working/submission.csv -n 15000 -r 5

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
        for (int k = 0; k < n; k++) {
            if (k != i && k != j) {
                if (overlap(pl[i], pl[k]) || overlap(pl[j], pl[k])) return true;
            }
        }
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
    pair<double, double> centroid() const {
        double sx = 0, sy = 0;
        for (int i = 0; i < n; i++) { sx += x[i]; sy += y[i]; }
        return {sx / n, sy / n};
    }
    tuple<double, double, double, double> getBBox() const {
        double gx0 = pl[0].x0, gx1 = pl[0].x1, gy0 = pl[0].y0, gy1 = pl[0].y1;
        for (int i = 1; i < n; i++) {
            gx0 = min(gx0, pl[i].x0); gx1 = max(gx1, pl[i].x1);
            gy0 = min(gy0, pl[i].y0); gy1 = max(gy1, pl[i].y1);
        }
        return {gx0, gy0, gx1, gy1};
    }
    vector<int> findCornerTrees() const {
        auto [gx0, gy0, gx1, gy1] = getBBox();
        double eps = 0.01;
        vector<int> corners;
        for (int i = 0; i < n; i++) {
            if (abs(pl[i].x0 - gx0) < eps || abs(pl[i].x1 - gx1) < eps ||
                abs(pl[i].y0 - gy0) < eps || abs(pl[i].y1 - gy1) < eps) {
                corners.push_back(i);
            }
        }
        return corners;
    }
};

Cfg sa_v3(Cfg c, int iter, double T0, double Tm, double ms, double rs, uint64_t seed) {
    rng.seed(seed);
    Cfg best = c, cur = c;
    double bs = best.side(), cs = bs, T = T0;
    double alpha = pow(Tm / T0, 1.0 / iter);
    int noImp = 0;
    for (int it = 0; it < iter; it++) {
        int moveType = ri(8);
        double sc = T / T0;
        if (moveType < 4) {
            int i = ri(c.n);
            double ox = cur.x[i], oy = cur.y[i], oa = cur.a[i];
            auto [cx, cy] = cur.centroid();
            if (moveType == 0) {
                cur.x[i] += (rf() - 0.5) * 2 * ms * sc;
                cur.y[i] += (rf() - 0.5) * 2 * ms * sc;
            } else if (moveType == 1) {
                double dx = cx - cur.x[i], dy = cy - cur.y[i];
                double d = sqrt(dx * dx + dy * dy);
                if (d > 1e-6) {
                    double st = rf() * ms * sc;
                    cur.x[i] += dx / d * st;
                    cur.y[i] += dy / d * st;
                }
            } else if (moveType == 2) {
                cur.a[i] += (rf() - 0.5) * 2 * rs * sc;
                cur.a[i] = fmod(cur.a[i] + 360, 360.0);
            } else {
                cur.x[i] += (rf() - 0.5) * ms * sc;
                cur.y[i] += (rf() - 0.5) * ms * sc;
                cur.a[i] += (rf() - 0.5) * rs * sc;
                cur.a[i] = fmod(cur.a[i] + 360, 360.0);
            }
            cur.upd(i);
            if (cur.hasOvl(i)) {
                cur.x[i] = ox; cur.y[i] = oy; cur.a[i] = oa;
                cur.upd(i);
                noImp++;
                T *= alpha; if (T < Tm) T = Tm;
                continue;
            }
        } else if (moveType == 4 && c.n > 1) {
            int i = ri(c.n), j = ri(c.n);
            while (j == i) j = ri(c.n);
            double oxi = cur.x[i], oyi = cur.y[i];
            double oxj = cur.x[j], oyj = cur.y[j];
            cur.x[i] = oxj; cur.y[i] = oyj;
            cur.x[j] = oxi; cur.y[j] = oyi;
            cur.upd(i); cur.upd(j);
            if (cur.hasOvlPair(i, j)) {
                cur.x[i] = oxi; cur.y[i] = oyi;
                cur.x[j] = oxj; cur.y[j] = oyj;
                cur.upd(i); cur.upd(j);
                noImp++;
                T *= alpha; if (T < Tm) T = Tm;
                continue;
            }
        } else if (moveType == 5) {
            int i = ri(c.n);
            double ox = cur.x[i], oy = cur.y[i];
            auto [gx0, gy0, gx1, gy1] = cur.getBBox();
            double bcx = (gx0 + gx1) / 2, bcy = (gy0 + gy1) / 2;
            double dx = bcx - cur.x[i], dy = bcy - cur.y[i];
            double d = sqrt(dx * dx + dy * dy);
            if (d > 1e-6) {
                double st = rf() * ms * sc * 0.5;
                cur.x[i] += dx / d * st;
                cur.y[i] += dy / d * st;
            }
            cur.upd(i);
            if (cur.hasOvl(i)) {
                cur.x[i] = ox; cur.y[i] = oy;
                cur.upd(i);
                noImp++;
                T *= alpha; if (T < Tm) T = Tm;
                continue;
            }
        } else if (moveType == 6) {
            auto corners = cur.findCornerTrees();
            if (!corners.empty()) {
                int idx = corners[ri(corners.size())];
                double ox = cur.x[idx], oy = cur.y[idx], oa = cur.a[idx];
                auto [gx0, gy0, gx1, gy1] = cur.getBBox();
                double bcx = (gx0 + gx1) / 2, bcy = (gy0 + gy1) / 2;
                double dx = bcx - cur.x[idx], dy = bcy - cur.y[idx];
                double d = sqrt(dx * dx + dy * dy);
                if (d > 1e-6) {
                double st = rf() * ms * sc * 0.3;
                    cur.x[idx] += dx / d * st;
                    cur.y[idx] += dy / d * st;
                    cur.a[idx] += (rf() - 0.5) * rs * sc * 0.5;
                    cur.a[idx] = fmod(cur.a[idx] + 360, 360.0);
                }
                cur.upd(idx);
                if (cur.hasOvl(idx)) {
                    cur.x[idx] = ox; cur.y[idx] = oy; cur.a[idx] = oa;
                    cur.upd(idx);
                    noImp++;
                    T *= alpha; if (T < Tm) T = Tm;
                    continue;
                }
            } else {
                noImp++;
                T *= alpha; if (T < Tm) T = Tm;
                continue;
            }
        } else {
            int i = ri(c.n);
            int j = (i + 1) % c.n;
            double oxi = cur.x[i], oyi = cur.y[i];
            double oxj = cur.x[j], oyj = cur.y[j];
            double dx = (rf() - 0.5) * ms * sc * 0.5;
            double dy = (rf() - 0.5) * ms * sc * 0.5;
            cur.x[i] += dx; cur.y[i] += dy;
            cur.x[j] += dx; cur.y[j] += dy;
            cur.upd(i); cur.upd(j);
            if (cur.hasOvlPair(i, j)) {
                cur.x[i] = oxi; cur.y[i] = oyi;
                cur.x[j] = oxj; cur.y[j] = oyj;
                cur.upd(i); cur.upd(j);
                noImp++;
                T *= alpha; if (T < Tm) T = Tm;
                continue;
            }
        }
        double ns = cur.side();
        double delta = ns - cs;
        if (delta < 0 || rf() < exp(-delta / T)) {
            cs = ns;
            if (ns < bs) {
                bs = ns;
                best = cur;
                noImp = 0;
            } else {
                noImp++;
            }
        } else {
            cur = best;
            cs = bs;
            noImp++;
        }
        if (noImp > 600) {
            T = min(T * 3.0, T0 * 0.7);
            noImp = 0;
        }
        T *= alpha;
        if (T < Tm) T = Tm;
    }
    return best;
}

Cfg ls_v3(Cfg c, int iter) {
    Cfg best = c;
    double bs = best.side();
    double ps[] = {0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002};
    double rs[] = {15.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25};
    int dx[] = {1, -1, 0, 0, 1, 1, -1, -1};
    int dy[] = {0, 0, 1, -1, 1, -1, 1, -1};
    for (int it = 0; it < iter; it++) {
        bool imp = false;
        auto corners = best.findCornerTrees();
        for (int ci : corners) {
            for (double st : ps) {
                for (int d = 0; d < 8; d++) {
                    double ox = best.x[ci], oy = best.y[ci];
                    best.x[ci] += dx[d] * st;
                    best.y[ci] += dy[d] * st;
                    best.upd(ci);
                    if (!best.hasOvl(ci)) {
                        double ns = best.side();
                        if (ns < bs - 1e-10) {
                            bs = ns;
                            imp = true;
                        } else {
                            best.x[ci] = ox; best.y[ci] = oy;
                            best.upd(ci);
                        }
                    } else {
                        best.x[ci] = ox; best.y[ci] = oy;
                        best.upd(ci);
                    }
                }
            }
            for (double st : rs) {
                for (double da : {st, -st}) {
                    double oa = best.a[ci];
                    best.a[ci] = fmod(best.a[ci] + da + 360, 360.0);
                    best.upd(ci);
                    if (!best.hasOvl(ci)) {
                        double ns = best.side();
                        if (ns < bs - 1e-10) {
                            bs = ns;
                            imp = true;
                        } else {
                            best.a[ci] = oa;
                            best.upd(ci);
                        }
                    } else {
                        best.a[ci] = oa;
                        best.upd(ci);
                    }
                }
            }
        }
        set<int> cornerSet(corners.begin(), corners.end());
        for (int i = 0; i < c.n; i++) {
            if (cornerSet.count(i)) continue;
            for (double st : ps) {
                for (int d = 0; d < 8; d++) {
                    double ox = best.x[i], oy = best.y[i];
                    best.x[i] += dx[d] * st;
                    best.y[i] += dy[d] * st;
                    best.upd(i);
                    if (!best.hasOvl(i)) {
                        double ns = best.side();
                        if (ns < bs - 1e-10) {
                            bs = ns;
                            imp = true;
                        } else {
                            best.x[i] = ox; best.y[i] = oy;
                            best.upd(i);
                        }
                    } else {
                        best.x[i] = ox; best.y[i] = oy;
                        best.upd(i);
                    }
                }
            }
            for (double st : rs) {
                for (double da : {st, -st}) {
                    double oa = best.a[i];
                    best.a[i] = fmod(best.a[i] + da + 360, 360.0);
                    best.upd(i);
                    if (!best.hasOvl(i)) {
                        double ns = best.side();
                        if (ns < bs - 1e-10) {
                            bs = ns;
                            imp = true;
                        } else {
                            best.a[i] = oa;
                            best.upd(i);
                        }
                    } else {
                        best.a[i] = oa;
                        best.upd(i);
                    }
                }
            }
        }
        if (!imp) break;
    }
    return best;
}

Cfg perturb(Cfg c, double strength, uint64_t seed) {
    rng.seed(seed);
    int numPerturb = max(1, (int)(c.n * 0.15));
    for (int k = 0; k < numPerturb; k++) {
        int i = ri(c.n);
        c.x[i] += (rf() - 0.5) * strength;
        c.y[i] += (rf() - 0.5) * strength;
        c.a[i] = fmod(c.a[i] + (rf() - 0.5) * 60 + 360, 360.0);
    }
    c.updAll();
    for (int iter = 0; iter < 100; iter++) {
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

// ← put this somewhere near the top (after includes)
void ensure_dir() {
    #ifdef _WIN32
        system("if not exist solutions mkdir solutions");
    #else
        system("mkdir -p solutions");
    #endif
}

// ← REPLACE YOUR WHOLE main() WITH THIS ONE
int main(int argc, char** argv) {
    ensure_dir();

    string in = "./submission_best.csv";
    int si = 20000, nr = 80;

    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "-i" && i + 1 < argc) in = argv[++i];
        else if (a == "-n" && i + 1 < argc) si = stoi(argv[++i]);
        else if (a == "-r" && i + 1 < argc) nr = stoi(argv[++i]);
    }

    int threads = omp_get_max_threads();
    omp_set_num_threads(threads);
    cout << "Using " << threads << " threads — endless mode ON\n";

    auto cfg = loadCSV(in);
    if (cfg.empty()) { cerr << "No data!\n"; return 1; }

    map<int, Cfg> best_so_far = cfg;
    double global_best_score = 0;
    for (const auto& [n, c] : best_so_far) global_best_score += c.score();

    cout << fixed << setprecision(6);
    cout << "Starting score: " << global_best_score << "\n\n";

    int generation = 0;
    int no_improvement_count = 0;
    int max_retries = 3; // KEEP High
    int max_retry_retries = 3; // KEEP High
    int retry_count = 0;
    while (true) {
        generation++;
        cout << "\n=== Generation " << generation << " ===" << endl;

        map<int, Cfg> current = best_so_far;        // start from current best every round
        double round_start_score = global_best_score;

        {
            map<int, Cfg> local;

            #pragma omp for schedule(dynamic, 1) nowait
            for (int n = 1; n <= 200; n++) {
                if (!current.count(n)) continue;

                Cfg c = current[n];

                int it = si, r = nr;
                if (n <= 20) { r = max(6, nr); it = int(si * 1.5); }
                else if (n <= 50) { r = max(5, nr); it = int(si * 1.3); }
                else if (n > 150) { r = max(4, nr); it = int(si * 0.8); }

                Cfg candidate = opt_v3(c, r, it);
                candidate = fractional_translation(candidate, 120);

                local[n] = candidate;
            }

            {
                for (auto& p : local) {
                    int n = p.first;
                    Cfg& cand = p.second;
                    double old_n_score = current[n].score();
                    double new_n_score = cand.score();

                    if (new_n_score < old_n_score - 1e-9) {
                        current[n] = cand;
                        double improvement = (old_n_score - new_n_score) / old_n_score * 100.0;
                        cout << "n=" << setw(3) << n << "  "
                             << old_n_score << " → " << new_n_score
                             << "  (+" << fixed << setprecision(4) << improvement << "%)" << endl;
                    }
                }
            }
        }

        double new_total = 0;
        for (const auto& [n, c] : current) new_total += c.score();

        bool improved = (new_total < global_best_score - 1e-8);

        if (improved) {
            global_best_score = new_total;
            best_so_far = current;

            char filename[64];
            snprintf(filename, sizeof(filename), "solutions/submission_%.6f.csv", global_best_score);

            saveCSV(filename, best_so_far);

            cout << "\nNEW GLOBAL BEST! → " << global_best_score
                 << "   saved as  " << filename << endl;
            no_improvement_count = 0;
        } else {
            cout << "Generation " << generation << " finished — no global improvement ("
                 << new_total << ")" << endl;
            no_improvement_count += 1;
        }
        retry_count += 1;
        if (no_improvement_count > max_retries) {
            break;
        }

        if (retry_count > max_retry_retries) {
            break;
        }
        
        this_thread::sleep_for(chrono::milliseconds(100));
    }

    return 0;
}