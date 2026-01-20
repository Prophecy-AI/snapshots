// SA with fractional translation - based on jonathanchan kernel
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

const int MAXN = 210;
const int V = 15;
double TX[V] = {0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125};
double TY[V] = {0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5};

mt19937 rng(42);
double rf() { return uniform_real_distribution<double>(0, 1)(rng); }

struct Cfg {
    int n;
    double x[MAXN], y[MAXN], a[MAXN];
    double px[MAXN][V], py[MAXN][V];
    
    void upd(int i) {
        double r = a[i] * M_PI / 180.0;
        double c = cos(r), s = sin(r);
        for (int j = 0; j < V; j++) {
            px[i][j] = c * TX[j] - s * TY[j] + x[i];
            py[i][j] = s * TX[j] + c * TY[j] + y[i];
        }
    }
    
    void updAll() { for (int i = 0; i < n; i++) upd(i); }
    
    double side() const {
        double mnx = 1e300, mny = 1e300, mxx = -1e300, mxy = -1e300;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < V; j++) {
                mnx = min(mnx, px[i][j]);
                mxx = max(mxx, px[i][j]);
                mny = min(mny, py[i][j]);
                mxy = max(mxy, py[i][j]);
            }
        }
        return max(mxx - mnx, mxy - mny);
    }
    
    double score() const { return side() * side() / n; }
    
    bool hasOvl(int idx) const {
        for (int i = 0; i < n; i++) {
            if (i == idx) continue;
            // Simple bounding box check first
            double mnx1 = 1e300, mxx1 = -1e300, mny1 = 1e300, mxy1 = -1e300;
            double mnx2 = 1e300, mxx2 = -1e300, mny2 = 1e300, mxy2 = -1e300;
            for (int j = 0; j < V; j++) {
                mnx1 = min(mnx1, px[idx][j]); mxx1 = max(mxx1, px[idx][j]);
                mny1 = min(mny1, py[idx][j]); mxy1 = max(mxy1, py[idx][j]);
                mnx2 = min(mnx2, px[i][j]); mxx2 = max(mxx2, px[i][j]);
                mny2 = min(mny2, py[i][j]); mxy2 = max(mxy2, py[i][j]);
            }
            if (mxx1 < mnx2 || mxx2 < mnx1 || mxy1 < mny2 || mxy2 < mny1) continue;
            
            // Detailed polygon intersection check using separating axis theorem
            // For simplicity, use a conservative overlap check
            for (int j1 = 0; j1 < V; j1++) {
                int j2 = (j1 + 1) % V;
                double ex = py[idx][j2] - py[idx][j1];
                double ey = -(px[idx][j2] - px[idx][j1]);
                double len = sqrt(ex*ex + ey*ey);
                if (len < 1e-12) continue;
                ex /= len; ey /= len;
                
                double mn1 = 1e300, mx1 = -1e300;
                for (int k = 0; k < V; k++) {
                    double d = px[idx][k] * ex + py[idx][k] * ey;
                    mn1 = min(mn1, d); mx1 = max(mx1, d);
                }
                double mn2 = 1e300, mx2 = -1e300;
                for (int k = 0; k < V; k++) {
                    double d = px[i][k] * ex + py[i][k] * ey;
                    mn2 = min(mn2, d); mx2 = max(mx2, d);
                }
                if (mx1 < mn2 + 1e-9 || mx2 < mn1 + 1e-9) goto no_overlap;
            }
            for (int j1 = 0; j1 < V; j1++) {
                int j2 = (j1 + 1) % V;
                double ex = py[i][j2] - py[i][j1];
                double ey = -(px[i][j2] - px[i][j1]);
                double len = sqrt(ex*ex + ey*ey);
                if (len < 1e-12) continue;
                ex /= len; ey /= len;
                
                double mn1 = 1e300, mx1 = -1e300;
                for (int k = 0; k < V; k++) {
                    double d = px[idx][k] * ex + py[idx][k] * ey;
                    mn1 = min(mn1, d); mx1 = max(mx1, d);
                }
                double mn2 = 1e300, mx2 = -1e300;
                for (int k = 0; k < V; k++) {
                    double d = px[i][k] * ex + py[i][k] * ey;
                    mn2 = min(mn2, d); mx2 = max(mx2, d);
                }
                if (mx1 < mn2 + 1e-9 || mx2 < mn1 + 1e-9) goto no_overlap;
            }
            return true;
            no_overlap:;
        }
        return false;
    }
};

Cfg sa_v3(Cfg c, int si, double t0, double tf, double pd, double ad, int seed) {
    mt19937 lrng(seed);
    auto lrf = [&]() { return uniform_real_distribution<double>(0, 1)(lrng); };
    
    Cfg best = c;
    double bs = best.side();
    double t = t0;
    double alpha = pow(tf / t0, 1.0 / si);
    
    for (int i = 0; i < si; i++) {
        int idx = lrng() % c.n;
        double ox = c.x[idx], oy = c.y[idx], oa = c.a[idx];
        
        c.x[idx] += (lrf() * 2 - 1) * pd * t;
        c.y[idx] += (lrf() * 2 - 1) * pd * t;
        c.a[idx] = fmod(c.a[idx] + (lrf() * 2 - 1) * ad * t + 360, 360.0);
        c.upd(idx);
        
        if (c.hasOvl(idx)) {
            c.x[idx] = ox; c.y[idx] = oy; c.a[idx] = oa;
            c.upd(idx);
        } else {
            double ns = c.side();
            double delta = ns - bs;
            if (delta < 0 || lrf() < exp(-delta / t)) {
                if (ns < bs) { bs = ns; best = c; }
            } else {
                c.x[idx] = ox; c.y[idx] = oy; c.a[idx] = oa;
                c.upd(idx);
            }
        }
        t *= alpha;
    }
    return best;
}

Cfg ls_v3(Cfg c, int mi) {
    Cfg best = c;
    double bs = best.side();
    double steps[] = {0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001};
    double dx[] = {0, 0, 1, -1, 1, 1, -1, -1};
    double dy[] = {1, -1, 0, 0, 1, -1, 1, -1};
    
    for (int iter = 0; iter < mi; iter++) {
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            for (double step : steps) {
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
            // Try rotation
            for (double da : {5.0, -5.0, 10.0, -10.0, 2.0, -2.0, 1.0, -1.0}) {
                double oa = best.a[i];
                best.a[i] = fmod(best.a[i] + da + 360, 360.0);
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

Cfg perturb(Cfg c, double scale, int seed) {
    mt19937 lrng(seed);
    auto lrf = [&]() { return uniform_real_distribution<double>(0, 1)(lrng); };
    for (int i = 0; i < c.n; i++) {
        c.x[i] += (lrf() * 2 - 1) * scale;
        c.y[i] += (lrf() * 2 - 1) * scale;
        c.a[i] = fmod(c.a[i] + (lrf() * 2 - 1) * 30 + 360, 360.0);
    }
    c.updAll();
    return c;
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
    string out = "submission_opt.csv";
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
    cout << "Input: " << in << ", Output: " << out << "\n";
    cout << "SA iterations: " << si << ", Restarts: " << nr << "\n";

    auto cfg = loadCSV(in);
    if (cfg.empty()) { cerr << "No data!\n"; return 1; }

    double start_score = 0;
    for (const auto& [n, c] : cfg) start_score += c.score();
    cout << fixed << setprecision(6);
    cout << "Starting score: " << start_score << "\n\n";

    map<int, Cfg> best_cfg = cfg;
    
    #pragma omp parallel for schedule(dynamic, 1)
    for (int n = 1; n <= 200; n++) {
        if (!cfg.count(n)) continue;
        
        Cfg c = cfg[n];
        double old_score = c.score();
        
        int it = si, r = nr;
        if (n <= 20) { r = max(6, nr); it = int(si * 1.5); }
        else if (n <= 50) { r = max(5, nr); it = int(si * 1.3); }
        else if (n > 150) { r = max(4, nr); it = int(si * 0.8); }
        
        Cfg opt = opt_v3(c, r, it);
        opt = fractional_translation(opt, 120);
        
        double new_score = opt.score();
        
        #pragma omp critical
        {
            if (new_score < old_score - 1e-9) {
                best_cfg[n] = opt;
                double improvement = (old_score - new_score) / old_score * 100.0;
                cout << "n=" << setw(3) << n << "  "
                     << old_score << " -> " << new_score
                     << "  (+" << fixed << setprecision(4) << improvement << "%)" << endl;
            }
        }
    }

    double final_score = 0;
    for (const auto& [n, c] : best_cfg) final_score += c.score();
    cout << "\nFinal score: " << final_score << "\n";
    cout << "Improvement: " << start_score - final_score << "\n";

    saveCSV(out, best_cfg);
    cout << "Saved to " << out << "\n";

    return 0;
}
