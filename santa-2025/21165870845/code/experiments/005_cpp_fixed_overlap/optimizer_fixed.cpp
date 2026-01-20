// Santa 2025 C++ Optimizer - FIXED OVERLAP DETECTION
// Based on jonathanchan kernel: SA + Local Search + Fractional Translation
// FIXED: Added point-in-polygon check to detect polygon containment

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

const int NV = 15;
const double TX[NV] = {0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125};
const double TY[NV] = {0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5};

struct Cfg {
    int n = 0;
    double x[200], y[200], a[200];
    double px[200][NV], py[200][NV];
    double mnx[200], mxx[200], mny[200], mxy[200];

    void upd(int i) {
        double r = a[i] * M_PI / 180.0;
        double c = cos(r), s = sin(r);
        mnx[i] = 1e9; mxx[i] = -1e9; mny[i] = 1e9; mxy[i] = -1e9;
        for (int j = 0; j < NV; j++) {
            px[i][j] = c * TX[j] - s * TY[j] + x[i];
            py[i][j] = s * TX[j] + c * TY[j] + y[i];
            mnx[i] = min(mnx[i], px[i][j]);
            mxx[i] = max(mxx[i], px[i][j]);
            mny[i] = min(mny[i], py[i][j]);
            mxy[i] = max(mxy[i], py[i][j]);
        }
    }

    void updAll() { for (int i = 0; i < n; i++) upd(i); }

    double side() const {
        double mnX = 1e9, mxX = -1e9, mnY = 1e9, mxY = -1e9;
        for (int i = 0; i < n; i++) {
            mnX = min(mnX, mnx[i]); mxX = max(mxX, mxx[i]);
            mnY = min(mnY, mny[i]); mxY = max(mxY, mxy[i]);
        }
        return max(mxX - mnX, mxY - mnY);
    }

    double score() const { double s = side(); return s * s / n; }

    // Point-in-polygon test using ray casting algorithm
    bool pointInPolygon(double testx, double testy, int polyIdx) const {
        int cnt = 0;
        for (int k = 0; k < NV; k++) {
            int k2 = (k + 1) % NV;
            double x1 = px[polyIdx][k], y1 = py[polyIdx][k];
            double x2 = px[polyIdx][k2], y2 = py[polyIdx][k2];
            
            // Ray casting: count intersections with horizontal ray to the right
            if ((y1 <= testy && testy < y2) || (y2 <= testy && testy < y1)) {
                double xint = x1 + (testy - y1) / (y2 - y1) * (x2 - x1);
                if (testx < xint) cnt++;
            }
        }
        return cnt % 2 == 1;
    }

    // Check if two line segments intersect (proper intersection, not just touching)
    bool segmentsIntersect(double x1, double y1, double x2, double y2,
                           double x3, double y3, double x4, double y4) const {
        double d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
        if (fabs(d) < 1e-12) return false;
        double t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d;
        double u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / d;
        // Use strict inequality to avoid counting edge touches as overlaps
        return t > 1e-9 && t < 1 - 1e-9 && u > 1e-9 && u < 1 - 1e-9;
    }

    bool hasOvl(int i) const {
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            
            // Quick bounding box check
            if (mxx[i] < mnx[j] - 1e-9 || mnx[i] > mxx[j] + 1e-9 || 
                mxy[i] < mny[j] - 1e-9 || mny[i] > mxy[j] + 1e-9) continue;
            
            // Check edge-edge intersections
            for (int a = 0; a < NV; a++) {
                int a2 = (a + 1) % NV;
                for (int b = 0; b < NV; b++) {
                    int b2 = (b + 1) % NV;
                    if (segmentsIntersect(px[i][a], py[i][a], px[i][a2], py[i][a2],
                                          px[j][b], py[j][b], px[j][b2], py[j][b2])) {
                        return true;
                    }
                }
            }
            
            // Check if any vertex of polygon i is inside polygon j
            for (int v = 0; v < NV; v++) {
                if (pointInPolygon(px[i][v], py[i][v], j)) {
                    return true;
                }
            }
            
            // Check if any vertex of polygon j is inside polygon i
            for (int v = 0; v < NV; v++) {
                if (pointInPolygon(px[j][v], py[j][v], i)) {
                    return true;
                }
            }
        }
        return false;
    }
};

mt19937 rng(42);
double rf() { return uniform_real_distribution<double>(0, 1)(rng); }

Cfg sa_v3(Cfg c, int steps, double T0, double Tf, double pd, double ad, int seed) {
    rng.seed(seed);
    Cfg best = c;
    double bs = best.side();
    double T = T0;
    double alpha = pow(Tf / T0, 1.0 / steps);
    
    for (int step = 0; step < steps; step++) {
        int i = rng() % c.n;
        double ox = c.x[i], oy = c.y[i], oa = c.a[i];
        int mv = rng() % 3;
        if (mv == 0) c.x[i] += (rf() * 2 - 1) * pd;
        else if (mv == 1) c.y[i] += (rf() * 2 - 1) * pd;
        else c.a[i] = fmod(c.a[i] + (rf() * 2 - 1) * ad + 360, 360.0);
        c.upd(i);
        
        if (c.hasOvl(i)) {
            c.x[i] = ox; c.y[i] = oy; c.a[i] = oa; c.upd(i);
            continue;
        }
        
        double ns = c.side();
        double delta = ns - bs;
        if (delta < 0 || rf() < exp(-delta / T)) {
            if (ns < bs) { bs = ns; best = c; }
        } else {
            c.x[i] = ox; c.y[i] = oy; c.a[i] = oa; c.upd(i);
        }
        T *= alpha;
    }
    return best;
}

Cfg ls_v3(Cfg c, int max_iter) {
    Cfg best = c;
    double bs = best.side();
    double steps[] = {0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001};
    
    for (int iter = 0; iter < max_iter; iter++) {
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            for (double step : steps) {
                double dx[] = {0, 0, 1, -1, 1, 1, -1, -1};
                double dy[] = {1, -1, 0, 0, 1, -1, 1, -1};
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
    
    for (int r = 0; r < nr; r++) {
        Cfg start = c;
        if (r > 0) {
            // Perturb starting configuration
            for (int i = 0; i < start.n; i++) {
                double ox = start.x[i], oy = start.y[i], oa = start.a[i];
                start.x[i] += (rf() * 2 - 1) * 0.1;
                start.y[i] += (rf() * 2 - 1) * 0.1;
                start.a[i] = fmod(start.a[i] + (rf() * 2 - 1) * 10 + 360, 360.0);
                start.upd(i);
                // Revert if causes overlap
                if (start.hasOvl(i)) {
                    start.x[i] = ox; start.y[i] = oy; start.a[i] = oa;
                    start.upd(i);
                }
            }
        }
        
        Cfg o = sa_v3(start, si, 1.0, 0.000005, 0.25, 70.0, 42 + r * 1000 + c.n);
        o = ls_v3(o, 300);
        o = fractional_translation(o, 150);
        
        double s = o.side();
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
    string out = "submission_optimized.csv";
    int si = 5000, nr = 5;

    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "-i" && i + 1 < argc) in = argv[++i];
        else if (a == "-o" && i + 1 < argc) out = argv[++i];
        else if (a == "-n" && i + 1 < argc) si = stoi(argv[++i]);
        else if (a == "-r" && i + 1 < argc) nr = stoi(argv[++i]);
    }

    cout << "Loading " << in << "..." << endl;
    auto cfg = loadCSV(in);
    if (cfg.empty()) { cerr << "No data!\n"; return 1; }

    double start_score = 0;
    for (const auto& [n, c] : cfg) start_score += c.score();
    cout << fixed << setprecision(6);
    cout << "Starting score: " << start_score << endl;

    map<int, Cfg> best_cfg = cfg;
    double best_total = start_score;

    // Optimize each N
    for (int n = 1; n <= 200; n++) {
        if (!cfg.count(n)) continue;
        
        Cfg c = cfg[n];
        double old_score = c.score();
        
        // Adjust iterations based on N
        int it = si, r = nr;
        if (n <= 20) { r = max(3, nr); it = int(si * 1.5); }
        else if (n <= 50) { r = max(2, nr); it = int(si * 1.2); }
        
        Cfg optimized = opt_v3(c, r, it);
        optimized = fractional_translation(optimized, 100);
        
        double new_score = optimized.score();
        
        // Only accept if no overlaps and better score
        bool has_overlap = false;
        for (int i = 0; i < optimized.n && !has_overlap; i++) {
            if (optimized.hasOvl(i)) has_overlap = true;
        }
        
        if (!has_overlap && new_score < old_score - 1e-9) {
            best_cfg[n] = optimized;
            cout << "n=" << setw(3) << n << ": " << old_score << " -> " << new_score 
                 << " (improved by " << (old_score - new_score) << ")" << endl;
        }
    }

    double final_score = 0;
    for (const auto& [n, c] : best_cfg) final_score += c.score();
    
    cout << "\nFinal score: " << final_score << endl;
    cout << "Improvement: " << (start_score - final_score) << endl;

    saveCSV(out, best_cfg);
    cout << "Saved to " << out << endl;

    return 0;
}
