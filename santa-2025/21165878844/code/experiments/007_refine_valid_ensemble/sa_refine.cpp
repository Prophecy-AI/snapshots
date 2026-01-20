// Gap-Constrained SA Optimizer - Refine Valid Ensemble
// Compile: g++ -O3 -march=native -std=c++17 -fopenmp -o sa_refine sa_refine.cpp
// Run: ./sa_refine -i valid_ensemble.csv -n 5000 -r 3

#include <bits/stdc++.h>
using namespace std;
#include <omp.h>

constexpr int MAX_N = 200;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;
constexpr double MIN_GAP = 1e-9;  // Very small gap - just enough to be valid

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

// Point to segment distance
double pointToSegDist(Pt p, Pt a, Pt b) {
    double dx = b.x - a.x, dy = b.y - a.y;
    double len2 = dx*dx + dy*dy;
    if (len2 < 1e-12) return sqrt((p.x-a.x)*(p.x-a.x) + (p.y-a.y)*(p.y-a.y));
    
    double t = max(0.0, min(1.0, ((p.x-a.x)*dx + (p.y-a.y)*dy) / len2));
    double projx = a.x + t * dx;
    double projy = a.y + t * dy;
    return sqrt((p.x-projx)*(p.x-projx) + (p.y-projy)*(p.y-projy));
}

// Minimum distance between two polygons
double minPolyDist(const Poly& a, const Poly& b) {
    double minDist = 1e9;
    
    for (int i = 0; i < NV; i++) {
        for (int j = 0; j < NV; j++) {
            double d = pointToSegDist(a.p[i], b.p[j], b.p[(j+1)%NV]);
            minDist = min(minDist, d);
            d = pointToSegDist(b.p[j], a.p[i], a.p[(i+1)%NV]);
            minDist = min(minDist, d);
        }
    }
    
    return minDist;
}

// Check if polygons overlap (standard check)
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

// GAP-CONSTRAINED overlap check
bool hasGapViolation(const Poly& a, const Poly& b) {
    // Quick bounding box check
    if (a.x1 + MIN_GAP < b.x0 || b.x1 + MIN_GAP < a.x0 || 
        a.y1 + MIN_GAP < b.y0 || b.y1 + MIN_GAP < a.y0) return false;
    
    if (overlap(a, b)) return true;
    
    double dist = minPolyDist(a, b);
    return dist < MIN_GAP;
}

struct Cfg {
    int n;
    double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];
    
    void upd(int i) { pl[i] = getPoly(x[i], y[i], a[i]); }
    void updAll() { for (int i = 0; i < n; i++) upd(i); }
    
    bool hasOvl(int i) const {
        for (int j = 0; j < n; j++) 
            if (i != j && hasGapViolation(pl[i], pl[j])) return true;
        return false;
    }
    
    bool anyOvl() const {
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (hasGapViolation(pl[i], pl[j])) return true;
        return false;
    }
    
    double minDist() const {
        double md = 1e9;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                md = min(md, minPolyDist(pl[i], pl[j]));
        return md;
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

// SA with gap constraints
Cfg sa_gap(Cfg c, int iter, double T0, double Tm, double ms, double rs, uint64_t seed) {
    rng.seed(seed);
    Cfg best = c, cur = c;
    double bs = best.side(), cs = bs, T = T0;
    double alpha = pow(Tm / T0, 1.0 / iter);
    
    for (int it = 0; it < iter; it++) {
        int i = ri(c.n);
        double ox = cur.x[i], oy = cur.y[i], oa = cur.a[i];
        double sc = T / T0;
        
        int moveType = ri(3);
        if (moveType == 0) {
            cur.x[i] += (rf() - 0.5) * 2 * ms * sc;
            cur.y[i] += (rf() - 0.5) * 2 * ms * sc;
        } else if (moveType == 1) {
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
            T *= alpha;
            continue;
        }
        
        double ns = cur.side();
        double delta = ns - cs;
        
        if (delta < 0 || rf() < exp(-delta / T)) {
            cs = ns;
            if (ns < bs) {
                bs = ns;
                best = cur;
            }
        } else {
            cur = best;
            cs = bs;
        }
        
        T *= alpha;
    }
    
    return best;
}

// Local search with fractional translation
Cfg fractional_translation(Cfg c, int max_iter = 100) {
    Cfg best = c;
    double bs = best.side();
    double frac_steps[] = {0.001, 0.0005, 0.0002, 0.0001, 0.00005};
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
    string input = "valid_ensemble.csv";
    int iterations = 5000;
    int rounds = 3;
    
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "-i" && i + 1 < argc) input = argv[++i];
        else if (a == "-n" && i + 1 < argc) iterations = stoi(argv[++i]);
        else if (a == "-r" && i + 1 < argc) rounds = stoi(argv[++i]);
    }
    
    cout << "Gap-Constrained SA - Refine Valid Ensemble" << endl;
    cout << "MIN_GAP = " << MIN_GAP << endl;
    cout << "Input: " << input << endl;
    cout << "Iterations = " << iterations << ", Rounds = " << rounds << endl;
    
    auto cfg = loadCSV(input);
    if (cfg.empty()) {
        cerr << "No data loaded!" << endl;
        return 1;
    }
    
    cout << "Loaded " << cfg.size() << " configurations" << endl;
    
    // Calculate initial score
    double initialScore = 0;
    for (auto& [n, c] : cfg) initialScore += c.score();
    cout << "Initial score: " << fixed << setprecision(6) << initialScore << endl;
    
    map<int, Cfg> results;
    double totalScore = 0;
    int improved = 0;
    
    for (int n = 1; n <= 200; n++) {
        if (!cfg.count(n)) continue;
        
        Cfg c = cfg[n];
        double origScore = c.score();
        double origSide = c.side();
        
        // Run multiple rounds of SA + fractional translation
        Cfg best = c;
        double bestScore = origScore;
        
        for (int r = 0; r < rounds; r++) {
            Cfg candidate = sa_gap(c, iterations, 0.5, 0.00001, 0.1, 30.0, 42 + r * 1000 + n);
            candidate = fractional_translation(candidate, 50);
            
            double candScore = candidate.score();
            if (candScore < bestScore && !candidate.anyOvl()) {
                bestScore = candScore;
                best = candidate;
            }
        }
        
        results[n] = best;
        totalScore += bestScore;
        
        if (bestScore < origScore - 1e-9) {
            improved++;
            cout << "N=" << setw(3) << n << ": " << origScore << " -> " << bestScore 
                 << " (improved by " << (origScore - bestScore) << ")" << endl;
        }
    }
    
    cout << "\nTotal score: " << totalScore << endl;
    cout << "Initial score: " << initialScore << endl;
    cout << "Improvement: " << (initialScore - totalScore) << endl;
    cout << "N values improved: " << improved << "/200" << endl;
    
    saveCSV("refined_result.csv", results);
    cout << "Saved to refined_result.csv" << endl;
    
    return 0;
}
