// Tree Packer - Simulated Annealing Optimizer
// Compile: g++ -O3 -march=native -std=c++17 -fopenmp -o tree_packer tree_packer.cpp
// Run: ./tree_packer -i input.csv -o output.csv -n 10000 -r 16

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
using namespace chrono;

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

    bool valid() const {
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (overlap(pl[i], pl[j])) return false;
        return true;
    }

    double score() const {
        double mnx = 1e9, mny = 1e9, mxx = -1e9, mxy = -1e9;
        for (int i = 0; i < n; i++) {
            mnx = min(mnx, pl[i].x0); mxx = max(mxx, pl[i].x1);
            mny = min(mny, pl[i].y0); mxy = max(mxy, pl[i].y1);
        }
        double side = max(mxx - mnx, mxy - mny);
        return side * side / n;
    }
};

Cfg configs[MAX_N + 1];
double bestScore[MAX_N + 1];

void loadCSV(const string& fn) {
    ifstream f(fn);
    string line;
    getline(f, line); // header
    while (getline(f, line)) {
        if (line.empty()) continue;
        // Parse: id,x,y,deg
        stringstream ss(line);
        string id, xs, ys, ds;
        getline(ss, id, ',');
        getline(ss, xs, ',');
        getline(ss, ys, ',');
        getline(ss, ds, ',');
        
        // Remove 's' prefix if present
        if (xs[0] == 's') xs = xs.substr(1);
        if (ys[0] == 's') ys = ys.substr(1);
        if (ds[0] == 's') ds = ds.substr(1);
        
        // Parse N from id (format: NNN_idx)
        int N = stoi(id.substr(0, 3));
        int idx = stoi(id.substr(4));
        
        if (N >= 1 && N <= MAX_N && idx < N) {
            configs[N].n = N;
            configs[N].x[idx] = stod(xs);
            configs[N].y[idx] = stod(ys);
            configs[N].a[idx] = stod(ds);
        }
    }
    
    for (int n = 1; n <= MAX_N; n++) {
        configs[n].updAll();
        bestScore[n] = configs[n].score();
    }
}

void saveCSV(const string& fn) {
    ofstream f(fn);
    f << "id,x,y,deg\n";
    f << fixed << setprecision(15);
    for (int n = 1; n <= MAX_N; n++) {
        for (int i = 0; i < n; i++) {
            f << setfill('0') << setw(3) << n << "_" << i << ",";
            f << "s" << configs[n].x[i] << ",";
            f << "s" << configs[n].y[i] << ",";
            f << "s" << configs[n].a[i] << "\n";
        }
    }
}

void optimize(int N, int iters, int restarts) {
    Cfg& cfg = configs[N];
    Cfg best = cfg;
    double bestS = cfg.score();
    
    for (int r = 0; r < restarts; r++) {
        Cfg cur = best;
        double T = 0.1;
        double Tmin = 1e-6;
        double alpha = 0.9999;
        
        for (int it = 0; it < iters; it++) {
            int i = ri(N);
            double ox = cur.x[i], oy = cur.y[i], oa = cur.a[i];
            
            // Random move
            double step = T * 0.5;
            cur.x[i] += (rf() * 2 - 1) * step;
            cur.y[i] += (rf() * 2 - 1) * step;
            cur.a[i] += (rf() * 2 - 1) * 10 * T;
            cur.upd(i);
            
            if (cur.hasOvl(i)) {
                cur.x[i] = ox; cur.y[i] = oy; cur.a[i] = oa;
                cur.upd(i);
            } else {
                double newS = cur.score();
                double delta = newS - bestS;
                if (delta < 0 || rf() < exp(-delta / T)) {
                    if (newS < bestS) {
                        bestS = newS;
                        best = cur;
                    }
                } else {
                    cur.x[i] = ox; cur.y[i] = oy; cur.a[i] = oa;
                    cur.upd(i);
                }
            }
            T = max(Tmin, T * alpha);
        }
    }
    
    if (bestS < bestScore[N]) {
        configs[N] = best;
        bestScore[N] = bestS;
    }
}

int main(int argc, char** argv) {
    string inFile = "submission.csv";
    string outFile = "output.csv";
    int iters = 10000;
    int restarts = 16;
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) inFile = argv[++i];
        else if (arg == "-o" && i + 1 < argc) outFile = argv[++i];
        else if (arg == "-n" && i + 1 < argc) iters = stoi(argv[++i]);
        else if (arg == "-r" && i + 1 < argc) restarts = stoi(argv[++i]);
    }
    
    cout << "Loading " << inFile << "..." << endl;
    loadCSV(inFile);
    
    double totalBefore = 0;
    for (int n = 1; n <= MAX_N; n++) totalBefore += bestScore[n];
    cout << "Initial score: " << totalBefore << endl;
    
    auto start = high_resolution_clock::now();
    
    #pragma omp parallel for schedule(dynamic)
    for (int n = 1; n <= MAX_N; n++) {
        optimize(n, iters, restarts);
        if (n % 20 == 0) {
            #pragma omp critical
            cout << "Completed N=" << n << ", score=" << bestScore[n] << endl;
        }
    }
    
    double totalAfter = 0;
    for (int n = 1; n <= MAX_N; n++) totalAfter += bestScore[n];
    
    auto end = high_resolution_clock::now();
    auto dur = duration_cast<seconds>(end - start).count();
    
    cout << "Final score: " << totalAfter << endl;
    cout << "Improvement: " << totalBefore - totalAfter << endl;
    cout << "Time: " << dur << "s" << endl;
    
    saveCSV(outFile);
    cout << "Saved to " << outFile << endl;
    
    return 0;
}
