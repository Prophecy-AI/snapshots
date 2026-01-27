// Chaos Optimizer - Aggressive Optimization
// Compile: g++ -Ofast -march=native -std=c++17 -fopenmp -o chaos_optimizer chaos_optimizer.cpp

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <string>
#include <chrono>
#include <random>
#include <omp.h>

using namespace std;

constexpr int MAX_N = 250;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;

// Fast RNG
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
    inline double rf2() { return rf() * 2.0 - 1.0; } // -1 to 1
    inline int ri(int n) { return next() % n; }
    inline double gaussian() {
        double u1 = rf() + 1e-10, u2 = rf();
        return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    }
};

alignas(64) const double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
alignas(64) const double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

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

    void upd(int i) { getPoly(x[i], y[i], a[i], pl[i]); }
    
    void updAll() {
        gx0 = gy0 = 1e9; gx1 = gy1 = -1e9;
        for (int i = 0; i < n; i++) {
            upd(i);
            if (pl[i].x0 < gx0) gx0 = pl[i].x0;
            if (pl[i].x1 > gx1) gx1 = pl[i].x1;
            if (pl[i].y0 < gy0) gy0 = pl[i].y0;
            if (pl[i].y1 > gy1) gy1 = pl[i].y1;
        }
    }
    
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

// Squeeze aggressively
Cfg squeeze(Cfg c) {
    double cx = (c.gx0 + c.gx1) / 2, cy = (c.gy0 + c.gy1) / 2;
    for (double f = 0.9995; f > 0.99; f -= 0.0005) {
        Cfg trial = c;
        for (int i = 0; i < c.n; i++) {
            trial.x[i] = cx + (c.x[i] - cx) * f;
            trial.y[i] = cy + (c.y[i] - cy) * f;
        }
        trial.updAll();
        if (!trial.anyOvl()) c = trial; else break;
    }
    return c;
}

// Chaos Optimization Loop
Cfg chaos_optimize(Cfg c, int iterations, double T0, double Tmin, FastRNG& rng) {
    Cfg best = c, cur = c;
    double bestScore = best.score();
    double curScore = bestScore;
    double T = T0;
    double alpha = pow(Tmin / T0, 1.0 / iterations);
    
    for (int iter = 0; iter < iterations; iter++) {
        int moveType = rng.ri(100);
        double scale = T / T0; 
        bool valid = true;
        
        // High temp -> large moves
        double moveScale = 0.05 * scale + 0.001;
        double rotScale = 30.0 * scale + 1.0;

        if (moveType < 40) { // Shift
            int i = rng.ri(c.n);
            double ox = cur.x[i], oy = cur.y[i];
            cur.x[i] += rng.gaussian() * moveScale;
            cur.y[i] += rng.gaussian() * moveScale;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }
        else if (moveType < 70) { // Rotate
            int i = rng.ri(c.n);
            double oa = cur.a[i];
            cur.a[i] += rng.gaussian() * rotScale;
            while (cur.a[i]<0) cur.a[i]+=360; while(cur.a[i]>=360) cur.a[i]-=360;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.a[i]=oa; cur.upd(i); valid=false; }
        }
        else if (moveType < 90) { // Shift Towards Center
            int i = rng.ri(c.n);
            double ox = cur.x[i], oy = cur.y[i];
            double cx = (cur.gx0+cur.gx1)/2, cy = (cur.gy0+cur.gy1)/2;
            double dx = cx - cur.x[i], dy = cy - cur.y[i];
            double d = sqrt(dx*dx+dy*dy);
            if(d > 1e-6) {
                cur.x[i] += dx/d * rng.rf() * moveScale;
                cur.y[i] += dy/d * rng.rf() * moveScale;
            }
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }
        else { // Chaos Jump (Teleport)
            int i = rng.ri(c.n);
            double ox=cur.x[i], oy=cur.y[i], oa=cur.a[i];
            cur.x[i] = (cur.gx0 + rng.rf()) * 1.01; // nearby
            cur.y[i] = (cur.gy0 + rng.rf()) * 1.01;
            cur.a[i] = rng.rf() * 360.0;
             cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.a[i]=oa; cur.upd(i); valid=false; }
        }

        if (!valid) { T *= alpha; continue; }
        
        cur.updGlobal();
        double newScore = cur.score();
        double delta = newScore - curScore; 
        
        // Minimization SA acceptance
        bool accept = (delta < 0) || (T > 1e-10 && rng.rf() < exp(-delta / T));
        
        if (accept) {
            curScore = newScore;
            if (newScore < bestScore) {
                bestScore = newScore;
                best = cur;
            }
        } else {
            cur = best; curScore = bestScore;
        }
        
        T *= alpha;
        if (T < Tmin) T = Tmin;
    }
    
    return squeeze(best);
}

// Map IO
map<int, Cfg> loadCSV(const string& fn) {
    map<int, Cfg> cfg;
    ifstream f(fn);
    if (!f) return cfg;
    string ln; getline(f, ln);
    map<int, vector<tuple<int,double,double,double>>> data;
    while (getline(f, ln)) {
        size_t p1=ln.find(','), p2=ln.find(',',p1+1), p3=ln.find(',',p2+1);
        if (p1 == string::npos) continue;
        string id=ln.substr(0,p1), xs=ln.substr(p1+1,p2-p1-1), ys=ln.substr(p2+1,p3-p2-1), ds=ln.substr(p3+1);
        if(!xs.empty() && xs[0]=='s') xs=xs.substr(1);
        if(!ys.empty() && ys[0]=='s') ys=ys.substr(1);
        if(!ds.empty() && ds[0]=='s') ds=ds.substr(1);
        try { data[stoi(id.substr(0,3))].push_back({stoi(id.substr(4)), stod(xs), stod(ys), stod(ds)}); } catch(...) {}
    }
    for (auto& [n,v] : data) {
        Cfg c; c.n = n;
        for (auto& [i,x,y,d] : v) if (i < n) { c.x[i]=x; c.y[i]=y; c.a[i]=d; }
        c.updAll();
        cfg[n] = c;
    }
    return cfg;
}

void saveCSV(const string& fn, const map<int, Cfg>& cfg) {
    ofstream f(fn);
    f << fixed << setprecision(17) << "id,x,y,deg\n";
    for (int n = 1; n <= 200; n++) {
        if (cfg.count(n)) {
            const Cfg& c = cfg.at(n);
            for (int i = 0; i < n; i++)
                f << setfill('0') << setw(3) << n << "_" << i << ",s" << c.x[i] << ",s" << c.y[i] << ",s" << c.a[i] << "\n";
        }
    }
}

int main(int argc, char** argv) {
    string in = "submission.csv", out = "submission_chaos.csv";
    int iter = 50000;
    int repeats = 1;
    double t0 = 50.0;
    
    for(int i=1;i<argc;i++) {
        string a=argv[i];
        if (a=="-i") in=argv[++i];
        else if (a=="-o") out=argv[++i];
        else if (a=="-n") iter=stoi(argv[++i]);
        else if (a=="-r") repeats=stoi(argv[++i]);
        else if (a=="-t") t0=stod(argv[++i]);
    }
    
    auto cfg = loadCSV(in);
    if (cfg.empty()) return 1;
    
    printf("CHAOS OPTIMIZER (AGGRESSIVE MODE)\n");
    printf("Input: %s, Iter: %d, Repeats: %d\n\n", in.c_str(), iter, repeats);
    
    vector<int> keys;    for(auto& [n,_]:cfg) keys.push_back(n);

    double totalInit = 0, totalFinal = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+:totalInit, totalFinal)
    for (int i = 0; i < (int)keys.size(); i++) {
        int n = keys[i];
        Cfg c = cfg[n];
        double startScore = c.score();
        totalInit += startScore;
        
        Cfg best = c;
        for (int r=0; r<repeats; r++) {
            FastRNG rng(n * 999 + r * 7 + 1);
            Cfg attempt = chaos_optimize(c, iter, t0, 0.0001, rng);
            if (!attempt.anyOvl() && attempt.score() < best.score()) {
                best = attempt;
            }
        }
        
        if (best.score() < startScore - 1e-9) {
            #pragma omp critical
            printf("N=%d: %.6f -> %.6f (CHAOS! %.4f%%)\n", n, startScore, best.score(), (best.score()-startScore)/startScore*100);
            cfg[n] = best;
        }
        totalFinal += cfg[n].score();
    }
    
    saveCSV(out, cfg);
    printf("\nTotal: %.6f -> %.6f\n", totalInit, totalFinal);
    printf("Diff: %+.6f\n", totalFinal - totalInit);
    return 0;
}
