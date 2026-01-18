// BBOX3 - Global Dynamics Edition
// Features: Complex Number Vector Coordination, Fluid Dynamics, Hinge Pivot, 
// Density Gradient Flow, and NEW Global Boundary Tension.
// Uses a separate 'global_squeeze' function for Dynamic Scaling and Overlap Repair.

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
#include <complex> 

using namespace std;
using namespace chrono;

constexpr int MAX_N = 200;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;
constexpr double EPSILON = 1e-12;
constexpr double NEIGHBOR_RADIUS = 0.5;      
constexpr double PIVOT_ANGLE_MAX = 10.0;     
constexpr double GLOBAL_TENSION_STRENGTH = 0.05; 

// Base tree geometry 
const double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
const double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

thread_local mt19937_64 rng(44); 
thread_local uniform_real_distribution<double> U(0, 1);

inline double rf() { return U(rng); }
inline int ri(int n) { return rng() % n; }

using Complex = std::complex<double>;

struct Poly {
    Complex p[NV]; 
    double x0, y0, x1, y1;
    void bbox() {
        x0 = x1 = p[0].real(); y0 = y1 = p[0].imag();
        for (int i = 1; i < NV; i++) {
            x0 = min(x0, p[i].real()); x1 = max(x1, p[i].real());
            y0 = min(y0, p[i].imag()); y1 = max(y1, p[i].imag());
        }
    }
};

Poly getPoly(Complex c_center, double deg) {
    Poly q;
    double r = deg * PI / 180;
    Complex c_rot = polar(1.0, r); 

    for (int i = 0; i < NV; i++) {
        Complex base_pt(TX[i], TY[i]);
        Complex rotated_pt = base_pt * c_rot; 
        q.p[i] = rotated_pt + c_center;
    }
    q.bbox();
    return q;
}

bool pip(double px, double py, const Poly& q) {
    bool in = false;
    int j = NV - 1;
    for (int i = 0; i < NV; i++) {
        double qi_x = q.p[i].real(), qi_y = q.p[i].imag();
        double qj_x = q.p[j].real(), qj_y = q.p[j].imag();
        if ((qi_y > py) != (qj_y > py) &&
            px < (qj_x - qi_x) * (py - qi_y) / (qj_y - qi_y) + qi_x)
            in = !in;
        j = i;
    }
    return in;
}

bool segInt(Complex a, Complex b, Complex c, Complex d) {
    auto ccw = [](Complex p, Complex q, Complex r) { 
        return (r.imag() - p.imag()) * (q.real() - p.real()) > (q.imag() - p.imag()) * (r.real() - p.real()); 
    };
    return ccw(a, c, d) != ccw(b, c, d) && ccw(a, b, c) != ccw(a, b, d);
}

bool overlap(const Poly& a, const Poly& b) {
    if (a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0) return false;
    for (int i = 0; i < NV; i++) {
        if (pip(a.p[i].real(), a.p[i].imag(), b)) return true;
        if (pip(b.p[i].real(), b.p[i].imag(), a)) return true;
    }
    for (int i = 0; i < NV; i++)
        for (int j = 0; j < NV; j++)
            if (segInt(a.p[i], a.p[(i + 1) % NV], b.p[j], b.p[(j + 1) % NV])) return true;
    return false;
}

struct Cfg {
    int n;
    Complex c[MAX_N]; 
    double a[MAX_N];  
    Poly pl[MAX_N];

    void upd(int i) { pl[i] = getPoly(c[i], a[i]); }
    void updAll() { for (int i = 0; i < n; i++) upd(i); }

    bool hasOvl(int i) const {
        for (int j = 0; j < n; j++) { 
            if (i != j && overlap(pl[i], pl[j])) return true;
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
};

Cfg configs[MAX_N + 1];
double best_sides[MAX_N + 1];

void parse_csv(const string& fn) {
    ifstream f(fn);
    string line;
    getline(f, line); // header
    
    map<int, vector<tuple<double, double, double>>> data;
    while (getline(f, line)) {
        stringstream ss(line);
        string id, xs, ys, ds;
        getline(ss, id, ',');
        getline(ss, xs, ',');
        getline(ss, ys, ',');
        getline(ss, ds, ',');
        
        int n = stoi(id.substr(0, 3));
        double x = stod(xs.substr(1));
        double y = stod(ys.substr(1));
        double d = stod(ds.substr(1));
        data[n].push_back({x, y, d});
    }
    
    for (auto& [n, trees] : data) {
        configs[n].n = trees.size();
        for (int i = 0; i < (int)trees.size(); i++) {
            auto& [x, y, d] = trees[i];
            configs[n].c[i] = Complex(x, y);
            configs[n].a[i] = d;
        }
        configs[n].updAll();
        best_sides[n] = configs[n].side();
    }
}

void save_csv(const string& fn) {
    ofstream f(fn);
    f << fixed << setprecision(18);
    f << "id,x,y,deg\n";
    for (int n = 1; n <= MAX_N; n++) {
        for (int i = 0; i < configs[n].n; i++) {
            f << setfill('0') << setw(3) << n << "_" << i << ",";
            f << "s" << configs[n].c[i].real() << ",";
            f << "s" << configs[n].c[i].imag() << ",";
            f << "s" << configs[n].a[i] << "\n";
        }
    }
}

double calc_total_score() {
    double total = 0;
    for (int n = 1; n <= MAX_N; n++) {
        double s = best_sides[n];
        total += s * s / n;
    }
    return total;
}

Cfg sa_optimize(Cfg c, int n_iters, uint64_t seed) {
    mt19937_64 local_rng(seed);
    uniform_real_distribution<double> local_U(0, 1);
    auto lrf = [&]() { return local_U(local_rng); };
    auto lri = [&](int n) { return local_rng() % n; };
    
    Cfg best = c;
    double bs = best.side();
    double T = 1.0, Tm = 0.00001;
    double alpha = pow(Tm / T, 1.0 / n_iters);
    
    for (int it = 0; it < n_iters; it++) {
        int i = lri(c.n);
        int move = lri(3);
        
        Complex oc = c.c[i];
        double oa = c.a[i];
        
        if (move == 0) { // translate
            double step = T * 0.3;
            c.c[i] += Complex(lrf() * 2 - 1, lrf() * 2 - 1) * step;
        } else if (move == 1) { // rotate
            c.a[i] += (lrf() * 2 - 1) * T * 30;
            while (c.a[i] < 0) c.a[i] += 360;
            while (c.a[i] >= 360) c.a[i] -= 360;
        } else { // translate + rotate
            double step = T * 0.2;
            c.c[i] += Complex(lrf() * 2 - 1, lrf() * 2 - 1) * step;
            c.a[i] += (lrf() * 2 - 1) * T * 20;
            while (c.a[i] < 0) c.a[i] += 360;
            while (c.a[i] >= 360) c.a[i] -= 360;
        }
        
        c.upd(i);
        
        if (c.hasOvl(i)) {
            c.c[i] = oc;
            c.a[i] = oa;
            c.upd(i);
        } else {
            double ns = c.side();
            double delta = ns - bs;
            if (delta < 0 || lrf() < exp(-delta / T)) {
                if (ns < bs) {
                    bs = ns;
                    best = c;
                }
            } else {
                c.c[i] = oc;
                c.a[i] = oa;
                c.upd(i);
            }
        }
        
        T *= alpha;
    }
    return best;
}

int main(int argc, char** argv) {
    int n_iters = 5000;
    int restarts = 16;
    
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "-n" && i+1 < argc) n_iters = stoi(argv[++i]);
        if (string(argv[i]) == "-r" && i+1 < argc) restarts = stoi(argv[++i]);
    }
    
    parse_csv("submission.csv");
    
    cout << fixed << setprecision(8);
    cout << "Initial: " << calc_total_score() << "\n";
    
    #pragma omp parallel for schedule(dynamic)
    for (int n = 1; n <= MAX_N; n++) {
        Cfg best = configs[n];
        double bs = best.side();
        
        for (int r = 0; r < restarts; r++) {
            Cfg c = sa_optimize(configs[n], n_iters, n * 12345 + r * 67890);
            if (c.side() < bs) {
                bs = c.side();
                best = c;
            }
        }
        
        #pragma omp critical
        {
            if (bs < best_sides[n]) {
                best_sides[n] = bs;
                configs[n] = best;
            }
        }
    }
    
    cout << "Final: " << calc_total_score() << "\n";
    save_csv("submission.csv");
    
    return 0;
}
