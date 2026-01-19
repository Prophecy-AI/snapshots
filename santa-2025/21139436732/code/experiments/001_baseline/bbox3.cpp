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
constexpr double EPSILON = 1e-16;
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

// --- Geometric Structures and Functions (Complex Numbers) ---

using Complex = std::complex<double>;

// Forward declarations for mutual dependency
struct Cfg;
struct Poly;

Cfg global_squeeze(Cfg c, uint64_t seed);
Complex getSeparationVector(const Poly& a, const Poly& b); 

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

// Minimal overlap check
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
            if (i != j && overlap(pl[i], pl[j])) {
                return true;
            }
        }
        return false;
    }

    bool hasOvlPair(int i, int j) const {
        if (overlap(pl[i], pl[j])) return true;
        for (int k = 0; k < n; k++) {
            if (k != i && k != j) {
                if (overlap(pl[i], pl[k]) || overlap(pl[j], pl[k])) {
                    return true;
                }
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

    Complex centroid() const {
        Complex sum = 0.0;
        for (int i = 0; i < n; i++) { sum += c[i]; }
        return sum / (double)n;
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

// --- Overlap Resolution Helper Function ---

Complex getSeparationVector(const Poly& a, const Poly& b) {
    Complex c_a = a.p[0]; for(int i=1; i<NV; ++i) c_a += a.p[i]; c_a /= (double)NV;
    Complex c_b = b.p[0]; for(int i=1; i<NV; ++i) c_b += b.p[i]; c_b /= (double)NV;

    Complex diff = c_a - c_b;
    double dist = abs(diff);
    double overlap_depth = 0.2; 

    if (dist > EPSILON) {
        return diff / dist * overlap_depth;
    } else {
        return Complex(rf() * 0.1, rf() * 0.1);
    }
}

// --- Optimization Routines ---

Cfg aggressive_repair(Cfg c, int max_cycles) {
    Cfg current = c;
    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        bool repaired = true;
        for (int i = 0; i < current.n; ++i) {
            for (int j = i + 1; j < current.n; ++j) {
                if (overlap(current.pl[i], current.pl[j])) {
                    repaired = false;
                    Complex sep_vector = getSeparationVector(current.pl[i], current.pl[j]);
                    current.c[i] += sep_vector * 0.5;
                    current.c[j] -= sep_vector * 0.5;
                    current.upd(i);
                    current.upd(j);
                }
            }
        }
        if (repaired) break;
    }
    return current;
}

Cfg global_squeeze(Cfg c, uint64_t seed) {
    rng.seed(seed);
    Cfg best = c;
    double best_side = c.side();

    for (int iter = 0; iter < 500; ++iter) {
        double scale = 0.995 + rf() * 0.004; // 0.995 to 0.999
        Cfg trial = c;
        Complex cent = trial.centroid();
        
        for (int i = 0; i < trial.n; ++i) {
            trial.c[i] = cent + (trial.c[i] - cent) * scale;
        }
        trial.updAll();
        
        // Repair overlaps
        trial = aggressive_repair(trial, 50);
        
        if (!trial.anyOvl()) {
            double s = trial.side();
            if (s < best_side) {
                best = trial;
                best_side = s;
            }
        }
    }
    return best;
}

Cfg local_search(Cfg c, int iters) {
    const double steps[] = {0.01, 0.004, 0.0015, 0.0006, 0.00025, 0.0001};
    const double angles[] = {5.0, 2.0, 0.8, 0.3, 0.1};
    const int dx[] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dy[] = {0, 0, 1, -1, 1, -1, 1, -1};
    
    Cfg best = c;
    double best_side = c.side();
    
    for (int it = 0; it < iters; ++it) {
        bool improved = false;
        
        // Focus on corner trees
        vector<int> corners = best.findCornerTrees();
        if (corners.empty()) {
            for (int i = 0; i < best.n; ++i) corners.push_back(i);
        }
        
        for (int idx : corners) {
            Complex orig_c = best.c[idx];
            double orig_a = best.a[idx];
            
            // Try position moves
            for (double step : steps) {
                for (int d = 0; d < 8; ++d) {
                    best.c[idx] = orig_c + Complex(dx[d] * step, dy[d] * step);
                    best.upd(idx);
                    
                    if (!best.hasOvl(idx)) {
                        double s = best.side();
                        if (s < best_side) {
                            best_side = s;
                            orig_c = best.c[idx];
                            improved = true;
                        }
                    }
                    best.c[idx] = orig_c;
                    best.upd(idx);
                }
            }
            
            // Try rotation moves
            for (double ang : angles) {
                for (int sign = -1; sign <= 1; sign += 2) {
                    best.a[idx] = orig_a + sign * ang;
                    best.upd(idx);
                    
                    if (!best.hasOvl(idx)) {
                        double s = best.side();
                        if (s < best_side) {
                            best_side = s;
                            orig_a = best.a[idx];
                            improved = true;
                        }
                    }
                    best.a[idx] = orig_a;
                    best.upd(idx);
                }
            }
        }
        
        if (!improved) break;
    }
    return best;
}

Cfg compaction(Cfg c, int iters) {
    Cfg best = c;
    double best_side = c.side();
    
    for (int it = 0; it < iters; ++it) {
        Complex cent = best.centroid();
        bool improved = false;
        
        for (int i = 0; i < best.n; ++i) {
            Complex orig = best.c[i];
            Complex dir = cent - orig;
            double dist = abs(dir);
            if (dist < EPSILON) continue;
            dir /= dist;
            
            // Move toward center
            for (double step = 0.1; step > 0.001; step *= 0.5) {
                best.c[i] = orig + dir * step;
                best.upd(i);
                
                if (!best.hasOvl(i)) {
                    double s = best.side();
                    if (s < best_side) {
                        best_side = s;
                        orig = best.c[i];
                        improved = true;
                    }
                }
                best.c[i] = orig;
                best.upd(i);
            }
        }
        
        if (!improved) break;
    }
    return best;
}

// --- I/O Functions ---

map<int, Cfg> loadCSV(const string& fname) {
    map<int, Cfg> cfgs;
    ifstream f(fname);
    string line;
    getline(f, line); // header
    
    while (getline(f, line)) {
        stringstream ss(line);
        string id, xs, ys, ds;
        getline(ss, id, ',');
        getline(ss, xs, ',');
        getline(ss, ys, ',');
        getline(ss, ds, ',');
        
        int n = stoi(id.substr(0, 3));
        int idx = stoi(id.substr(4));
        
        double x = stod(xs.substr(1));
        double y = stod(ys.substr(1));
        double d = stod(ds.substr(1));
        
        if (cfgs.find(n) == cfgs.end()) {
            cfgs[n].n = n;
        }
        cfgs[n].c[idx] = Complex(x, y);
        cfgs[n].a[idx] = d;
    }
    
    for (auto& [n, cfg] : cfgs) {
        cfg.updAll();
    }
    return cfgs;
}

void saveCSV(const string& fname, const map<int, Cfg>& cfgs) {
    ofstream f(fname);
    f << "id,x,y,deg\n";
    f << fixed << setprecision(12);
    
    for (int n = 1; n <= 200; ++n) {
        if (cfgs.find(n) == cfgs.end()) continue;
        const Cfg& cfg = cfgs.at(n);
        for (int i = 0; i < cfg.n; ++i) {
            f << setfill('0') << setw(3) << n << "_" << i << ",";
            f << "s" << cfg.c[i].real() << ",";
            f << "s" << cfg.c[i].imag() << ",";
            f << "s" << cfg.a[i] << "\n";
        }
    }
}

double totalScore(const map<int, Cfg>& cfgs) {
    double total = 0;
    for (const auto& [n, cfg] : cfgs) {
        total += cfg.score();
    }
    return total;
}

int main(int argc, char* argv[]) {
    string input_file = "submission.csv";
    string output_file = "submission_optimized.csv";
    int n_iters = 100;
    int n_restarts = 10;
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "-o" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "-n" && i + 1 < argc) n_iters = stoi(argv[++i]);
        else if (arg == "-r" && i + 1 < argc) n_restarts = stoi(argv[++i]);
    }
    
    cout << "Loading " << input_file << "..." << endl;
    map<int, Cfg> cfgs = loadCSV(input_file);
    cout << "Initial score: " << totalScore(cfgs) << endl;
    
    auto start = high_resolution_clock::now();
    
    #pragma omp parallel for schedule(dynamic)
    for (int n = 1; n <= 200; ++n) {
        if (cfgs.find(n) == cfgs.end()) continue;
        
        Cfg best = cfgs[n];
        double best_score = best.score();
        
        for (int r = 0; r < n_restarts; ++r) {
            Cfg trial = cfgs[n];
            
            // Apply optimization sequence
            trial = global_squeeze(trial, r * 12345 + n);
            trial = local_search(trial, n_iters);
            trial = compaction(trial, n_iters);
            trial = local_search(trial, n_iters);
            
            if (!trial.anyOvl()) {
                double s = trial.score();
                if (s < best_score) {
                    best = trial;
                    best_score = s;
                }
            }
        }
        
        #pragma omp critical
        {
            cfgs[n] = best;
            if (n % 20 == 0) {
                cout << "Processed N=" << n << ", score=" << best_score << endl;
            }
        }
    }
    
    auto end = high_resolution_clock::now();
    double elapsed = duration_cast<seconds>(end - start).count();
    
    cout << "Final score: " << totalScore(cfgs) << endl;
    cout << "Time: " << elapsed << " seconds" << endl;
    
    saveCSV(output_file, cfgs);
    cout << "Saved to " << output_file << endl;
    
    return 0;
}
