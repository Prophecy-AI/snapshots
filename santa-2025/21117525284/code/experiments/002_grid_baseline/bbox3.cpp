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

    // Overlap checks
    bool hasOvl(int i) const {
        bool overlap_found = false;
        #pragma omp parallel for reduction(||:overlap_found)
        for (int j = 0; j < n; j++) { 
            if (overlap_found) continue;
            if (i != j && overlap(pl[i], pl[j])) {
                overlap_found = true;
            }
        }
        return overlap_found;
    }

    bool hasOvlPair(int i, int j) const {
        if (overlap(pl[i], pl[j])) return true;

        bool overlap_found = false;
        #pragma omp parallel for reduction(||:overlap_found)
        for (int k = 0; k < n; k++) {
            if (overlap_found) continue;
            if (k != i && k != j) {
                if (overlap(pl[i], pl[k]) || overlap(pl[j], pl[k])) {
                    overlap_found = true;
                }
            }
        }
        return overlap_found;
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
    double best_side = best.side();
    
    for (int iter = 0; iter < 100; ++iter) {
        Cfg trial = best;
        double scale = 0.995 + rf() * 0.01;
        
        Complex center = trial.centroid();
        for (int i = 0; i < trial.n; ++i) {
            trial.c[i] = center + (trial.c[i] - center) * scale;
        }
        trial.updAll();
        
        if (trial.anyOvl()) {
            trial = aggressive_repair(trial, 50);
        }
        
        if (!trial.anyOvl() && trial.side() < best_side) {
            best = trial;
            best_side = trial.side();
        }
    }
    return best;
}

Cfg simulated_annealing(Cfg c, int iterations, int rotation_steps, double timeout_sec, uint64_t seed) {
    rng.seed(seed);
    
    Cfg best = c;
    Cfg current = c;
    double best_score = best.score();
    double current_score = best_score;
    
    double T = 0.1;
    double T_min = 1e-6;
    double alpha = pow(T_min / T, 1.0 / iterations);
    
    auto start = high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        auto now = high_resolution_clock::now();
        if (duration_cast<seconds>(now - start).count() >= timeout_sec) break;
        
        Cfg trial = current;
        int i = ri(trial.n);
        
        int move_type = ri(3);
        if (move_type == 0) {
            // Translation
            double dx = (rf() - 0.5) * 0.1 * T / 0.1;
            double dy = (rf() - 0.5) * 0.1 * T / 0.1;
            trial.c[i] += Complex(dx, dy);
        } else if (move_type == 1) {
            // Rotation
            double da = (rf() - 0.5) * 20 * T / 0.1;
            trial.a[i] += da;
        } else {
            // Swap
            int j = ri(trial.n);
            if (i != j) {
                swap(trial.c[i], trial.c[j]);
                swap(trial.a[i], trial.a[j]);
                trial.upd(j);
            }
        }
        trial.upd(i);
        
        if (!trial.hasOvl(i)) {
            double trial_score = trial.score();
            double delta = trial_score - current_score;
            
            if (delta < 0 || rf() < exp(-delta / T)) {
                current = trial;
                current_score = trial_score;
                
                if (current_score < best_score) {
                    best = current;
                    best_score = current_score;
                }
            }
        }
        
        T *= alpha;
    }
    
    return best;
}

// --- I/O Functions ---

map<int, Cfg> loadCSV(const string& fn) {
    map<int, Cfg> cfgs;
    ifstream f(fn);
    string line;
    getline(f, line); // header
    
    while (getline(f, line)) {
        stringstream ss(line);
        string id, xs, ys, ds;
        getline(ss, id, ',');
        getline(ss, xs, ',');
        getline(ss, ys, ',');
        getline(ss, ds, ',');
        
        // Parse id: NNN_T
        int n = stoi(id.substr(0, 3));
        int t = stoi(id.substr(4));
        
        // Remove 's' prefix
        double x = stod(xs.substr(1));
        double y = stod(ys.substr(1));
        double d = stod(ds.substr(1));
        
        if (cfgs.find(n) == cfgs.end()) {
            cfgs[n].n = n;
        }
        cfgs[n].c[t] = Complex(x, y);
        cfgs[n].a[t] = d;
    }
    
    for (auto& [n, cfg] : cfgs) {
        cfg.updAll();
    }
    
    return cfgs;
}

void saveCSV(const string& fn, const map<int, Cfg>& cfgs) {
    ofstream f(fn);
    f << fixed << setprecision(6);
    f << "id,x,y,deg\n";
    
    for (int n = 1; n <= 200; ++n) {
        if (cfgs.find(n) == cfgs.end()) continue;
        const Cfg& cfg = cfgs.at(n);
        for (int t = 0; t < cfg.n; ++t) {
            f << setfill('0') << setw(3) << n << "_" << t << ",";
            f << "s" << cfg.c[t].real() << ",";
            f << "s" << cfg.c[t].imag() << ",";
            f << "s" << cfg.a[t] << "\n";
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
    string output_file = "submission.csv";
    int iterations = 1000;
    int rotation_steps = 30;
    double timeout = 60;
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "-o" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "-n" && i + 1 < argc) iterations = stoi(argv[++i]);
        else if (arg == "-r" && i + 1 < argc) rotation_steps = stoi(argv[++i]);
        else if (arg == "-t" && i + 1 < argc) timeout = stod(argv[++i]);
    }
    
    cout << "Loading " << input_file << "..." << endl;
    auto cfgs = loadCSV(input_file);
    
    double initial_score = totalScore(cfgs);
    cout << "Initial score: " << fixed << setprecision(6) << initial_score << endl;
    
    cout << "Optimizing with " << iterations << " iterations, timeout " << timeout << "s..." << endl;
    
    #pragma omp parallel for schedule(dynamic)
    for (int n = 1; n <= 200; ++n) {
        if (cfgs.find(n) == cfgs.end()) continue;
        
        Cfg& cfg = cfgs[n];
        uint64_t seed = n * 12345;
        
        // Run simulated annealing
        Cfg optimized = simulated_annealing(cfg, iterations, rotation_steps, timeout / 200.0, seed);
        
        // Apply global squeeze
        optimized = global_squeeze(optimized, seed + 1);
        
        if (!optimized.anyOvl() && optimized.score() < cfg.score()) {
            #pragma omp critical
            {
                cfg = optimized;
            }
        }
    }
    
    double final_score = totalScore(cfgs);
    cout << "Final score: " << fixed << setprecision(6) << final_score << endl;
    cout << "Improvement: " << fixed << setprecision(6) << (initial_score - final_score) << endl;
    
    cout << "Saving to " << output_file << "..." << endl;
    saveCSV(output_file, cfgs);
    
    return 0;
}
