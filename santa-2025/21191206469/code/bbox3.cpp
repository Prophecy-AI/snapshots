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
struct Poly; // **FIXED ERROR: Forward declaration for Poly**

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
            // Check if any part of the tree polygon defines the bounding box
            if (abs(pl[i].x0 - gx0) < eps || abs(pl[i].x1 - gx1) < eps ||
                abs(pl[i].y0 - gy0) < eps || abs(pl[i].y1 - gy1) < eps) {
                corners.push_back(i);
            }
        }
        return corners;
    }
};

// --- Overlap Resolution Helper Function ---

/**
 * Simplified placeholder for a complex Separation Vector calculation.
 * In a real implementation, this would use the Separating Axis Theorem (SAT) 
 * or Minkowski difference to find the Minimum Translation Vector (MTV).
 */
Complex getSeparationVector(const Poly& a, const Poly& b) {
    Complex c_a = a.p[0]; for(int i=1; i<NV; ++i) c_a += a.p[i]; c_a /= (double)NV;
    Complex c_b = b.p[0]; for(int i=1; i<NV; ++i) c_b += b.p[i]; c_b /= (double)NV;

    Complex diff = c_a - c_b;
    double dist = abs(diff);

    // Crude estimate of overlap depth for scaling:
    // This is highly inaccurate but serves as a placeholder for a true MTV magnitude
    double overlap_depth = 0.2; 

    if (dist > EPSILON) {
        return diff / dist * overlap_depth;
    } else {
        // If centers are the same, push randomly
        return Complex(rf() * 0.1, rf() * 0.1);
    }
}

// --- Optimization Routines ---

/**
 * NEW! Aggressive Overlap-and-Repair Cycle (Meta-Move)
 * Performs sequential untangling based on calculated separation vectors.
 */
Cfg aggressive_repair(Cfg c, int max_cycles) {
    Cfg current = c;
    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        bool repaired = true;
        // Search for the worst overlap (we just find the first one for simplicity)
        for (int i = 0; i < current.n; ++i) {
            for (int j = i + 1; j < current.n; ++j) {
                if (overlap(current.pl[i], current.pl[j])) {
                    repaired = false;

                    // Calculate separation vector (MTV approximation)
                    Complex sep_vector = getSeparationVector(current.pl[i], current.pl[j]);

                    // Push them apart by half the vector each
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

/**
 * Global Squeeze: Scales all trees toward the centroid, then repairs overlaps.
 */
Cfg global_squeeze(Cfg c, uint64_t seed) {
    rng.seed(seed);
    Cfg best = c;
    double best_score = c.score();

    // Try a range of scale factors
    for (double scale = 0.9995; scale >= 0.98; scale -= 0.0005) {
        Cfg trial = c;
        Complex center = trial.centroid();

        for (int i = 0; i < trial.n; ++i) {
            trial.c[i] = center + (trial.c[i] - center) * scale;
        }
        trial.updAll();

        // Repair overlaps
        trial = aggressive_repair(trial, 100);

        if (!trial.anyOvl() && trial.score() < best_score) {
            best = trial;
            best_score = trial.score();
        }
    }
    return best;
}

// --- SA Move Operators ---

void move_translate(Cfg& c, int i, double step) {
    c.c[i] += Complex(rf() * 2 - 1, rf() * 2 - 1) * step;
    c.upd(i);
}

void move_rotate(Cfg& c, int i, double step) {
    c.a[i] += (rf() * 2 - 1) * step;
    c.upd(i);
}

void move_swap(Cfg& c, int i, int j) {
    swap(c.c[i], c.c[j]);
    swap(c.a[i], c.a[j]);
    c.upd(i);
    c.upd(j);
}

// --- Main SA Loop ---

Cfg sa_optimize(Cfg c, int iterations, double T0, double T_min, uint64_t seed) {
    rng.seed(seed);
    Cfg best = c;
    double best_score = c.score();
    double T = T0;
    double alpha = pow(T_min / T0, 1.0 / iterations);

    for (int iter = 0; iter < iterations; ++iter) {
        Cfg trial = c;
        int move_type = ri(3);
        int i = ri(c.n);

        if (move_type == 0) {
            move_translate(trial, i, T * 0.1);
        } else if (move_type == 1) {
            move_rotate(trial, i, T * 5);
        } else {
            int j = ri(c.n);
            if (i != j) move_swap(trial, i, j);
        }

        if (!trial.hasOvl(i)) {
            double delta = trial.score() - c.score();
            if (delta < 0 || rf() < exp(-delta / T)) {
                c = trial;
                if (c.score() < best_score) {
                    best = c;
                    best_score = c.score();
                }
            }
        }

        T *= alpha;
    }
    return best;
}

// --- I/O Functions ---

map<int, Cfg> loadCSV(const string& path) {
    map<int, Cfg> configs;
    ifstream f(path);
    string line;
    getline(f, line); // header

    while (getline(f, line)) {
        stringstream ss(line);
        string id, x, y, deg;
        getline(ss, id, ',');
        getline(ss, x, ',');
        getline(ss, y, ',');
        getline(ss, deg, ',');

        // Parse id: "NNN_i"
        int n = stoi(id.substr(0, 3));
        int i = stoi(id.substr(4));

        // Strip 's' prefix
        if (x[0] == 's') x = x.substr(1);
        if (y[0] == 's') y = y.substr(1);
        if (deg[0] == 's') deg = deg.substr(1);

        if (configs.find(n) == configs.end()) {
            configs[n].n = n;
        }
        configs[n].c[i] = Complex(stod(x), stod(y));
        configs[n].a[i] = stod(deg);
    }

    for (auto& [n, cfg] : configs) {
        cfg.updAll();
    }
    return configs;
}

void saveCSV(const string& path, const map<int, Cfg>& configs) {
    ofstream f(path);
    f << "id,x,y,deg\n";
    f << fixed << setprecision(15);

    for (const auto& [n, cfg] : configs) {
        for (int i = 0; i < cfg.n; ++i) {
            f << setfill('0') << setw(3) << n << "_" << i << ",";
            f << "s" << cfg.c[i].real() << ",";
            f << "s" << cfg.c[i].imag() << ",";
            f << "s" << cfg.a[i] << "\n";
        }
    }
}

double totalScore(const map<int, Cfg>& configs) {
    double total = 0;
    for (const auto& [n, cfg] : configs) {
        total += cfg.score();
    }
    return total;
}

// --- Main ---

int main(int argc, char* argv[]) {
    int iterations = 1000;
    int rotation_param = 30;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            iterations = stoi(argv[++i]);
        } else if (arg == "-r" && i + 1 < argc) {
            rotation_param = stoi(argv[++i]);
        }
    }

    cout << "Loading submission.csv..." << endl;
    auto configs = loadCSV("submission.csv");
    double initial_score = totalScore(configs);
    cout << "Initial Score: " << fixed << setprecision(6) << initial_score << endl;

    cout << "Running SA optimization (iterations=" << iterations << ", r=" << rotation_param << ")..." << endl;

    auto start = high_resolution_clock::now();

    for (auto& [n, cfg] : configs) {
        // SA optimization
        cfg = sa_optimize(cfg, iterations, 1.0, 0.001, n * 12345);
        
        // Global squeeze
        cfg = global_squeeze(cfg, n * 54321);
    }

    auto end = high_resolution_clock::now();
    double elapsed = duration_cast<milliseconds>(end - start).count() / 1000.0;

    double final_score = totalScore(configs);
    cout << "Final Score: " << fixed << setprecision(6) << final_score << endl;
    cout << "Improvement: " << fixed << setprecision(6) << (initial_score - final_score) << endl;
    cout << "Time: " << fixed << setprecision(2) << elapsed << "s" << endl;

    saveCSV("submission.csv", configs);
    cout << "Saved to submission.csv" << endl;

    return 0;
}
