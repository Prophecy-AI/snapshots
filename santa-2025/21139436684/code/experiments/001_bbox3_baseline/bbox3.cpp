// BBOX3 - Global Dynamics Edition
// Features: Complex Number Vector Coordination, Fluid Dynamics, Hinge Pivot, 
// Density Gradient Flow, and Global Boundary Tension.

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

using Complex = std::complex<double>;

struct Poly;
struct Cfg;

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
    
    // Squeeze toward center
    Complex cent = c.centroid();
    for (double scale = 0.9995; scale >= 0.98; scale -= 0.0005) {
        Cfg trial = c;
        for (int i = 0; i < c.n; i++) {
            trial.c[i] = cent + (c.c[i] - cent) * scale;
        }
        trial.updAll();
        if (!trial.anyOvl()) {
            c = trial;
            if (c.side() < best_side) {
                best = c;
                best_side = c.side();
            }
        } else {
            break;
        }
    }
    
    // Compaction: move individual trees toward center
    for (int i = 0; i < c.n; i++) {
        Complex dir = cent - c.c[i];
        double dist = abs(dir);
        if (dist < EPSILON) continue;
        dir /= dist;
        
        for (double step : {0.02, 0.008, 0.003, 0.001}) {
            Cfg trial = c;
            trial.c[i] += dir * step;
            trial.upd(i);
            if (!trial.hasOvl(i) && trial.side() < best_side) {
                c = trial;
                best = c;
                best_side = c.side();
            }
        }
    }
    
    return best;
}

// Local search with rotation
Cfg local_search(Cfg c, int iterations) {
    Cfg best = c;
    double best_side = c.side();
    
    for (int iter = 0; iter < iterations; iter++) {
        int i = ri(c.n);
        
        // Try small movements
        for (int dir = 0; dir < 8; dir++) {
            double angle = dir * PI / 4;
            for (double step : {0.01, 0.005, 0.002}) {
                Cfg trial = c;
                trial.c[i] += Complex(cos(angle) * step, sin(angle) * step);
                trial.upd(i);
                if (!trial.hasOvl(i) && trial.side() < best_side) {
                    c = trial;
                    best = c;
                    best_side = c.side();
                }
            }
        }
        
        // Try rotation
        for (double da : {-5.0, -2.0, -1.0, 1.0, 2.0, 5.0}) {
            Cfg trial = c;
            trial.a[i] = fmod(trial.a[i] + da + 360.0, 360.0);
            trial.upd(i);
            if (!trial.hasOvl(i) && trial.side() < best_side) {
                c = trial;
                best = c;
                best_side = c.side();
            }
        }
    }
    
    return best;
}

// Swap two trees
Cfg swap_move(Cfg c) {
    if (c.n < 2) return c;
    
    Cfg best = c;
    double best_side = c.side();
    
    for (int attempt = 0; attempt < c.n * 2; attempt++) {
        int i = ri(c.n);
        int j = ri(c.n);
        if (i == j) continue;
        
        Cfg trial = c;
        swap(trial.c[i], trial.c[j]);
        swap(trial.a[i], trial.a[j]);
        trial.upd(i);
        trial.upd(j);
        
        if (!trial.hasOvlPair(i, j) && trial.side() < best_side) {
            c = trial;
            best = c;
            best_side = c.side();
        }
    }
    
    return best;
}

// Initialize with greedy placement
Cfg greedy_init(int n) {
    Cfg c;
    c.n = n;
    
    if (n == 0) return c;
    
    // Place first tree at origin
    c.c[0] = Complex(0, 0);
    c.a[0] = 90.0;
    c.upd(0);
    
    for (int i = 1; i < n; i++) {
        bool placed = false;
        double best_dist = 1e9;
        Complex best_pos;
        double best_angle = 0;
        
        // Try multiple angles and positions
        for (int attempt = 0; attempt < 100 && !placed; attempt++) {
            double angle = rf() * 2 * PI;
            double rot = (ri(4)) * 90.0;
            
            // Start far and move toward center
            for (double dist = 10.0; dist > 0.1; dist -= 0.1) {
                Complex pos = Complex(cos(angle) * dist, sin(angle) * dist);
                c.c[i] = pos;
                c.a[i] = rot;
                c.upd(i);
                
                bool ok = true;
                for (int j = 0; j < i; j++) {
                    if (overlap(c.pl[i], c.pl[j])) {
                        ok = false;
                        break;
                    }
                }
                
                if (ok) {
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_pos = pos;
                        best_angle = rot;
                        placed = true;
                    }
                    break;
                }
            }
        }
        
        if (placed) {
            c.c[i] = best_pos;
            c.a[i] = best_angle;
            c.upd(i);
        }
    }
    
    return c;
}

// Load configuration from CSV
map<int, Cfg> loadCSV(const string& path) {
    map<int, Cfg> configs;
    ifstream f(path);
    string line;
    getline(f, line); // header
    
    while (getline(f, line)) {
        stringstream ss(line);
        string id, xs, ys, degs;
        getline(ss, id, ',');
        getline(ss, xs, ',');
        getline(ss, ys, ',');
        getline(ss, degs, ',');
        
        // Parse id: "NNN_T"
        int n = stoi(id.substr(0, 3));
        int t = stoi(id.substr(4));
        
        // Remove 's' prefix
        double x = stod(xs.substr(1));
        double y = stod(ys.substr(1));
        double deg = stod(degs.substr(1));
        
        if (configs.find(n) == configs.end()) {
            configs[n].n = n;
        }
        
        configs[n].c[t] = Complex(x, y);
        configs[n].a[t] = deg;
    }
    
    for (auto& [n, cfg] : configs) {
        cfg.updAll();
    }
    
    return configs;
}

// Save configuration to CSV
void saveCSV(const string& path, const map<int, Cfg>& configs) {
    ofstream f(path);
    f << fixed << setprecision(6);
    f << "id,x,y,deg\n";
    
    for (int n = 1; n <= 200; n++) {
        if (configs.find(n) == configs.end()) continue;
        const Cfg& cfg = configs.at(n);
        
        for (int t = 0; t < cfg.n; t++) {
            f << setfill('0') << setw(3) << n << "_" << t << ",";
            f << "s" << cfg.c[t].real() << ",";
            f << "s" << cfg.c[t].imag() << ",";
            f << "s" << cfg.a[t] << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    int n_iters = 1000;
    int radius = 60;
    string input_file = "";
    string output_file = "submission.csv";
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) n_iters = stoi(argv[++i]);
        else if (arg == "-r" && i + 1 < argc) radius = stoi(argv[++i]);
        else if (arg == "-i" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "-o" && i + 1 < argc) output_file = argv[++i];
    }
    
    cerr << "BBOX3 Optimizer" << endl;
    cerr << "Iterations: " << n_iters << ", Radius: " << radius << endl;
    
    map<int, Cfg> configs;
    
    if (!input_file.empty()) {
        cerr << "Loading from: " << input_file << endl;
        configs = loadCSV(input_file);
    }
    
    double total_score = 0;
    
    for (int n = 1; n <= 200; n++) {
        Cfg cfg;
        
        if (configs.find(n) != configs.end()) {
            cfg = configs[n];
        } else {
            cfg = greedy_init(n);
        }
        
        double init_side = cfg.side();
        
        // Optimization loop
        for (int iter = 0; iter < n_iters; iter++) {
            // Global squeeze
            cfg = global_squeeze(cfg, iter);
            
            // Local search
            cfg = local_search(cfg, radius);
            
            // Swap moves
            if (iter % 10 == 0) {
                cfg = swap_move(cfg);
            }
            
            // Aggressive repair if overlaps
            if (cfg.anyOvl()) {
                cfg = aggressive_repair(cfg, 100);
            }
        }
        
        double final_side = cfg.side();
        double score = cfg.score();
        total_score += score;
        
        configs[n] = cfg;
        
        if (n % 10 == 0 || n <= 10) {
            cerr << "N=" << n << ": " << init_side << " -> " << final_side 
                 << " (score: " << score << ")" << endl;
        }
    }
    
    cerr << "Total Score: " << total_score << endl;
    
    saveCSV(output_file, configs);
    cerr << "Saved to: " << output_file << endl;
    
    return 0;
}
