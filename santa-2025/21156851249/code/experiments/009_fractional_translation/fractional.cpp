// Fractional Translation Optimization
// Uses very small step sizes (0.001 to 0.00001) in 8 directions
// Different from SA - deterministic greedy improvement

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

constexpr int MAX_N = 205;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;

const double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
const double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

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

bool contains(const Poly& poly, Pt pt) {
    bool inside = false;
    for (int i = 0, j = NV - 1; i < NV; j = i++) {
        if (((poly.p[i].y > pt.y) != (poly.p[j].y > pt.y)) &&
            (pt.x < (poly.p[j].x - poly.p[i].x) * (pt.y - poly.p[i].y) / (poly.p[j].y - poly.p[i].y) + poly.p[i].x))
            inside = !inside;
    }
    return inside;
}

inline bool overlap(const Poly& a, const Poly& b) {
    if (a.x1 < b.x0 - 1e-13 || b.x1 < a.x0 - 1e-13 || a.y1 < b.y0 - 1e-13 || b.y1 < a.y0 - 1e-13) return false;
    auto ccw = [](Pt p, Pt q, Pt r) { 
        long double v = (long double)(q.y - p.y) * (r.x - q.x) - (long double)(q.x - p.x) * (r.y - q.y);
        return (v > 1e-20L) ? 1 : (v < -1e-20L ? -1 : 0); 
    };
    for (int i = 0; i < NV; i++) {
        for (int j = 0; j < NV; j++) {
            Pt p1 = a.p[i], q1 = a.p[(i+1)%NV], p2 = b.p[j], q2 = b.p[(j+1)%NV];
            if (ccw(p1, q1, p2) != ccw(p1, q1, q2) && ccw(p2, q2, p1) != ccw(p2, q2, q1)) return true;
        }
    }
    return contains(a, b.p[0]) || contains(b, a.p[0]);
}

Poly getPoly(double cx, double cy, double deg) {
    Poly q;
    double r = deg * PI / 180.0, c = cos(r), s = sin(r);
    for (int i = 0; i < NV; i++) {
        q.p[i].x = TX[i] * c - TY[i] * s + cx;
        q.p[i].y = TX[i] * s + TY[i] * c + cy;
    }
    q.bbox();
    return q;
}

struct Cfg {
    int n;
    double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];
    void upd(int i) { pl[i] = getPoly(x[i], y[i], a[i]); }
    void updAll() { for (int i = 0; i < n; i++) upd(i); }
    double side() {
        double xmin = 1e18, xmax = -1e18, ymin = 1e18, ymax = -1e18;
        for (int i = 0; i < n; i++) {
            xmin = min(xmin, pl[i].x0); xmax = max(xmax, pl[i].x1);
            ymin = min(ymin, pl[i].y0); ymax = max(ymax, pl[i].y1);
        }
        return max(xmax - xmin, ymax - ymin);
    }
    bool check_valid() {
        for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (overlap(pl[i], pl[j])) return false;
        return true;
    }
    bool hasOvl(int i) {
        for (int j = 0; j < n; j++) if (i != j && overlap(pl[i], pl[j])) return true;
        return false;
    }
    double score() { double s = side(); return s * s / n; }
};

// 8 directions: N, S, E, W, NE, NW, SE, SW
const double DX[8] = {0, 0, 1, -1, 1, -1, 1, -1};
const double DY[8] = {1, -1, 0, 0, 1, 1, -1, -1};

// Fractional translation step sizes
const double FRAC_STEPS[] = {0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001};
const int NUM_STEPS = 7;

// Fractional translation optimization
int fractionalTranslation(Cfg& c, int max_passes = 10) {
    int total_improvements = 0;
    
    for (int pass = 0; pass < max_passes; pass++) {
        int pass_improvements = 0;
        double initial_side = c.side();
        
        // For each step size
        for (int s = 0; s < NUM_STEPS; s++) {
            double step = FRAC_STEPS[s];
            
            // For each tree
            for (int i = 0; i < c.n; i++) {
                double best_side = c.side();
                double best_dx = 0, best_dy = 0;
                
                // Try all 8 directions
                for (int d = 0; d < 8; d++) {
                    double ox = c.x[i], oy = c.y[i];
                    
                    c.x[i] = ox + DX[d] * step;
                    c.y[i] = oy + DY[d] * step;
                    c.upd(i);
                    
                    if (!c.hasOvl(i)) {
                        double new_side = c.side();
                        if (new_side < best_side - 1e-15) {
                            best_side = new_side;
                            best_dx = DX[d] * step;
                            best_dy = DY[d] * step;
                        }
                    }
                    
                    // Restore
                    c.x[i] = ox;
                    c.y[i] = oy;
                    c.upd(i);
                }
                
                // Apply best move if found
                if (best_dx != 0 || best_dy != 0) {
                    c.x[i] += best_dx;
                    c.y[i] += best_dy;
                    c.upd(i);
                    pass_improvements++;
                }
            }
        }
        
        total_improvements += pass_improvements;
        
        double final_side = c.side();
        if (pass_improvements == 0 || final_side >= initial_side - 1e-15) {
            break;  // No improvement this pass
        }
    }
    
    return total_improvements;
}

// Also try angle adjustments with fractional steps
int fractionalRotation(Cfg& c, int max_passes = 5) {
    int total_improvements = 0;
    const double ANGLE_STEPS[] = {0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001};
    const int NUM_ANGLE_STEPS = 7;
    
    for (int pass = 0; pass < max_passes; pass++) {
        int pass_improvements = 0;
        double initial_side = c.side();
        
        for (int s = 0; s < NUM_ANGLE_STEPS; s++) {
            double step = ANGLE_STEPS[s];
            
            for (int i = 0; i < c.n; i++) {
                double best_side = c.side();
                double best_da = 0;
                
                // Try +/- angle step
                for (int dir = -1; dir <= 1; dir += 2) {
                    double oa = c.a[i];
                    c.a[i] = oa + dir * step;
                    c.upd(i);
                    
                    if (!c.hasOvl(i)) {
                        double new_side = c.side();
                        if (new_side < best_side - 1e-15) {
                            best_side = new_side;
                            best_da = dir * step;
                        }
                    }
                    
                    c.a[i] = oa;
                    c.upd(i);
                }
                
                if (best_da != 0) {
                    c.a[i] += best_da;
                    c.upd(i);
                    pass_improvements++;
                }
            }
        }
        
        total_improvements += pass_improvements;
        
        double final_side = c.side();
        if (pass_improvements == 0 || final_side >= initial_side - 1e-15) {
            break;
        }
    }
    
    return total_improvements;
}

map<int, Cfg> loadCSV(string fn) {
    map<int, Cfg> res; ifstream f(fn); string ln, h; if(!f) return res;
    getline(f, h);
    while (getline(f, ln)) {
        stringstream ss(ln); string id, sx, sy, sa;
        if(!getline(ss, id, ',')) continue;
        getline(ss, sx, ','); getline(ss, sy, ','); getline(ss, sa, ',');
        int n = stoi(id.substr(0, 3)), idx = stoi(id.substr(4));
        auto p = [](string s) { 
            size_t st = s.find_first_of("0123456789.-"); 
            return (st == string::npos) ? 0.0 : stod(s.substr(st)); 
        };
        res[n].n = n; res[n].x[idx] = p(sx); res[n].y[idx] = p(sy); res[n].a[idx] = p(sa);
    }
    for (auto& pair : res) pair.second.updAll();
    return res;
}

void saveCSV(string fn, map<int, Cfg>& res) {
    ofstream f(fn); f << "id,x,y,deg" << endl;
    for (int n = 1; n <= 200; n++) {
        if (!res.count(n)) continue;
        for (int i = 0; i < n; i++)
            f << setfill('0') << setw(3) << n << "_" << i << ",s" << fixed << setprecision(18) 
              << res[n].x[i] << ",s" << res[n].y[i] << ",s" << res[n].a[i] << "\n";
    }
}

int main() {
    cout << "Loading baseline..." << endl;
    auto baseline = loadCSV("/home/submission/submission.csv");
    
    double baseline_total = 0;
    for (auto& [n, c] : baseline) baseline_total += c.score();
    cout << "Baseline total score: " << fixed << setprecision(6) << baseline_total << endl;
    
    map<int, Cfg> best = baseline;
    int total_improvements = 0;
    
    cout << "\nRunning Fractional Translation on all N values..." << endl;
    cout << "Step sizes: 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001" << endl;
    cout << "Directions: N, S, E, W, NE, NW, SE, SW" << endl;
    
    vector<int> all_n;
    for (int n = 1; n <= 200; n++) all_n.push_back(n);
    
    #pragma omp parallel for schedule(dynamic) reduction(+:total_improvements)
    for (int i = 0; i < (int)all_n.size(); i++) {
        int n = all_n[i];
        Cfg c = baseline[n];
        double initial_side = c.side();
        
        // Apply fractional translation
        int trans_imp = fractionalTranslation(c, 10);
        
        // Apply fractional rotation
        int rot_imp = fractionalRotation(c, 5);
        
        double final_side = c.side();
        
        if (final_side < initial_side - 1e-12) {
            #pragma omp critical
            {
                if (c.check_valid() && c.side() < best[n].side() - 1e-12) {
                    cout << "IMPROVEMENT! N=" << n << ": " << initial_side << " -> " << final_side 
                         << " (saved " << initial_side - final_side << ")" << endl;
                    best[n] = c;
                    total_improvements++;
                }
            }
        }
    }
    
    // Calculate final score
    double final_total = 0;
    for (auto& [n, c] : best) final_total += c.score();
    
    cout << "\n========================================" << endl;
    cout << "Fractional Translation Complete" << endl;
    cout << "Baseline Score: " << fixed << setprecision(6) << baseline_total << endl;
    cout << "Final Score:    " << final_total << endl;
    cout << "Improvement:    " << baseline_total - final_total << endl;
    cout << "N values improved: " << total_improvements << endl;
    cout << "========================================" << endl;
    
    saveCSV("/home/submission/submission.csv", best);
    cout << "Saved to /home/submission/submission.csv" << endl;
    
    return 0;
}
