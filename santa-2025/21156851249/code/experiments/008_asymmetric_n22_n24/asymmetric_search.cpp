// Asymmetric Configuration Search for N=22 and N=24
// Focus on finding configurations that break symmetric patterns
// Target: N=22 score < 0.36 (vs current 0.375258)

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
#include <random>
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
    double score() { double s = side(); return s * s / n; }
};

// Generate asymmetric starting configuration
Cfg generateAsymmetric(int n, mt19937_64& rng) {
    Cfg c;
    c.n = n;
    
    uniform_real_distribution<double> U(0, 1);
    uniform_real_distribution<double> angle_dist(0, 360);
    
    // Use different angle distributions for asymmetry
    // Mix of 0°, 45°, 90°, 180°, and random angles
    vector<double> base_angles = {0, 45, 90, 135, 180, 225, 270, 315};
    
    double area_per_tree = 0.5;
    double side_estimate = sqrt(n * area_per_tree) * 1.3;
    
    for (int i = 0; i < n; i++) {
        // Asymmetric placement - not a regular grid
        double r = U(rng);
        if (r < 0.3) {
            // Use base angle with small perturbation
            c.a[i] = base_angles[rng() % base_angles.size()] + (U(rng) - 0.5) * 20;
        } else if (r < 0.6) {
            // Use completely random angle
            c.a[i] = angle_dist(rng);
        } else {
            // Use angle that's offset from common patterns
            c.a[i] = 23.6 + (rng() % 4) * 90 + (U(rng) - 0.5) * 30;
        }
        
        // Asymmetric positions - not centered
        c.x[i] = (U(rng) - 0.3) * side_estimate;  // Offset from center
        c.y[i] = (U(rng) - 0.4) * side_estimate;  // Different offset
    }
    c.updAll();
    return c;
}

// Repair overlaps
bool repairOverlaps(Cfg& c, mt19937_64& rng, int max_attempts = 500) {
    uniform_real_distribution<double> U(0, 1);
    
    for (int attempt = 0; attempt < max_attempts; attempt++) {
        bool has_overlap = false;
        for (int i = 0; i < c.n && !has_overlap; i++) {
            for (int j = i + 1; j < c.n; j++) {
                if (overlap(c.pl[i], c.pl[j])) {
                    has_overlap = true;
                    double dx = c.x[j] - c.x[i];
                    double dy = c.y[j] - c.y[i];
                    double dist = sqrt(dx*dx + dy*dy);
                    if (dist < 0.01) { 
                        dx = U(rng) - 0.5; 
                        dy = U(rng) - 0.5; 
                        dist = sqrt(dx*dx + dy*dy); 
                    }
                    double push = 0.05 + U(rng) * 0.1;
                    c.x[i] -= push * dx / dist;
                    c.y[i] -= push * dy / dist;
                    c.x[j] += push * dx / dist;
                    c.y[j] += push * dy / dist;
                    c.upd(i);
                    c.upd(j);
                    break;
                }
            }
        }
        if (!has_overlap) return true;
    }
    return c.check_valid();
}

// Intensive local search
double localSearch(Cfg& c, mt19937_64& rng, int iterations) {
    uniform_real_distribution<double> U(0, 1);
    double best_s = c.side();
    
    vector<double> scales = {0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005};
    
    for (double scale : scales) {
        for (int iter = 0; iter < iterations / scales.size(); iter++) {
            int i = rng() % c.n;
            double ox = c.x[i], oy = c.y[i], oa = c.a[i];
            
            double r = U(rng);
            if (r < 0.4) {
                c.x[i] += (U(rng) - 0.5) * scale * 2;
                c.y[i] += (U(rng) - 0.5) * scale * 2;
            } else if (r < 0.8) {
                c.a[i] += (U(rng) - 0.5) * scale * 90;
            } else {
                c.x[i] += (U(rng) - 0.5) * scale;
                c.y[i] += (U(rng) - 0.5) * scale;
                c.a[i] += (U(rng) - 0.5) * scale * 45;
            }
            c.upd(i);
            
            bool hit = false;
            for (int j = 0; j < c.n; j++) if (i != j && overlap(c.pl[i], c.pl[j])) { hit = true; break; }
            
            if (!hit && c.side() <= best_s + 1e-15) {
                if (c.side() < best_s - 1e-15) best_s = c.side();
            } else {
                c.x[i] = ox; c.y[i] = oy; c.a[i] = oa; c.upd(i);
            }
        }
    }
    return best_s;
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
    
    // Focus on N=22 and N=24
    vector<int> target_n = {22, 24};
    
    for (int n : target_n) {
        cout << "\n=== N=" << n << " ===" << endl;
        cout << "Current side: " << baseline[n].side() << ", score: " << baseline[n].score() << endl;
    }
    
    map<int, Cfg> best = baseline;
    int num_restarts = 1000;  // Many random restarts
    int local_iterations = 50000;  // Intensive local search
    
    cout << "\nRunning " << num_restarts << " asymmetric restarts for N=22 and N=24..." << endl;
    
    int improvements = 0;
    
    #pragma omp parallel
    {
        mt19937_64 rng(random_device{}() + omp_get_thread_num());
        
        #pragma omp for schedule(dynamic)
        for (int restart = 0; restart < num_restarts; restart++) {
            for (int n : target_n) {
                // Generate asymmetric starting configuration
                Cfg c = generateAsymmetric(n, rng);
                
                // Repair overlaps
                if (!repairOverlaps(c, rng)) continue;
                
                // Intensive local search
                localSearch(c, rng, local_iterations);
                
                // Check if better
                #pragma omp critical
                {
                    if (c.check_valid() && c.side() < best[n].side() - 1e-10) {
                        cout << "IMPROVEMENT! N=" << n << ": " << best[n].side() << " -> " << c.side() 
                             << " (saved " << best[n].side() - c.side() << ")" << endl;
                        best[n] = c;
                        improvements++;
                    }
                }
            }
            
            if (restart % 100 == 0) {
                #pragma omp critical
                cout << "Completed restart " << restart << "/" << num_restarts << endl;
            }
        }
    }
    
    // Calculate final score
    double final_total = 0;
    for (auto& [n, c] : best) final_total += c.score();
    
    cout << "\n========================================" << endl;
    cout << "Asymmetric Search Complete" << endl;
    cout << "Baseline Score: " << fixed << setprecision(6) << baseline_total << endl;
    cout << "Final Score:    " << final_total << endl;
    cout << "Improvement:    " << baseline_total - final_total << endl;
    cout << "Improvements found: " << improvements << endl;
    
    for (int n : target_n) {
        cout << "N=" << n << ": " << baseline[n].side() << " -> " << best[n].side() 
             << " (score: " << baseline[n].score() << " -> " << best[n].score() << ")" << endl;
    }
    cout << "========================================" << endl;
    
    saveCSV("/home/submission/submission.csv", best);
    cout << "Saved to /home/submission/submission.csv" << endl;
    
    return 0;
}
