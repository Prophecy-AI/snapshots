// Random Restart Optimizer for Santa 2025
// Generates random starting configurations and optimizes them
// to explore different basins of attraction

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

// Generate random starting configuration
Cfg generateRandom(int n, mt19937_64& rng) {
    Cfg c;
    c.n = n;
    
    // Estimate reasonable area based on n
    double area_per_tree = 0.5;  // Approximate area per tree
    double total_area = n * area_per_tree;
    double side_estimate = sqrt(total_area) * 1.5;  // Add some slack
    
    uniform_real_distribution<double> pos_dist(-side_estimate/2, side_estimate/2);
    uniform_real_distribution<double> angle_dist(0, 360);
    
    for (int i = 0; i < n; i++) {
        c.x[i] = pos_dist(rng);
        c.y[i] = pos_dist(rng);
        c.a[i] = angle_dist(rng);
    }
    c.updAll();
    return c;
}

// Simple SA optimization
double optimize(Cfg& cur, int iterations, mt19937_64& rng) {
    uniform_real_distribution<double> U(0, 1);
    double best_s = cur.side();
    
    vector<double> scales = {0.5, 0.1, 0.05, 0.01, 0.005, 0.001};
    
    for (double scale : scales) {
        for (int it = 0; it < iterations / scales.size(); it++) {
            int i = rng() % cur.n;
            double ox = cur.x[i], oy = cur.y[i], oa = cur.a[i];
            
            double r = U(rng);
            if (r < 0.4) {
                cur.x[i] += (U(rng) - 0.5) * scale * 2;
                cur.y[i] += (U(rng) - 0.5) * scale * 2;
            } else if (r < 0.8) {
                cur.a[i] += (U(rng) - 0.5) * scale * 90;
            } else {
                cur.x[i] += (U(rng) - 0.5) * scale;
                cur.y[i] += (U(rng) - 0.5) * scale;
                cur.a[i] += (U(rng) - 0.5) * scale * 45;
            }
            
            cur.upd(i);
            bool hit = false;
            for (int j = 0; j < cur.n; j++) if (i != j && overlap(cur.pl[i], cur.pl[j])) { hit = true; break; }
            
            if (!hit && cur.side() <= best_s + 1e-15) {
                if (cur.side() < best_s - 1e-15) best_s = cur.side();
            } else {
                cur.x[i] = ox; cur.y[i] = oy; cur.a[i] = oa; cur.upd(i);
            }
        }
    }
    return best_s;
}

// Repair overlaps by pushing trees apart
bool repairOverlaps(Cfg& c, mt19937_64& rng, int max_attempts = 1000) {
    uniform_real_distribution<double> U(0, 1);
    
    for (int attempt = 0; attempt < max_attempts; attempt++) {
        bool has_overlap = false;
        for (int i = 0; i < c.n && !has_overlap; i++) {
            for (int j = i + 1; j < c.n; j++) {
                if (overlap(c.pl[i], c.pl[j])) {
                    has_overlap = true;
                    // Push trees apart
                    double dx = c.x[j] - c.x[i];
                    double dy = c.y[j] - c.y[i];
                    double dist = sqrt(dx*dx + dy*dy);
                    if (dist < 0.01) { dx = U(rng) - 0.5; dy = U(rng) - 0.5; dist = sqrt(dx*dx + dy*dy); }
                    double push = 0.1;
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

int main(int argc, char** argv) {
    int num_restarts = 50;
    int iterations_per_restart = 100000;
    string input_file = "/home/submission/submission.csv";
    string output_file = "/home/submission/submission.csv";
    
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "-r" && i + 1 < argc) num_restarts = stoi(argv[++i]);
        else if (a == "-n" && i + 1 < argc) iterations_per_restart = stoi(argv[++i]);
        else if (a == "-i" && i + 1 < argc) input_file = argv[++i];
        else if (a == "-o" && i + 1 < argc) output_file = argv[++i];
    }
    
    cout << "Loading baseline from " << input_file << "...\n";
    auto baseline = loadCSV(input_file);
    
    double baseline_total = 0;
    for (auto& [n, c] : baseline) baseline_total += c.score();
    cout << "Baseline total score: " << fixed << setprecision(6) << baseline_total << "\n\n";
    
    map<int, Cfg> best = baseline;
    map<int, double> best_scores;
    for (auto& [n, c] : baseline) best_scores[n] = c.score();
    
    cout << "Running " << num_restarts << " random restarts with " << iterations_per_restart << " iterations each...\n";
    cout << "Using " << omp_get_max_threads() << " threads.\n\n";
    
    int improvements_found = 0;
    
    // Focus on small N values (2-20) where improvements are most impactful
    vector<int> target_n;
    for (int n = 2; n <= 20; n++) target_n.push_back(n);
    
    #pragma omp parallel
    {
        mt19937_64 rng(random_device{}() + omp_get_thread_num());
        
        #pragma omp for schedule(dynamic)
        for (int restart = 0; restart < num_restarts; restart++) {
            for (int n : target_n) {
                // Generate random starting configuration
                Cfg c = generateRandom(n, rng);
                
                // Repair overlaps
                if (!repairOverlaps(c, rng)) continue;
                
                // Optimize
                double s = optimize(c, iterations_per_restart, rng);
                
                // Check if better than current best
                #pragma omp critical
                {
                    if (c.check_valid() && c.score() < best_scores[n] - 1e-10) {
                        cout << "IMPROVEMENT! N=" << n << ": " << best_scores[n] << " -> " << c.score() 
                             << " (saved " << best_scores[n] - c.score() << ")\n";
                        best[n] = c;
                        best_scores[n] = c.score();
                        improvements_found++;
                    }
                }
            }
            
            if (restart % 10 == 0) {
                #pragma omp critical
                cout << "Completed restart " << restart << "/" << num_restarts << "\n";
            }
        }
    }
    
    // Calculate final score
    double final_total = 0;
    for (auto& [n, c] : best) final_total += c.score();
    
    cout << "\n========================================\n";
    cout << "Random Restart Complete\n";
    cout << "Baseline Score: " << fixed << setprecision(6) << baseline_total << "\n";
    cout << "Final Score:    " << final_total << "\n";
    cout << "Improvement:    " << baseline_total - final_total << "\n";
    cout << "Improvements found: " << improvements_found << "\n";
    cout << "========================================\n";
    
    saveCSV(output_file, best);
    cout << "Saved to " << output_file << "\n";
    
    return 0;
}
