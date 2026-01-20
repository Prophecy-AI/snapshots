// NFP-Based Constructive Heuristic for Large N
// Builds solutions from scratch using greedy placement
// Explores different topologies than SA-based optimization

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
    bool hasOvl(int i) {
        for (int j = 0; j < n; j++) if (i != j && overlap(pl[i], pl[j])) return true;
        return false;
    }
    double score() { double s = side(); return s * s / n; }
};

// Bottom-left placement heuristic
// Place tree at position that minimizes bounding box while avoiding overlaps
bool tryPlace(Cfg& c, int idx, double angle, double& best_x, double& best_y) {
    double best_side = 1e18;
    bool found = false;
    
    // Get current bounding box
    double cur_xmin = 1e18, cur_xmax = -1e18, cur_ymin = 1e18, cur_ymax = -1e18;
    for (int i = 0; i < idx; i++) {
        cur_xmin = min(cur_xmin, c.pl[i].x0);
        cur_xmax = max(cur_xmax, c.pl[i].x1);
        cur_ymin = min(cur_ymin, c.pl[i].y0);
        cur_ymax = max(cur_ymax, c.pl[i].y1);
    }
    
    // Try positions on a grid around the current bounding box
    double step = 0.1;
    double margin = 2.0;
    
    for (double x = cur_xmin - margin; x <= cur_xmax + margin; x += step) {
        for (double y = cur_ymin - margin; y <= cur_ymax + margin; y += step) {
            c.x[idx] = x;
            c.y[idx] = y;
            c.a[idx] = angle;
            c.upd(idx);
            
            // Check for overlaps with existing trees
            bool has_overlap = false;
            for (int j = 0; j < idx; j++) {
                if (overlap(c.pl[idx], c.pl[j])) {
                    has_overlap = true;
                    break;
                }
            }
            
            if (!has_overlap) {
                // Calculate new bounding box
                double new_xmin = min(cur_xmin, c.pl[idx].x0);
                double new_xmax = max(cur_xmax, c.pl[idx].x1);
                double new_ymin = min(cur_ymin, c.pl[idx].y0);
                double new_ymax = max(cur_ymax, c.pl[idx].y1);
                double new_side = max(new_xmax - new_xmin, new_ymax - new_ymin);
                
                if (new_side < best_side) {
                    best_side = new_side;
                    best_x = x;
                    best_y = y;
                    found = true;
                }
            }
        }
    }
    
    return found;
}

// Construct solution for N trees using greedy placement
Cfg constructSolution(int n, mt19937_64& rng) {
    Cfg c;
    c.n = n;
    
    // Use alternating angles (0 and 180) for double-lattice pattern
    vector<double> angles;
    for (int i = 0; i < n; i++) {
        angles.push_back((i % 2 == 0) ? 0.0 : 180.0);
    }
    
    // Shuffle placement order for variety
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    shuffle(order.begin(), order.end(), rng);
    
    // Place first tree at origin
    c.x[0] = 0;
    c.y[0] = 0;
    c.a[0] = angles[0];
    c.upd(0);
    
    // Place remaining trees using greedy heuristic
    for (int i = 1; i < n; i++) {
        double best_x, best_y;
        if (tryPlace(c, i, angles[i], best_x, best_y)) {
            c.x[i] = best_x;
            c.y[i] = best_y;
            c.a[i] = angles[i];
            c.upd(i);
        } else {
            // Fallback: place at a random position and repair
            uniform_real_distribution<double> U(-5, 5);
            c.x[i] = U(rng);
            c.y[i] = U(rng);
            c.a[i] = angles[i];
            c.upd(i);
        }
    }
    
    return c;
}

// Local optimization after construction
void localOptimize(Cfg& c, mt19937_64& rng, int iterations = 10000) {
    uniform_real_distribution<double> U(0, 1);
    double best_s = c.side();
    
    for (int iter = 0; iter < iterations; iter++) {
        int i = rng() % c.n;
        double ox = c.x[i], oy = c.y[i], oa = c.a[i];
        
        double scale = 0.01 * (1.0 - iter / (double)iterations);
        c.x[i] += (U(rng) - 0.5) * scale;
        c.y[i] += (U(rng) - 0.5) * scale;
        c.upd(i);
        
        bool hit = c.hasOvl(i);
        if (!hit && c.side() <= best_s) {
            best_s = c.side();
        } else {
            c.x[i] = ox; c.y[i] = oy; c.a[i] = oa; c.upd(i);
        }
    }
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
    int improvements = 0;
    
    // Focus on large N values (100-200)
    vector<int> target_n;
    for (int n = 100; n <= 200; n++) target_n.push_back(n);
    
    cout << "\nRunning NFP-based construction for N=100-200..." << endl;
    cout << "Trying 10 random constructions per N..." << endl;
    
    int num_constructions = 10;
    
    #pragma omp parallel
    {
        mt19937_64 rng(random_device{}() + omp_get_thread_num());
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < (int)target_n.size(); i++) {
            int n = target_n[i];
            double baseline_side = baseline[n].side();
            
            for (int trial = 0; trial < num_constructions; trial++) {
                Cfg c = constructSolution(n, rng);
                
                if (c.check_valid()) {
                    localOptimize(c, rng, 5000);
                    
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
            }
        }
    }
    
    // Calculate final score
    double final_total = 0;
    for (auto& [n, c] : best) final_total += c.score();
    
    cout << "\n========================================" << endl;
    cout << "NFP Construction Complete" << endl;
    cout << "Baseline Score: " << fixed << setprecision(6) << baseline_total << endl;
    cout << "Final Score:    " << final_total << endl;
    cout << "Improvement:    " << baseline_total - final_total << endl;
    cout << "N values improved: " << improvements << endl;
    cout << "========================================" << endl;
    
    saveCSV("/home/submission/submission.csv", best);
    cout << "Saved to /home/submission/submission.csv" << endl;
    
    return 0;
}
