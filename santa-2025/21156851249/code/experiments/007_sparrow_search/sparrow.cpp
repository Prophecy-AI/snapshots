// Sparrow Search Algorithm (SSA) for Christmas Tree Packing
// Population-based swarm optimization with discoverers and joiners
// Explores different basins of attraction than SA

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

// Sparrow Search Algorithm
class SSA {
public:
    int pop_size;
    double pd_ratio;  // Producer/Discoverer ratio
    double sd_ratio;  // Safety detection ratio
    double st;        // Safety threshold
    mt19937_64 rng;
    
    SSA(int pop = 50, double pd = 0.2, double sd = 0.1, double safety = 0.8) 
        : pop_size(pop), pd_ratio(pd), sd_ratio(sd), st(safety), rng(random_device{}()) {}
    
    Cfg optimize(Cfg initial, int max_iter) {
        uniform_real_distribution<double> U(0, 1);
        normal_distribution<double> N(0, 1);
        
        int n = initial.n;
        vector<Cfg> population(pop_size);
        vector<double> fitness(pop_size);
        
        // Initialize population with perturbations of initial solution
        for (int i = 0; i < pop_size; i++) {
            population[i] = initial;
            // Add random perturbations
            for (int j = 0; j < n; j++) {
                population[i].x[j] += (U(rng) - 0.5) * 0.5;
                population[i].y[j] += (U(rng) - 0.5) * 0.5;
                population[i].a[j] += (U(rng) - 0.5) * 30;
            }
            population[i].updAll();
            
            // Repair overlaps
            repairOverlaps(population[i]);
            
            fitness[i] = population[i].check_valid() ? population[i].side() : 1e9;
        }
        
        // Also include the original solution
        population[0] = initial;
        population[0].updAll();
        fitness[0] = population[0].side();
        
        Cfg best = initial;
        double best_fitness = initial.side();
        
        int num_producers = max(1, (int)(pop_size * pd_ratio));
        int num_scouts = max(1, (int)(pop_size * sd_ratio));
        
        for (int iter = 0; iter < max_iter; iter++) {
            // Sort population by fitness
            vector<int> indices(pop_size);
            iota(indices.begin(), indices.end(), 0);
            sort(indices.begin(), indices.end(), [&](int a, int b) { return fitness[a] < fitness[b]; });
            
            double R2 = U(rng);  // Alarm value
            
            // Update producers (discoverers)
            for (int i = 0; i < num_producers; i++) {
                int idx = indices[i];
                Cfg& sparrow = population[idx];
                
                if (R2 < st) {
                    // Safe environment - explore widely
                    double alpha = U(rng);
                    for (int j = 0; j < n; j++) {
                        sparrow.x[j] *= exp(-iter / (alpha * max_iter + 1e-10));
                        sparrow.y[j] *= exp(-iter / (alpha * max_iter + 1e-10));
                    }
                } else {
                    // Danger - move toward best
                    double Q = N(rng);
                    for (int j = 0; j < n; j++) {
                        sparrow.x[j] += Q;
                        sparrow.y[j] += Q;
                    }
                }
                sparrow.updAll();
            }
            
            // Update scroungers (joiners)
            int worst_idx = indices[pop_size - 1];
            for (int i = num_producers; i < pop_size; i++) {
                int idx = indices[i];
                Cfg& sparrow = population[idx];
                Cfg& best_producer = population[indices[0]];
                
                if (i > pop_size / 2) {
                    // Worse half - move toward best producer
                    double Q = N(rng);
                    for (int j = 0; j < n; j++) {
                        sparrow.x[j] = Q * exp((sparrow.x[j] - best_producer.x[j]) / ((i+1) * (i+1)));
                        sparrow.y[j] = Q * exp((sparrow.y[j] - best_producer.y[j]) / ((i+1) * (i+1)));
                    }
                } else {
                    // Better half - random walk around best
                    double A = (U(rng) > 0.5 ? 1 : -1) * 0.1;
                    for (int j = 0; j < n; j++) {
                        sparrow.x[j] = best_producer.x[j] + abs(sparrow.x[j] - best_producer.x[j]) * A;
                        sparrow.y[j] = best_producer.y[j] + abs(sparrow.y[j] - best_producer.y[j]) * A;
                    }
                }
                sparrow.updAll();
            }
            
            // Update scouts (safety detection)
            for (int i = 0; i < num_scouts; i++) {
                int idx = indices[rng() % pop_size];
                Cfg& sparrow = population[idx];
                
                if (fitness[idx] > fitness[indices[0]]) {
                    // Move toward best
                    double beta = N(rng);
                    for (int j = 0; j < n; j++) {
                        sparrow.x[j] = population[indices[0]].x[j] + beta * abs(sparrow.x[j] - population[indices[0]].x[j]);
                        sparrow.y[j] = population[indices[0]].y[j] + beta * abs(sparrow.y[j] - population[indices[0]].y[j]);
                    }
                } else {
                    // Random exploration
                    double K = (2 * U(rng) - 1) * 0.1;
                    for (int j = 0; j < n; j++) {
                        sparrow.x[j] += K * (sparrow.x[j] - population[worst_idx].x[j]) / (fitness[idx] - fitness[worst_idx] + 1e-10);
                        sparrow.y[j] += K * (sparrow.y[j] - population[worst_idx].y[j]) / (fitness[idx] - fitness[worst_idx] + 1e-10);
                    }
                }
                sparrow.updAll();
            }
            
            // Repair and evaluate
            for (int i = 0; i < pop_size; i++) {
                repairOverlaps(population[i]);
                fitness[i] = population[i].check_valid() ? population[i].side() : 1e9;
                
                if (fitness[i] < best_fitness) {
                    best = population[i];
                    best_fitness = fitness[i];
                }
            }
            
            // Local search on best solution
            localSearch(best);
            if (best.side() < best_fitness) {
                best_fitness = best.side();
            }
        }
        
        return best;
    }
    
    void repairOverlaps(Cfg& c) {
        uniform_real_distribution<double> U(0, 1);
        for (int attempt = 0; attempt < 100; attempt++) {
            bool has_overlap = false;
            for (int i = 0; i < c.n && !has_overlap; i++) {
                for (int j = i + 1; j < c.n; j++) {
                    if (overlap(c.pl[i], c.pl[j])) {
                        has_overlap = true;
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
            if (!has_overlap) break;
        }
    }
    
    void localSearch(Cfg& c) {
        uniform_real_distribution<double> U(0, 1);
        double best_s = c.side();
        
        for (int iter = 0; iter < 1000; iter++) {
            int i = rng() % c.n;
            double ox = c.x[i], oy = c.y[i], oa = c.a[i];
            
            double scale = 0.01;
            c.x[i] += (U(rng) - 0.5) * scale;
            c.y[i] += (U(rng) - 0.5) * scale;
            c.a[i] += (U(rng) - 0.5) * scale * 10;
            c.upd(i);
            
            bool hit = false;
            for (int j = 0; j < c.n; j++) if (i != j && overlap(c.pl[i], c.pl[j])) { hit = true; break; }
            
            if (!hit && c.side() <= best_s) {
                best_s = c.side();
            } else {
                c.x[i] = ox; c.y[i] = oy; c.a[i] = oa; c.upd(i);
            }
        }
    }
};

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
    cout << "Baseline score: " << fixed << setprecision(6) << baseline_total << endl;
    
    map<int, Cfg> best = baseline;
    int improvements = 0;
    
    // Focus on N=2-30 where improvements matter most
    vector<int> target_n;
    for (int n = 2; n <= 30; n++) target_n.push_back(n);
    
    cout << "\nRunning Sparrow Search Algorithm on N=2-30..." << endl;
    cout << "Population: 50, Iterations: 100 per N" << endl;
    
    #pragma omp parallel
    {
        SSA ssa(50, 0.2, 0.1, 0.8);
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < (int)target_n.size(); i++) {
            int n = target_n[i];
            Cfg result = ssa.optimize(baseline[n], 100);
            
            #pragma omp critical
            {
                if (result.check_valid() && result.side() < best[n].side() - 1e-10) {
                    cout << "IMPROVEMENT! N=" << n << ": " << best[n].side() << " -> " << result.side() 
                         << " (saved " << best[n].side() - result.side() << ")" << endl;
                    best[n] = result;
                    improvements++;
                }
            }
        }
    }
    
    double final_total = 0;
    for (auto& [n, c] : best) final_total += c.score();
    
    cout << "\n========================================" << endl;
    cout << "Sparrow Search Complete" << endl;
    cout << "Baseline Score: " << fixed << setprecision(6) << baseline_total << endl;
    cout << "Final Score:    " << final_total << endl;
    cout << "Improvement:    " << baseline_total - final_total << endl;
    cout << "Improvements found: " << improvements << endl;
    cout << "========================================" << endl;
    
    saveCSV("/home/submission/submission.csv", best);
    cout << "Saved to /home/submission/submission.csv" << endl;
    
    return 0;
}
