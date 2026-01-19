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
const double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0.0,0.0,-0.2,-0.2,0.0,0.0,0.25,0.25,0.5,0.5};

struct Tree { double x, y, a; };
struct Config { int n; vector<Tree> trees; double side; };

map<int, Config> configs;
int num_iterations = 1000;
int num_rounds = 16;
string input_file = "submission.csv";
string output_file = "submission.csv";

void get_vertices(const Tree& t, double vx[], double vy[]) {
    double rad = t.a * PI / 180.0;
    double c = cos(rad), s = sin(rad);
    for (int i = 0; i < NV; i++) {
        vx[i] = t.x + TX[i] * c - TY[i] * s;
        vy[i] = t.y + TX[i] * s + TY[i] * c;
    }
}

double get_side(const vector<Tree>& trees) {
    double minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9;
    double vx[NV], vy[NV];
    for (const auto& t : trees) {
        get_vertices(t, vx, vy);
        for (int i = 0; i < NV; i++) {
            minx = min(minx, vx[i]); maxx = max(maxx, vx[i]);
            miny = min(miny, vy[i]); maxy = max(maxy, vy[i]);
        }
    }
    return max(maxx - minx, maxy - miny);
}

bool segments_intersect(double ax1, double ay1, double ax2, double ay2,
                        double bx1, double by1, double bx2, double by2) {
    auto cross = [](double ox, double oy, double ax, double ay, double bx, double by) {
        return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);
    };
    double d1 = cross(bx1, by1, bx2, by2, ax1, ay1);
    double d2 = cross(bx1, by1, bx2, by2, ax2, ay2);
    double d3 = cross(ax1, ay1, ax2, ay2, bx1, by1);
    double d4 = cross(ax1, ay1, ax2, ay2, bx2, by2);
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
        ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)))
        return true;
    return false;
}

bool polygons_overlap(const Tree& t1, const Tree& t2) {
    double vx1[NV], vy1[NV], vx2[NV], vy2[NV];
    get_vertices(t1, vx1, vy1);
    get_vertices(t2, vx2, vy2);
    
    // Check edge intersections
    for (int i = 0; i < NV; i++) {
        int ni = (i + 1) % NV;
        for (int j = 0; j < NV; j++) {
            int nj = (j + 1) % NV;
            if (segments_intersect(vx1[i], vy1[i], vx1[ni], vy1[ni],
                                   vx2[j], vy2[j], vx2[nj], vy2[nj]))
                return true;
        }
    }
    
    // Check if one polygon is inside the other (simplified check)
    // Check centroid of t2 in t1
    double cx2 = 0, cy2 = 0;
    for (int i = 0; i < NV; i++) { cx2 += vx2[i]; cy2 += vy2[i]; }
    cx2 /= NV; cy2 /= NV;
    
    // Ray casting for point in polygon
    int crossings = 0;
    for (int i = 0; i < NV; i++) {
        int ni = (i + 1) % NV;
        if ((vy1[i] <= cy2 && vy1[ni] > cy2) || (vy1[ni] <= cy2 && vy1[i] > cy2)) {
            double t = (cy2 - vy1[i]) / (vy1[ni] - vy1[i]);
            if (cx2 < vx1[i] + t * (vx1[ni] - vx1[i]))
                crossings++;
        }
    }
    if (crossings % 2 == 1) return true;
    
    return false;
}

bool has_any_overlap(const vector<Tree>& trees) {
    for (size_t i = 0; i < trees.size(); i++) {
        for (size_t j = i + 1; j < trees.size(); j++) {
            if (polygons_overlap(trees[i], trees[j]))
                return true;
        }
    }
    return false;
}

void load_csv(const string& filename) {
    ifstream f(filename);
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
        double x = stod(xs.substr(1));
        double y = stod(ys.substr(1));
        double a = stod(ds.substr(1));
        
        configs[n].n = n;
        configs[n].trees.push_back({x, y, a});
    }
    for (auto& [n, cfg] : configs) {
        cfg.side = get_side(cfg.trees);
    }
}

void save_csv(const string& filename) {
    ofstream f(filename);
    f << "id,x,y,deg" << endl;
    f << fixed << setprecision(15);
    for (int n = 1; n <= MAX_N; n++) {
        if (configs.find(n) == configs.end()) continue;
        const auto& trees = configs[n].trees;
        for (size_t i = 0; i < trees.size(); i++) {
            f << setw(3) << setfill('0') << n << "_" << i << ",";
            f << "s" << trees[i].x << ",";
            f << "s" << trees[i].y << ",";
            f << "s" << trees[i].a << endl;
        }
    }
}

void optimize_config(Config& cfg) {
    if (cfg.n <= 1) return;
    
    mt19937 rng(42 + cfg.n);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    uniform_real_distribution<double> angle_dist(-5.0, 5.0);
    
    double best_side = cfg.side;
    vector<Tree> best_trees = cfg.trees;
    
    for (int iter = 0; iter < num_iterations; iter++) {
        // Pick a random tree
        int idx = rng() % cfg.trees.size();
        Tree& t = cfg.trees[idx];
        
        // Save original
        double ox = t.x, oy = t.y, oa = t.a;
        
        // Try small perturbation
        double scale = 0.1 * (1.0 - (double)iter / num_iterations);
        t.x += dist(rng) * scale;
        t.y += dist(rng) * scale;
        t.a += angle_dist(rng) * scale;
        
        // Check if valid
        bool valid = true;
        for (size_t j = 0; j < cfg.trees.size(); j++) {
            if (j != (size_t)idx && polygons_overlap(t, cfg.trees[j])) {
                valid = false;
                break;
            }
        }
        
        if (valid) {
            double new_side = get_side(cfg.trees);
            if (new_side < best_side) {
                best_side = new_side;
                best_trees = cfg.trees;
            } else {
                // Revert with some probability
                if (dist(rng) > 0.1) {
                    t.x = ox; t.y = oy; t.a = oa;
                }
            }
        } else {
            t.x = ox; t.y = oy; t.a = oa;
        }
    }
    
    cfg.trees = best_trees;
    cfg.side = best_side;
}

int main(int argc, char* argv[]) {
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) num_iterations = stoi(argv[++i]);
        else if (arg == "-r" && i + 1 < argc) num_rounds = stoi(argv[++i]);
        else if (arg == "-i" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "-o" && i + 1 < argc) output_file = argv[++i];
    }
    
    cout << "Loading " << input_file << "..." << endl;
    load_csv(input_file);
    
    double initial_score = 0;
    for (const auto& [n, cfg] : configs) {
        initial_score += cfg.side * cfg.side / n;
    }
    cout << "Initial score: " << fixed << setprecision(6) << initial_score << endl;
    
    for (int round = 0; round < num_rounds; round++) {
        cout << "Round " << round + 1 << "/" << num_rounds << endl;
        
        #pragma omp parallel for schedule(dynamic)
        for (int n = 2; n <= MAX_N; n++) {
            if (configs.find(n) != configs.end()) {
                optimize_config(configs[n]);
            }
        }
        
        double score = 0;
        for (const auto& [n, cfg] : configs) {
            score += cfg.side * cfg.side / n;
        }
        cout << "Score after round " << round + 1 << ": " << score << endl;
    }
    
    cout << "Saving to " << output_file << "..." << endl;
    save_csv(output_file);
    
    double final_score = 0;
    for (const auto& [n, cfg] : configs) {
        final_score += cfg.side * cfg.side / n;
    }
    cout << "Final score: " << final_score << endl;
    
    return 0;
}
