// Optimize lattice vectors for large N values
// Start from the structure found in baseline solutions and try to improve

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <omp.h>

using namespace std;

const double PI = 3.14159265358979323846;

// CORRECT Tree polygon vertices (15 points)
const int NV = 15;
const double TX[NV] = {0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125};
const double TY[NV] = {0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5};

struct Point { double x, y; };

struct Poly {
    Point p[NV];
    double x0, x1, y0, y1;
    void bbox() {
        x0 = y0 = 1e18; x1 = y1 = -1e18;
        for (int i = 0; i < NV; i++) {
            x0 = min(x0, p[i].x); x1 = max(x1, p[i].x);
            y0 = min(y0, p[i].y); y1 = max(y1, p[i].y);
        }
    }
};

struct Tree {
    double x, y, angle;
    Poly poly;
};

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

double cross(double ax, double ay, double bx, double by) { return ax * by - ay * bx; }

bool contains(const Poly& poly, Point pt) {
    int cnt = 0;
    for (int i = 0; i < NV; i++) {
        int j = (i + 1) % NV;
        double y1 = poly.p[i].y, y2 = poly.p[j].y;
        double x1 = poly.p[i].x, x2 = poly.p[j].x;
        if ((y1 <= pt.y && pt.y < y2) || (y2 <= pt.y && pt.y < y1)) {
            double x_int = x1 + (pt.y - y1) / (y2 - y1) * (x2 - x1);
            if (pt.x < x_int) cnt++;
        }
    }
    return cnt % 2 == 1;
}

bool segIntersect(Point a, Point b, Point c, Point d) {
    double d1 = cross(d.x-c.x, d.y-c.y, a.x-c.x, a.y-c.y);
    double d2 = cross(d.x-c.x, d.y-c.y, b.x-c.x, b.y-c.y);
    double d3 = cross(b.x-a.x, b.y-a.y, c.x-a.x, c.y-a.y);
    double d4 = cross(b.x-a.x, b.y-a.y, d.x-a.x, d.y-a.y);
    return (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
            ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)));
}

bool overlap(const Poly& a, const Poly& b) {
    if (a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0) return false;
    for (int i = 0; i < NV; i++) {
        int i2 = (i + 1) % NV;
        for (int j = 0; j < NV; j++) {
            int j2 = (j + 1) % NV;
            if (segIntersect(a.p[i], a.p[i2], b.p[j], b.p[j2])) return true;
        }
    }
    return contains(a, b.p[0]) || contains(b, a.p[0]);
}

void updatePoly(Tree& tree) { tree.poly = getPoly(tree.x, tree.y, tree.angle); }

double calculateSide(const vector<Tree>& trees) {
    double minX = 1e18, maxX = -1e18, minY = 1e18, maxY = -1e18;
    for (const auto& tree : trees) {
        minX = min(minX, tree.poly.x0); maxX = max(maxX, tree.poly.x1);
        minY = min(minY, tree.poly.y0); maxY = max(maxY, tree.poly.y1);
    }
    return max(maxX - minX, maxY - minY);
}

bool hasOverlap(const vector<Tree>& trees) {
    int n = trees.size();
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (overlap(trees[i].poly, trees[j].poly)) return true;
    return false;
}

// Generate lattice packing with optimized parameters
vector<Tree> generateOptimizedLattice(int N, double ux, double uy, double vx, double vy,
                                       double base_angle, double offset_x, double offset_y) {
    vector<Tree> trees;
    int grid_size = (int)ceil(sqrt(N * 2.0)) + 3;
    
    vector<Tree> candidates;
    for (int i = -grid_size; i <= grid_size; i++) {
        for (int j = -grid_size; j <= grid_size; j++) {
            Tree t;
            t.x = i * ux + j * vx + offset_x;
            t.y = i * uy + j * vy + offset_y;
            t.angle = base_angle + ((i + j) % 2 == 0 ? 0.0 : 180.0);
            updatePoly(t);
            candidates.push_back(t);
        }
    }
    
    // Sort by distance from center
    sort(candidates.begin(), candidates.end(), [](const Tree& a, const Tree& b) {
        return a.x*a.x + a.y*a.y < b.x*b.x + b.y*b.y;
    });
    
    // Select N trees that don't overlap
    for (const auto& cand : candidates) {
        if ((int)trees.size() >= N) break;
        bool overlaps = false;
        for (const auto& existing : trees) {
            if (overlap(cand.poly, existing.poly)) { overlaps = true; break; }
        }
        if (!overlaps) trees.push_back(cand);
    }
    
    return trees;
}

void centerTrees(vector<Tree>& trees) {
    if (trees.empty()) return;
    double minX = 1e18, maxX = -1e18, minY = 1e18, maxY = -1e18;
    for (const auto& tree : trees) {
        minX = min(minX, tree.poly.x0); maxX = max(maxX, tree.poly.x1);
        minY = min(minY, tree.poly.y0); maxY = max(maxY, tree.poly.y1);
    }
    double cx = (minX + maxX) / 2.0, cy = (minY + maxY) / 2.0;
    for (auto& tree : trees) { tree.x -= cx; tree.y -= cy; updatePoly(tree); }
}

// Optimize lattice vectors using gradient-free optimization
double optimizeLatticeVectors(int N, double& ux, double& uy, double& vx, double& vy,
                               double& base_angle, mt19937& rng) {
    uniform_real_distribution<double> dist(-1.0, 1.0);
    
    double best_side = 1e18;
    double best_ux = ux, best_uy = uy, best_vx = vx, best_vy = vy, best_angle = base_angle;
    
    // Try to generate initial valid packing
    auto trees = generateOptimizedLattice(N, ux, uy, vx, vy, base_angle, 0, 0);
    if ((int)trees.size() >= N) {
        centerTrees(trees);
        if (!hasOverlap(trees)) {
            best_side = calculateSide(trees);
        }
    }
    
    // Simulated annealing on lattice parameters
    double temp = 0.1;
    double cooling = 0.999;
    
    for (int iter = 0; iter < 10000; iter++) {
        double new_ux = ux + dist(rng) * temp * 0.1;
        double new_uy = uy + dist(rng) * temp * 0.1;
        double new_vx = vx + dist(rng) * temp * 0.1;
        double new_vy = vy + dist(rng) * temp * 0.1;
        double new_angle = base_angle + dist(rng) * temp * 10;
        
        auto trees = generateOptimizedLattice(N, new_ux, new_uy, new_vx, new_vy, new_angle, 0, 0);
        if ((int)trees.size() >= N) {
            centerTrees(trees);
            if (!hasOverlap(trees)) {
                double side = calculateSide(trees);
                if (side < best_side) {
                    best_side = side;
                    best_ux = new_ux; best_uy = new_uy;
                    best_vx = new_vx; best_vy = new_vy;
                    best_angle = new_angle;
                    ux = new_ux; uy = new_uy;
                    vx = new_vx; vy = new_vy;
                    base_angle = new_angle;
                }
            }
        }
        
        temp *= cooling;
    }
    
    ux = best_ux; uy = best_uy;
    vx = best_vx; vy = best_vy;
    base_angle = best_angle;
    
    return best_side;
}

map<int, vector<Tree>> readBaseline(const string& filename) {
    map<int, vector<Tree>> solutions;
    ifstream file(filename);
    string line;
    getline(file, line);
    
    while (getline(file, line)) {
        stringstream ss(line);
        string id, x_str, y_str, deg_str;
        getline(ss, id, ',');
        getline(ss, x_str, ',');
        getline(ss, y_str, ',');
        getline(ss, deg_str, ',');
        
        int N = stoi(id.substr(0, 3));
        double x = stod(x_str.substr(1));
        double y = stod(y_str.substr(1));
        double deg = stod(deg_str.substr(1));
        
        Tree t; t.x = x; t.y = y; t.angle = deg;
        updatePoly(t);
        solutions[N].push_back(t);
    }
    return solutions;
}

double calculateScore(const map<int, vector<Tree>>& solutions) {
    double total = 0.0;
    for (const auto& [N, trees] : solutions) {
        double side = calculateSide(trees);
        total += (side * side) / N;
    }
    return total;
}

int main() {
    string baseline_file = "/home/submission/submission.csv";
    string output_file = "/home/code/experiments/012_crystalline_lattice/improved.csv";
    
    cout << "=== Optimized Lattice Packing ===" << endl;
    
    auto baseline = readBaseline(baseline_file);
    double baseline_score = calculateScore(baseline);
    cout << "Baseline score: " << fixed << setprecision(6) << baseline_score << endl;
    
    auto improved = baseline;
    int num_improved = 0;
    
    // Process large N values
    #pragma omp parallel for schedule(dynamic)
    for (int N = 58; N <= 200; N++) {
        mt19937 rng(42 + N);
        uniform_real_distribution<double> dist(-1.0, 1.0);
        uniform_real_distribution<double> angle_dist(0.0, 360.0);
        
        double baseline_side = calculateSide(baseline[N]);
        double best_side = baseline_side;
        vector<Tree> best_trees = baseline[N];
        
        // Try many different starting lattice configurations
        for (int trial = 0; trial < 200; trial++) {
            // Random lattice vectors
            double base_mag = 0.4 + dist(rng) * 0.3;
            double angle1 = angle_dist(rng);
            double ux = base_mag * cos(angle1 * PI / 180.0);
            double uy = base_mag * sin(angle1 * PI / 180.0);
            
            double angle2 = angle1 + 60.0 + dist(rng) * 60.0;
            double mag2 = base_mag * (0.8 + dist(rng) * 0.4);
            double vx = mag2 * cos(angle2 * PI / 180.0);
            double vy = mag2 * sin(angle2 * PI / 180.0);
            
            double base_angle = angle_dist(rng);
            
            // Optimize these lattice vectors
            double side = optimizeLatticeVectors(N, ux, uy, vx, vy, base_angle, rng);
            
            if (side < best_side) {
                auto trees = generateOptimizedLattice(N, ux, uy, vx, vy, base_angle, 0, 0);
                if ((int)trees.size() >= N) {
                    centerTrees(trees);
                    if (!hasOverlap(trees)) {
                        #pragma omp critical
                        {
                            if (side < best_side) {
                                best_side = side;
                                best_trees = trees;
                            }
                        }
                    }
                }
            }
        }
        
        #pragma omp critical
        {
            if (best_side < baseline_side - 1e-9) {
                improved[N] = best_trees;
                num_improved++;
                cout << "N=" << N << ": " << baseline_side << " -> " << best_side 
                     << " (improved by " << (baseline_side - best_side) << ")" << endl;
            }
        }
    }
    
    double final_score = calculateScore(improved);
    
    cout << "\n=== Results ===" << endl;
    cout << "Baseline Score: " << fixed << setprecision(6) << baseline_score << endl;
    cout << "Final Score: " << fixed << setprecision(6) << final_score << endl;
    cout << "Improvement: " << fixed << setprecision(6) << (baseline_score - final_score) << endl;
    cout << "N values improved: " << num_improved << endl;
    
    // Write output
    ofstream out(output_file);
    out << "id,x,y,deg" << endl;
    out << fixed << setprecision(20);
    
    for (int N = 1; N <= 200; N++) {
        const auto& trees = improved[N];
        for (size_t i = 0; i < trees.size(); i++) {
            out << setfill('0') << setw(3) << N << "_" << i << ",";
            out << "s" << trees[i].x << ",";
            out << "s" << trees[i].y << ",";
            out << "s" << trees[i].angle << endl;
        }
    }
    
    out.close();
    cout << "Output written to: " << output_file << endl;
    
    return 0;
}
