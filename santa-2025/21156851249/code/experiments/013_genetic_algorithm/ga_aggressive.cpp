// Aggressive Genetic Algorithm with Topology Crossover
// Focus on specific N values with more compute

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

struct Tree { double x, y, angle; Poly poly; };

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
        if ((poly.p[i].y <= pt.y && pt.y < poly.p[j].y) || (poly.p[j].y <= pt.y && pt.y < poly.p[i].y)) {
            double x_int = poly.p[i].x + (pt.y - poly.p[i].y) / (poly.p[j].y - poly.p[i].y) * (poly.p[j].x - poly.p[i].x);
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
    return (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)));
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

struct Individual {
    vector<Tree> trees;
    double fitness;
    void evaluate() {
        if (hasOverlap(trees)) fitness = -1e9;
        else fitness = -calculateSide(trees);
    }
};

// Generate a completely random configuration
Individual generateRandom(int N, mt19937& rng) {
    Individual ind;
    ind.trees.resize(N);
    
    uniform_real_distribution<double> pos_dist(-5.0, 5.0);
    uniform_real_distribution<double> angle_dist(0.0, 360.0);
    
    for (int i = 0; i < N; i++) {
        ind.trees[i].x = pos_dist(rng);
        ind.trees[i].y = pos_dist(rng);
        ind.trees[i].angle = angle_dist(rng);
        updatePoly(ind.trees[i]);
    }
    
    return ind;
}

// Greedy placement from scratch
Individual greedyConstruct(int N, mt19937& rng) {
    Individual ind;
    uniform_real_distribution<double> angle_dist(0.0, 360.0);
    
    // Place first tree at origin
    Tree first;
    first.x = 0; first.y = 0;
    first.angle = angle_dist(rng);
    updatePoly(first);
    ind.trees.push_back(first);
    
    // Place remaining trees
    for (int i = 1; i < N; i++) {
        Tree best;
        double best_side = 1e18;
        
        // Try many positions
        for (int attempt = 0; attempt < 100; attempt++) {
            Tree t;
            // Place near existing trees
            uniform_int_distribution<int> tree_dist(0, ind.trees.size() - 1);
            int ref = tree_dist(rng);
            uniform_real_distribution<double> offset(-1.0, 1.0);
            t.x = ind.trees[ref].x + offset(rng);
            t.y = ind.trees[ref].y + offset(rng);
            t.angle = angle_dist(rng);
            updatePoly(t);
            
            // Check overlap
            bool overlaps = false;
            for (const auto& existing : ind.trees) {
                if (overlap(t.poly, existing.poly)) {
                    overlaps = true;
                    break;
                }
            }
            
            if (!overlaps) {
                // Calculate side with this tree
                ind.trees.push_back(t);
                double side = calculateSide(ind.trees);
                ind.trees.pop_back();
                
                if (side < best_side) {
                    best_side = side;
                    best = t;
                }
            }
        }
        
        if (best_side < 1e17) {
            ind.trees.push_back(best);
        } else {
            // Fallback: place anywhere
            for (int attempt = 0; attempt < 1000; attempt++) {
                Tree t;
                uniform_real_distribution<double> pos(-10.0, 10.0);
                t.x = pos(rng);
                t.y = pos(rng);
                t.angle = angle_dist(rng);
                updatePoly(t);
                
                bool overlaps = false;
                for (const auto& existing : ind.trees) {
                    if (overlap(t.poly, existing.poly)) {
                        overlaps = true;
                        break;
                    }
                }
                
                if (!overlaps) {
                    ind.trees.push_back(t);
                    break;
                }
            }
        }
    }
    
    centerTrees(ind.trees);
    ind.evaluate();
    return ind;
}

void localOptimize(Individual& ind, mt19937& rng, int iterations) {
    if (ind.trees.empty()) return;
    
    double current_side = calculateSide(ind.trees);
    double best_side = current_side;
    vector<Tree> best_trees = ind.trees;
    
    double temp = 0.1;
    double cooling = 0.9995;
    
    uniform_real_distribution<double> dist01(0.0, 1.0);
    uniform_int_distribution<int> tree_dist(0, ind.trees.size() - 1);
    
    for (int iter = 0; iter < iterations; iter++) {
        int idx = tree_dist(rng);
        Tree old_tree = ind.trees[idx];
        
        double scale = temp * 0.1;
        ind.trees[idx].x += (dist01(rng) - 0.5) * scale;
        ind.trees[idx].y += (dist01(rng) - 0.5) * scale;
        ind.trees[idx].angle += (dist01(rng) - 0.5) * scale * 20;
        updatePoly(ind.trees[idx]);
        
        bool has_overlap = false;
        for (int j = 0; j < (int)ind.trees.size(); j++) {
            if (j != idx && overlap(ind.trees[idx].poly, ind.trees[j].poly)) {
                has_overlap = true;
                break;
            }
        }
        
        if (has_overlap) {
            ind.trees[idx] = old_tree;
        } else {
            double new_side = calculateSide(ind.trees);
            if (new_side < current_side || dist01(rng) < exp((current_side - new_side) / temp)) {
                current_side = new_side;
                if (new_side < best_side) {
                    best_side = new_side;
                    best_trees = ind.trees;
                }
            } else {
                ind.trees[idx] = old_tree;
            }
        }
        
        temp *= cooling;
    }
    
    ind.trees = best_trees;
    ind.evaluate();
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
        Tree t;
        t.x = stod(x_str.substr(1));
        t.y = stod(y_str.substr(1));
        t.angle = stod(deg_str.substr(1));
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
    string output_file = "/home/code/experiments/013_genetic_algorithm/improved.csv";
    
    cout << "=== Aggressive GA with Greedy Construction ===" << endl;
    
    auto baseline = readBaseline(baseline_file);
    double baseline_score = calculateScore(baseline);
    cout << "Baseline score: " << fixed << setprecision(6) << baseline_score << endl;
    
    auto improved = baseline;
    int num_improved = 0;
    
    // Focus on small-medium N values where there might be more room
    vector<int> target_ns;
    for (int n = 2; n <= 50; n++) target_ns.push_back(n);
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)target_ns.size(); i++) {
        int N = target_ns[i];
        mt19937 rng(42 + N);
        
        double baseline_side = calculateSide(baseline[N]);
        double best_side = baseline_side;
        vector<Tree> best_trees = baseline[N];
        
        // Try many greedy constructions
        for (int trial = 0; trial < 100; trial++) {
            Individual ind = greedyConstruct(N, rng);
            
            if ((int)ind.trees.size() == N && !hasOverlap(ind.trees)) {
                localOptimize(ind, rng, 10000);
                centerTrees(ind.trees);
                
                if (!hasOverlap(ind.trees)) {
                    double side = calculateSide(ind.trees);
                    if (side < best_side) {
                        best_side = side;
                        best_trees = ind.trees;
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
