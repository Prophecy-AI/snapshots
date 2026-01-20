// Exhaustive search for small N values (N=2, 3, 4, 5)
// Uses fine-grained angle search with position optimization

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <random>
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

// Optimize positions for given angles using gradient-free optimization
double optimizePositions(vector<Tree>& trees, mt19937& rng, int iterations = 5000) {
    double best_side = calculateSide(trees);
    vector<Tree> best_trees = trees;
    
    double temp = 0.5;
    double cooling = 0.999;
    
    uniform_real_distribution<double> dist01(0.0, 1.0);
    uniform_int_distribution<int> tree_dist(0, trees.size() - 1);
    
    for (int iter = 0; iter < iterations; iter++) {
        int idx = tree_dist(rng);
        Tree old_tree = trees[idx];
        
        double scale = temp * 0.2;
        trees[idx].x += (dist01(rng) - 0.5) * scale;
        trees[idx].y += (dist01(rng) - 0.5) * scale;
        updatePoly(trees[idx]);
        
        bool has_overlap = false;
        for (int j = 0; j < (int)trees.size(); j++) {
            if (j != idx && overlap(trees[idx].poly, trees[j].poly)) {
                has_overlap = true;
                break;
            }
        }
        
        if (has_overlap) {
            trees[idx] = old_tree;
        } else {
            double new_side = calculateSide(trees);
            if (new_side < best_side) {
                best_side = new_side;
                best_trees = trees;
            } else if (dist01(rng) < exp((best_side - new_side) / temp)) {
                // Accept worse solution with probability
            } else {
                trees[idx] = old_tree;
            }
        }
        
        temp *= cooling;
    }
    
    trees = best_trees;
    return best_side;
}

// Search for optimal N=2 configuration
pair<double, vector<Tree>> searchN2(mt19937& rng) {
    double best_side = 1e9;
    vector<Tree> best_trees(2);
    
    // Search all angle pairs (double-lattice: angle2 = angle1 + 180)
    for (double angle1 = 0; angle1 < 180; angle1 += 0.5) {
        double angle2 = angle1 + 180;
        
        // Initialize trees
        vector<Tree> trees(2);
        trees[0].x = -0.3; trees[0].y = 0; trees[0].angle = angle1;
        trees[1].x = 0.3; trees[1].y = 0; trees[1].angle = angle2;
        for (auto& t : trees) updatePoly(t);
        
        // Optimize positions
        double side = optimizePositions(trees, rng, 3000);
        centerTrees(trees);
        
        if (!hasOverlap(trees) && side < best_side) {
            best_side = side;
            best_trees = trees;
        }
    }
    
    // Fine-tune around best
    double best_angle1 = best_trees[0].angle;
    for (double angle1 = best_angle1 - 2; angle1 < best_angle1 + 2; angle1 += 0.1) {
        double angle2 = angle1 + 180;
        
        vector<Tree> trees(2);
        trees[0].x = best_trees[0].x; trees[0].y = best_trees[0].y; trees[0].angle = angle1;
        trees[1].x = best_trees[1].x; trees[1].y = best_trees[1].y; trees[1].angle = angle2;
        for (auto& t : trees) updatePoly(t);
        
        double side = optimizePositions(trees, rng, 5000);
        centerTrees(trees);
        
        if (!hasOverlap(trees) && side < best_side) {
            best_side = side;
            best_trees = trees;
        }
    }
    
    return {best_side, best_trees};
}

// Search for optimal N=3 configuration
pair<double, vector<Tree>> searchN3(mt19937& rng) {
    double best_side = 1e9;
    vector<Tree> best_trees(3);
    
    // Try various angle combinations
    for (double angle1 = 0; angle1 < 180; angle1 += 5) {
        for (double angle2 = angle1; angle2 < 360; angle2 += 5) {
            for (double angle3 = angle2; angle3 < 360; angle3 += 5) {
                vector<Tree> trees(3);
                trees[0].x = -0.5; trees[0].y = 0; trees[0].angle = angle1;
                trees[1].x = 0.5; trees[1].y = 0; trees[1].angle = angle2;
                trees[2].x = 0; trees[2].y = 0.5; trees[2].angle = angle3;
                for (auto& t : trees) updatePoly(t);
                
                double side = optimizePositions(trees, rng, 2000);
                centerTrees(trees);
                
                if (!hasOverlap(trees) && side < best_side) {
                    best_side = side;
                    best_trees = trees;
                }
            }
        }
    }
    
    return {best_side, best_trees};
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
    string output_file = "/home/code/experiments/014_exhaustive_small_n/improved.csv";
    
    cout << "=== Exhaustive Search for Small N ===" << endl;
    
    auto baseline = readBaseline(baseline_file);
    double baseline_score = calculateScore(baseline);
    cout << "Baseline score: " << fixed << setprecision(6) << baseline_score << endl;
    
    auto improved = baseline;
    int num_improved = 0;
    
    mt19937 rng(42);
    
    // Search N=2
    cout << "\nSearching N=2..." << endl;
    double baseline_side_2 = calculateSide(baseline[2]);
    cout << "  Baseline side: " << baseline_side_2 << endl;
    
    auto [side_2, trees_2] = searchN2(rng);
    cout << "  Best found side: " << side_2 << endl;
    
    if (side_2 < baseline_side_2 - 1e-9) {
        improved[2] = trees_2;
        num_improved++;
        cout << "  IMPROVED! " << baseline_side_2 << " -> " << side_2 << endl;
    }
    
    // Search N=3
    cout << "\nSearching N=3..." << endl;
    double baseline_side_3 = calculateSide(baseline[3]);
    cout << "  Baseline side: " << baseline_side_3 << endl;
    
    auto [side_3, trees_3] = searchN3(rng);
    cout << "  Best found side: " << side_3 << endl;
    
    if (side_3 < baseline_side_3 - 1e-9) {
        improved[3] = trees_3;
        num_improved++;
        cout << "  IMPROVED! " << baseline_side_3 << " -> " << side_3 << endl;
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
