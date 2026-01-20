// True Crystalline Lattice Packing for Large N Values
// Key insight: Use PERIODIC positions based on lattice vectors u and v
// This is fundamentally different from random positions with fixed angles

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

// Tree polygon vertices (centered at origin)
const vector<pair<double, double>> TREE_VERTICES = {
    {0.0, 0.5}, {-0.25, 0.0}, {-0.1, 0.0}, {-0.1, -0.5}, {0.1, -0.5}, {0.1, 0.0}, {0.25, 0.0}
};

struct Tree {
    double x, y, angle;
};

// Rotate a point around origin
pair<double, double> rotatePoint(double px, double py, double angle_deg) {
    double angle_rad = angle_deg * PI / 180.0;
    double cos_a = cos(angle_rad);
    double sin_a = sin(angle_rad);
    return {px * cos_a - py * sin_a, px * sin_a + py * cos_a};
}

// Get rotated and translated polygon vertices
vector<pair<double, double>> getTransformedVertices(const Tree& tree) {
    vector<pair<double, double>> result;
    for (const auto& v : TREE_VERTICES) {
        auto rotated = rotatePoint(v.first, v.second, tree.angle);
        result.push_back({rotated.first + tree.x, rotated.second + tree.y});
    }
    return result;
}

// Check if two line segments intersect
bool segmentsIntersect(double x1, double y1, double x2, double y2,
                       double x3, double y3, double x4, double y4) {
    auto cross = [](double ax, double ay, double bx, double by) {
        return ax * by - ay * bx;
    };
    
    double d1 = cross(x4-x3, y4-y3, x1-x3, y1-y3);
    double d2 = cross(x4-x3, y4-y3, x2-x3, y2-y3);
    double d3 = cross(x2-x1, y2-y1, x3-x1, y3-y1);
    double d4 = cross(x2-x1, y2-y1, x4-x1, y4-y1);
    
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
        ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
        return true;
    }
    
    return false;
}

// Point in polygon test
bool pointInPolygon(double px, double py, const vector<pair<double, double>>& poly) {
    int n = poly.size();
    int count = 0;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        double x1 = poly[i].first, y1 = poly[i].second;
        double x2 = poly[j].first, y2 = poly[j].second;
        
        if ((y1 <= py && py < y2) || (y2 <= py && py < y1)) {
            double x_intersect = x1 + (py - y1) / (y2 - y1) * (x2 - x1);
            if (px < x_intersect) count++;
        }
    }
    return count % 2 == 1;
}

// Check if two polygons overlap
bool polygonsOverlap(const vector<pair<double, double>>& poly1,
                     const vector<pair<double, double>>& poly2) {
    // Check edge intersections
    int n1 = poly1.size(), n2 = poly2.size();
    for (int i = 0; i < n1; i++) {
        int i_next = (i + 1) % n1;
        for (int j = 0; j < n2; j++) {
            int j_next = (j + 1) % n2;
            if (segmentsIntersect(poly1[i].first, poly1[i].second,
                                  poly1[i_next].first, poly1[i_next].second,
                                  poly2[j].first, poly2[j].second,
                                  poly2[j_next].first, poly2[j_next].second)) {
                return true;
            }
        }
    }
    
    // Check if one polygon is inside the other
    if (pointInPolygon(poly1[0].first, poly1[0].second, poly2)) return true;
    if (pointInPolygon(poly2[0].first, poly2[0].second, poly1)) return true;
    
    return false;
}

// Check if any trees overlap
bool hasOverlap(const vector<Tree>& trees) {
    int n = trees.size();
    vector<vector<pair<double, double>>> polys(n);
    for (int i = 0; i < n; i++) {
        polys[i] = getTransformedVertices(trees[i]);
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (polygonsOverlap(polys[i], polys[j])) {
                return true;
            }
        }
    }
    return false;
}

// Calculate bounding box side length
double calculateSide(const vector<Tree>& trees) {
    double minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
    for (const auto& tree : trees) {
        auto verts = getTransformedVertices(tree);
        for (const auto& v : verts) {
            minX = min(minX, v.first);
            maxX = max(maxX, v.first);
            minY = min(minY, v.second);
            maxY = max(maxY, v.second);
        }
    }
    return max(maxX - minX, maxY - minY);
}

// Generate lattice packing with given vectors u and v
// Trees are placed at positions i*u + j*v with alternating angles
vector<Tree> generateLatticePacking(int N, double ux, double uy, double vx, double vy,
                                     double base_angle, double offset_x, double offset_y) {
    vector<Tree> trees;
    
    // Estimate grid size needed
    int grid_size = (int)ceil(sqrt(N * 2.0)) + 2;
    
    // Generate all lattice positions
    vector<Tree> candidates;
    for (int i = -grid_size; i <= grid_size; i++) {
        for (int j = -grid_size; j <= grid_size; j++) {
            Tree t;
            t.x = i * ux + j * vx + offset_x;
            t.y = i * uy + j * vy + offset_y;
            // Alternating angles (0 and 180 degrees apart)
            t.angle = base_angle + ((i + j) % 2 == 0 ? 0.0 : 180.0);
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
        
        // Check overlap with existing trees
        auto cand_verts = getTransformedVertices(cand);
        bool overlaps = false;
        for (const auto& existing : trees) {
            auto existing_verts = getTransformedVertices(existing);
            if (polygonsOverlap(cand_verts, existing_verts)) {
                overlaps = true;
                break;
            }
        }
        
        if (!overlaps) {
            trees.push_back(cand);
        }
    }
    
    return trees;
}

// Center the trees to minimize bounding box
void centerTrees(vector<Tree>& trees) {
    if (trees.empty()) return;
    
    double minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
    for (const auto& tree : trees) {
        auto verts = getTransformedVertices(tree);
        for (const auto& v : verts) {
            minX = min(minX, v.first);
            maxX = max(maxX, v.first);
            minY = min(minY, v.second);
            maxY = max(maxY, v.second);
        }
    }
    
    double cx = (minX + maxX) / 2.0;
    double cy = (minY + maxY) / 2.0;
    
    for (auto& tree : trees) {
        tree.x -= cx;
        tree.y -= cy;
    }
}

// Local optimization using simulated annealing
void localOptimize(vector<Tree>& trees, int iterations, mt19937& rng) {
    if (trees.empty()) return;
    
    double current_side = calculateSide(trees);
    double best_side = current_side;
    vector<Tree> best_trees = trees;
    
    double temp = 0.1;
    double cooling = 0.9999;
    
    uniform_real_distribution<double> dist01(0.0, 1.0);
    uniform_int_distribution<int> tree_dist(0, trees.size() - 1);
    
    for (int iter = 0; iter < iterations; iter++) {
        int idx = tree_dist(rng);
        Tree old_tree = trees[idx];
        
        // Small perturbation
        double scale = temp * 0.1;
        trees[idx].x += (dist01(rng) - 0.5) * scale;
        trees[idx].y += (dist01(rng) - 0.5) * scale;
        trees[idx].angle += (dist01(rng) - 0.5) * scale * 10;
        
        if (hasOverlap(trees)) {
            trees[idx] = old_tree;
        } else {
            double new_side = calculateSide(trees);
            if (new_side < current_side || dist01(rng) < exp((current_side - new_side) / temp)) {
                current_side = new_side;
                if (new_side < best_side) {
                    best_side = new_side;
                    best_trees = trees;
                }
            } else {
                trees[idx] = old_tree;
            }
        }
        
        temp *= cooling;
    }
    
    trees = best_trees;
}

// Read baseline solution
map<int, vector<Tree>> readBaseline(const string& filename) {
    map<int, vector<Tree>> solutions;
    ifstream file(filename);
    string line;
    getline(file, line); // Skip header
    
    while (getline(file, line)) {
        stringstream ss(line);
        string id, x_str, y_str, deg_str;
        getline(ss, id, ',');
        getline(ss, x_str, ',');
        getline(ss, y_str, ',');
        getline(ss, deg_str, ',');
        
        // Parse N from id (format: NNN_i)
        int N = stoi(id.substr(0, 3));
        
        // Remove 's' prefix
        double x = stod(x_str.substr(1));
        double y = stod(y_str.substr(1));
        double deg = stod(deg_str.substr(1));
        
        solutions[N].push_back({x, y, deg});
    }
    
    return solutions;
}

// Calculate score for a solution
double calculateScore(const map<int, vector<Tree>>& solutions) {
    double total = 0.0;
    for (const auto& [N, trees] : solutions) {
        double side = calculateSide(trees);
        total += (side * side) / N;
    }
    return total;
}

int main(int argc, char* argv[]) {
    string baseline_file = "/home/code/exploration/datasets/santa-2025.csv";
    string output_file = "/home/code/experiments/012_crystalline_lattice/improved.csv";
    int min_n = 58;  // Start from N=58 (large N where lattice should work better)
    int max_n = 200;
    int num_lattice_configs = 50;  // Number of different lattice configurations to try
    int local_opt_iters = 5000;
    
    cout << "=== Crystalline Lattice Packing ===" << endl;
    cout << "Target N range: " << min_n << " to " << max_n << endl;
    cout << "Lattice configurations per N: " << num_lattice_configs << endl;
    
    // Read baseline
    auto baseline = readBaseline(baseline_file);
    double baseline_score = calculateScore(baseline);
    cout << "Baseline score: " << fixed << setprecision(6) << baseline_score << endl;
    
    // Copy baseline to output
    auto improved = baseline;
    
    int num_improved = 0;
    double total_improvement = 0.0;
    
    // Process each N value
    #pragma omp parallel for schedule(dynamic)
    for (int N = min_n; N <= max_n; N++) {
        mt19937 rng(42 + N);
        uniform_real_distribution<double> dist(-1.0, 1.0);
        uniform_real_distribution<double> angle_dist(0.0, 360.0);
        
        double baseline_side = calculateSide(baseline[N]);
        double best_side = baseline_side;
        vector<Tree> best_trees = baseline[N];
        
        // Try different lattice configurations
        for (int config = 0; config < num_lattice_configs; config++) {
            // Generate random lattice vectors
            // The key is to find vectors that tile the plane efficiently
            
            // Base lattice vector magnitudes (related to tree size)
            double base_mag = 0.6 + dist(rng) * 0.2;  // Around 0.4 to 0.8
            
            // First lattice vector
            double angle1 = angle_dist(rng);
            double ux = base_mag * cos(angle1 * PI / 180.0);
            double uy = base_mag * sin(angle1 * PI / 180.0);
            
            // Second lattice vector (not parallel to first)
            double angle2 = angle1 + 60.0 + dist(rng) * 60.0;  // 60-120 degrees apart
            double mag2 = base_mag * (0.8 + dist(rng) * 0.4);
            double vx = mag2 * cos(angle2 * PI / 180.0);
            double vy = mag2 * sin(angle2 * PI / 180.0);
            
            // Base angle for trees
            double base_angle = angle_dist(rng);
            
            // Offset
            double offset_x = dist(rng) * 0.1;
            double offset_y = dist(rng) * 0.1;
            
            // Generate lattice packing
            auto trees = generateLatticePacking(N, ux, uy, vx, vy, base_angle, offset_x, offset_y);
            
            if ((int)trees.size() < N) continue;  // Couldn't place all trees
            
            // Center and optimize
            centerTrees(trees);
            localOptimize(trees, local_opt_iters, rng);
            centerTrees(trees);
            
            // Check for overlaps
            if (hasOverlap(trees)) continue;
            
            double side = calculateSide(trees);
            
            #pragma omp critical
            {
                if (side < best_side) {
                    best_side = side;
                    best_trees = trees;
                }
            }
        }
        
        // Also try optimizing the baseline with lattice-inspired moves
        // (shift all trees by lattice vector, then optimize)
        for (int config = 0; config < 10; config++) {
            vector<Tree> trees = baseline[N];
            
            // Apply small lattice-like shift to all trees
            double shift_x = dist(rng) * 0.05;
            double shift_y = dist(rng) * 0.05;
            for (auto& t : trees) {
                t.x += shift_x;
                t.y += shift_y;
            }
            
            centerTrees(trees);
            localOptimize(trees, local_opt_iters, rng);
            centerTrees(trees);
            
            if (!hasOverlap(trees)) {
                double side = calculateSide(trees);
                #pragma omp critical
                {
                    if (side < best_side) {
                        best_side = side;
                        best_trees = trees;
                    }
                }
            }
        }
        
        #pragma omp critical
        {
            if (best_side < baseline_side - 1e-9) {
                improved[N] = best_trees;
                double improvement = baseline_side - best_side;
                total_improvement += improvement;
                num_improved++;
                cout << "N=" << N << ": " << baseline_side << " -> " << best_side 
                     << " (improved by " << improvement << ")" << endl;
            }
        }
    }
    
    // Calculate final score
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
