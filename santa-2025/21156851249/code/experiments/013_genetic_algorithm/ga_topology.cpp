// Genetic Algorithm with Topology Crossover for Tree Packing
// Key insight: Evolve the ARRANGEMENT of trees, not just positions
// This explores a fundamentally different search space than SA/gradient methods

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

// Tree polygon vertices (15 points)
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

// Individual in the population
struct Individual {
    vector<Tree> trees;
    double fitness;  // negative side (we minimize)
    
    void evaluate() {
        if (hasOverlap(trees)) {
            fitness = -1e9;  // Invalid
        } else {
            fitness = -calculateSide(trees);
        }
    }
};

// Tournament selection
int tournamentSelect(const vector<Individual>& pop, mt19937& rng, int tournament_size = 3) {
    uniform_int_distribution<int> dist(0, pop.size() - 1);
    int best = dist(rng);
    for (int i = 1; i < tournament_size; i++) {
        int candidate = dist(rng);
        if (pop[candidate].fitness > pop[best].fitness) {
            best = candidate;
        }
    }
    return best;
}

// Crossover: Exchange tree subsets between parents
// This is the KEY innovation - we exchange ARRANGEMENTS, not just positions
Individual crossover(const Individual& parent1, const Individual& parent2, mt19937& rng) {
    int n = parent1.trees.size();
    Individual child;
    child.trees.resize(n);
    
    uniform_real_distribution<double> dist01(0.0, 1.0);
    uniform_int_distribution<int> tree_dist(0, n - 1);
    
    // Strategy 1: Uniform crossover - each tree comes from either parent
    for (int i = 0; i < n; i++) {
        if (dist01(rng) < 0.5) {
            child.trees[i] = parent1.trees[i];
        } else {
            child.trees[i] = parent2.trees[i];
        }
        updatePoly(child.trees[i]);
    }
    
    return child;
}

// Crossover: Exchange a contiguous region of trees
Individual crossoverRegion(const Individual& parent1, const Individual& parent2, mt19937& rng) {
    int n = parent1.trees.size();
    Individual child;
    child.trees = parent1.trees;
    
    uniform_int_distribution<int> dist(0, n - 1);
    int start = dist(rng);
    int len = dist(rng) % (n / 2) + 1;  // Exchange up to half the trees
    
    for (int i = 0; i < len; i++) {
        int idx = (start + i) % n;
        child.trees[idx] = parent2.trees[idx];
        updatePoly(child.trees[idx]);
    }
    
    return child;
}

// Mutation: Small perturbation to positions and angles
void mutate(Individual& ind, mt19937& rng, double mutation_rate = 0.1, double scale = 0.05) {
    uniform_real_distribution<double> dist01(0.0, 1.0);
    uniform_real_distribution<double> perturb(-1.0, 1.0);
    
    for (auto& tree : ind.trees) {
        if (dist01(rng) < mutation_rate) {
            tree.x += perturb(rng) * scale;
            tree.y += perturb(rng) * scale;
            tree.angle += perturb(rng) * scale * 20;
            updatePoly(tree);
        }
    }
}

// Mutation: Swap two trees
void mutateSwap(Individual& ind, mt19937& rng) {
    int n = ind.trees.size();
    if (n < 2) return;
    
    uniform_int_distribution<int> dist(0, n - 1);
    int i = dist(rng);
    int j = dist(rng);
    while (j == i) j = dist(rng);
    
    swap(ind.trees[i].x, ind.trees[j].x);
    swap(ind.trees[i].y, ind.trees[j].y);
    updatePoly(ind.trees[i]);
    updatePoly(ind.trees[j]);
}

// Mutation: Rotate a group of trees around their centroid
void mutateRotateGroup(Individual& ind, mt19937& rng) {
    int n = ind.trees.size();
    if (n < 3) return;
    
    uniform_int_distribution<int> dist(0, n - 1);
    uniform_real_distribution<double> angle_dist(-10.0, 10.0);
    
    // Select a random subset of trees
    int group_size = dist(rng) % (n / 3) + 2;
    vector<int> indices;
    for (int i = 0; i < group_size; i++) {
        indices.push_back(dist(rng));
    }
    
    // Calculate centroid
    double cx = 0, cy = 0;
    for (int idx : indices) {
        cx += ind.trees[idx].x;
        cy += ind.trees[idx].y;
    }
    cx /= indices.size();
    cy /= indices.size();
    
    // Rotate around centroid
    double angle = angle_dist(rng) * PI / 180.0;
    double cos_a = cos(angle), sin_a = sin(angle);
    
    for (int idx : indices) {
        double dx = ind.trees[idx].x - cx;
        double dy = ind.trees[idx].y - cy;
        ind.trees[idx].x = cx + dx * cos_a - dy * sin_a;
        ind.trees[idx].y = cy + dx * sin_a + dy * cos_a;
        ind.trees[idx].angle += angle * 180.0 / PI;
        updatePoly(ind.trees[idx]);
    }
}

// Repair: Try to fix overlaps by small perturbations
bool repair(Individual& ind, mt19937& rng, int max_attempts = 100) {
    uniform_real_distribution<double> perturb(-0.1, 0.1);
    
    for (int attempt = 0; attempt < max_attempts; attempt++) {
        bool has_overlap = false;
        for (int i = 0; i < (int)ind.trees.size(); i++) {
            for (int j = i + 1; j < (int)ind.trees.size(); j++) {
                if (overlap(ind.trees[i].poly, ind.trees[j].poly)) {
                    has_overlap = true;
                    // Push trees apart
                    double dx = ind.trees[j].x - ind.trees[i].x;
                    double dy = ind.trees[j].y - ind.trees[i].y;
                    double dist = sqrt(dx*dx + dy*dy) + 1e-6;
                    double push = 0.02;
                    ind.trees[i].x -= push * dx / dist;
                    ind.trees[i].y -= push * dy / dist;
                    ind.trees[j].x += push * dx / dist;
                    ind.trees[j].y += push * dy / dist;
                    updatePoly(ind.trees[i]);
                    updatePoly(ind.trees[j]);
                }
            }
        }
        if (!has_overlap) return true;
    }
    return false;
}

// Local optimization using SA
void localOptimize(Individual& ind, mt19937& rng, int iterations = 1000) {
    double current_side = calculateSide(ind.trees);
    double best_side = current_side;
    vector<Tree> best_trees = ind.trees;
    
    double temp = 0.05;
    double cooling = 0.999;
    
    uniform_real_distribution<double> dist01(0.0, 1.0);
    uniform_int_distribution<int> tree_dist(0, ind.trees.size() - 1);
    
    for (int iter = 0; iter < iterations; iter++) {
        int idx = tree_dist(rng);
        Tree old_tree = ind.trees[idx];
        
        double scale = temp * 0.05;
        ind.trees[idx].x += (dist01(rng) - 0.5) * scale;
        ind.trees[idx].y += (dist01(rng) - 0.5) * scale;
        ind.trees[idx].angle += (dist01(rng) - 0.5) * scale * 10;
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

// Run GA for a single N value
vector<Tree> runGA(int N, const vector<Tree>& initial, mt19937& rng,
                   int pop_size = 50, int generations = 100) {
    // Initialize population
    vector<Individual> population(pop_size);
    
    // First individual is the baseline
    population[0].trees = initial;
    population[0].evaluate();
    
    // Create variations for rest of population
    uniform_real_distribution<double> perturb(-0.1, 0.1);
    for (int i = 1; i < pop_size; i++) {
        population[i].trees = initial;
        // Add random perturbations
        for (auto& tree : population[i].trees) {
            tree.x += perturb(rng) * 0.1;
            tree.y += perturb(rng) * 0.1;
            tree.angle += perturb(rng) * 5;
            updatePoly(tree);
        }
        repair(population[i], rng);
        population[i].evaluate();
    }
    
    // Track best
    Individual best = population[0];
    for (const auto& ind : population) {
        if (ind.fitness > best.fitness) best = ind;
    }
    
    // Evolution loop
    for (int gen = 0; gen < generations; gen++) {
        vector<Individual> new_population;
        
        // Elitism: keep best
        new_population.push_back(best);
        
        // Generate rest of population
        while ((int)new_population.size() < pop_size) {
            // Select parents
            int p1 = tournamentSelect(population, rng);
            int p2 = tournamentSelect(population, rng);
            
            // Crossover
            Individual child;
            uniform_real_distribution<double> dist01(0.0, 1.0);
            if (dist01(rng) < 0.5) {
                child = crossover(population[p1], population[p2], rng);
            } else {
                child = crossoverRegion(population[p1], population[p2], rng);
            }
            
            // Mutation
            if (dist01(rng) < 0.3) mutate(child, rng);
            if (dist01(rng) < 0.2) mutateSwap(child, rng);
            if (dist01(rng) < 0.1) mutateRotateGroup(child, rng);
            
            // Repair and evaluate
            repair(child, rng);
            child.evaluate();
            
            // Local optimization for promising individuals
            if (child.fitness > -1e8 && dist01(rng) < 0.2) {
                localOptimize(child, rng, 500);
            }
            
            new_population.push_back(child);
        }
        
        population = new_population;
        
        // Update best
        for (const auto& ind : population) {
            if (ind.fitness > best.fitness) {
                best = ind;
            }
        }
    }
    
    // Final local optimization on best
    localOptimize(best, rng, 5000);
    centerTrees(best.trees);
    
    return best.trees;
}

int main() {
    string baseline_file = "/home/submission/submission.csv";
    string output_file = "/home/code/experiments/013_genetic_algorithm/improved.csv";
    
    cout << "=== Genetic Algorithm with Topology Crossover ===" << endl;
    
    auto baseline = readBaseline(baseline_file);
    double baseline_score = calculateScore(baseline);
    cout << "Baseline score: " << fixed << setprecision(6) << baseline_score << endl;
    
    auto improved = baseline;
    int num_improved = 0;
    
    // GA parameters
    int pop_size = 30;
    int generations = 50;
    
    // Process all N values
    #pragma omp parallel for schedule(dynamic)
    for (int N = 2; N <= 200; N++) {
        mt19937 rng(42 + N);
        
        double baseline_side = calculateSide(baseline[N]);
        
        // Run GA
        auto result = runGA(N, baseline[N], rng, pop_size, generations);
        
        if (!hasOverlap(result)) {
            double new_side = calculateSide(result);
            
            #pragma omp critical
            {
                if (new_side < baseline_side - 1e-9) {
                    improved[N] = result;
                    num_improved++;
                    cout << "N=" << N << ": " << baseline_side << " -> " << new_side 
                         << " (improved by " << (baseline_side - new_side) << ")" << endl;
                }
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
