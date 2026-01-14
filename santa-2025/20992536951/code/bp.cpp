// Backward Propagation Optimizer
// Based on: https://www.kaggle.com/code/guntasdhanjal/santa-2025-simple-optimization-v2
//
// Key idea: If removing one tree from N-tree config gives better (N-1)-tree config, propagate it backward
// Compile: g++ -O3 -std=c++17 -o bp bp.cpp

#include <bits/stdc++.h>
using namespace std;

constexpr int MAX_N = 200;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;

alignas(64) const long double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
alignas(64) const long double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

struct Poly {
    long double px[NV], py[NV];
    long double x0, y0, x1, y1;
};

inline void getPoly(long double cx, long double cy, long double deg, Poly& q) {
    long double rad = deg * (PI / 180.0L);
    long double s = sinl(rad), c = cosl(rad);
    long double minx = 1e9L, miny = 1e9L, maxx = -1e9L, maxy = -1e9L;
    for (int i = 0; i < NV; i++) {
        long double x = TX[i] * c - TY[i] * s + cx;
        long double y = TX[i] * s + TY[i] * c + cy;
        q.px[i] = x; q.py[i] = y;
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
    }
    q.x0 = minx; q.y0 = miny; q.x1 = maxx; q.y1 = maxy;
}

struct Cfg {
    int n;
    long double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];
    long double gx0, gy0, gx1, gy1;

    inline void upd(int i) { getPoly(x[i], y[i], a[i], pl[i]); }

    inline void calc_bounds() {
        gx0 = gy0 = 1e9L;
        gx1 = gy1 = -1e9L;
        for (int i = 0; i < n; i++) {
            if (pl[i].x0 < gx0) gx0 = pl[i].x0;
            if (pl[i].y0 < gy0) gy0 = pl[i].y0;
            if (pl[i].x1 > gx1) gx1 = pl[i].x1;
            if (pl[i].y1 > gy1) gy1 = pl[i].y1;
        }
    }

    inline long double side() const {
        return max(gx1 - gx0, gy1 - gy0);
    }

    void remove_tree(int idx) {
        // Remove tree at index idx by shifting remaining trees
        for (int i = idx; i < n - 1; i++) {
            x[i] = x[i + 1];
            y[i] = y[i + 1];
            a[i] = a[i + 1];
            pl[i] = pl[i + 1];
        }
        n--;
    }

    void rebuild_polys() {
        for (int i = 0; i < n; i++) {
            upd(i);
        }
        calc_bounds();
    }
};

// Global storage for all configurations
Cfg configs[MAX_N + 1];  // configs[n] stores the best n-tree configuration
long double best_sides[MAX_N + 1];

void parse_csv(const string& filename) {
    ifstream f(filename);
    string line;
    getline(f, line); // Skip header

    map<int, vector<tuple<long double, long double, long double>>> data;

    while (getline(f, line)) {
        stringstream ss(line);
        string id_str, x_str, y_str, deg_str;

        getline(ss, id_str, ',');
        getline(ss, x_str, ',');
        getline(ss, y_str, ',');
        getline(ss, deg_str);

        // Parse id like "010_0"
        int n = stoi(id_str.substr(0, 3));

        // Remove 's' prefix
        long double x = stold(x_str);
        long double y = stold(y_str);
        long double deg = stold(deg_str);

        data[n].push_back({x, y, deg});
    }

    // Populate configs
    for (auto& [n, trees] : data) {
        configs[n].n = n;
        for (int i = 0; i < n; i++) {
            auto [x, y, a] = trees[i];
            configs[n].x[i] = x;
            configs[n].y[i] = y;
            configs[n].a[i] = a;
        }
        configs[n].rebuild_polys();
        best_sides[n] = configs[n].side();
    }
}

void save_csv(const string& filename) {
    ofstream f(filename);
    f << "id,x,y,deg\n";
    f << fixed << setprecision(17);

    for (int n = 1; n <= MAX_N; n++) {
        for (int i = 0; i < configs[n].n; i++) {
            f << setw(3) << setfill('0') << n << "_" << i << ",";
            f << configs[n].x[i] << ",";
            f << configs[n].y[i] << ",";
            f << configs[n].a[i] << "\n";
        }
    }
}

long double calc_total_score() {
    long double score = 0.0L;
    for (int n = 1; n <= MAX_N; n++) {
        long double side = configs[n].side();
        score += (side * side) / n;
    }
    return score;
}

vector<int> get_bbox_touching_tree_indices(const Cfg& cfg) {
    vector<int> touching_indices;
    const long double eps = 1e-9L;
    
    for (int i = 0; i < cfg.n; i++) {
        const Poly& p = cfg.pl[i];
        bool touches = false;
        
        // Check if tree touches left boundary: tree's left edge aligns with global left
        if (abs(p.x0 - cfg.gx0) < eps && p.y1 >= cfg.gy0 - eps && p.y0 <= cfg.gy1 + eps) {
            touches = true;
        }
        // Check if tree touches right boundary: tree's right edge aligns with global right
        if (abs(p.x1 - cfg.gx1) < eps && p.y1 >= cfg.gy0 - eps && p.y0 <= cfg.gy1 + eps) {
            touches = true;
        }
        // Check if tree touches bottom boundary: tree's bottom edge aligns with global bottom
        if (abs(p.y0 - cfg.gy0) < eps && p.x1 >= cfg.gx0 - eps && p.x0 <= cfg.gx1 + eps) {
            touches = true;
        }
        // Check if tree touches top boundary: tree's top edge aligns with global top
        if (abs(p.y1 - cfg.gy1) < eps && p.x1 >= cfg.gx0 - eps && p.x0 <= cfg.gx1 + eps) {
            touches = true;
        }
        
        if (touches) {
            touching_indices.push_back(i);
        }
    }
    
    // If no trees touch boundary (shouldn't happen), return all indices as fallback
    if (touching_indices.empty()) {
        for (int i = 0; i < cfg.n; i++) {
            touching_indices.push_back(i);
        }
    }
    
    return touching_indices;
}

void backward_propagation() {
    cout << "Starting Backward Propagation...\n";
    cout << fixed << setprecision(8) << "Initial score: " << calc_total_score() << "\n\n";

    int total_improvements = 0;

    // Go from N=200 down to N=2
    for (int n = MAX_N; n >= 2; n--) {

        // Start with a working copy of the n-tree configuration
        Cfg candidate = configs[n];

        // Keep removing trees until we can't improve anymore
        while (candidate.n > 1) {
            int target_size = candidate.n - 1;
            long double best_current_side = best_sides[target_size];
            long double best_new_side = 1e9L;
            int best_tree_to_delete = -1;

            // Get trees that touch the bounding box boundary
            vector<int> touching_indices = get_bbox_touching_tree_indices(candidate);

            // Try deleting each boundary-touching tree
            for (int tree_idx : touching_indices) {
                // Create a test copy
                Cfg test_candidate = candidate;
                test_candidate.remove_tree(tree_idx);
                test_candidate.calc_bounds();

                long double test_side = test_candidate.side();

                // Track the best deletion
                if (test_side < best_new_side) {
                    best_new_side = test_side;
                    best_tree_to_delete = tree_idx;
                }
            }

            // If we found a deletion candidate, always remove it and continue
            if (best_tree_to_delete != -1) {
                // Remove the best tree
                candidate.remove_tree(best_tree_to_delete);
                candidate.calc_bounds();

                // If this improves the target_size configuration, save it
                if (best_new_side < best_current_side) {
                    cout << "improved " << candidate.n << " from n=" << n << " " << best_current_side << " -> " << best_new_side << "\n";
                    configs[target_size] = candidate;
                    best_sides[target_size] = best_new_side;
                    total_improvements++;
                }
                // Continue the loop even if not better than stored - keep optimizing
            } else {
                // Can't find any valid deletion, stop for this configuration
                break;
            }
        }
    }

    long double final_score = calc_total_score();
    cout << "\n\nBackward Propagation Complete!\n";
    cout << "Total improvements: " << total_improvements << "\n";
    cout << fixed << setprecision(12) << "Final score: " << final_score << "\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: ./bp input.csv output.csv\n";
        return 1;
    }

    string input_file = argv[1];
    string output_file = argv[2];

    cout << "Backward Propagation Optimizer\n";
    cout << "===============================\n";
    cout << "Loading " << input_file << "...\n";

    parse_csv(input_file);

    cout << "Loaded " << MAX_N << " configurations\n";

    backward_propagation();

    cout << "Saving to " << output_file << "...\n";
    save_csv(output_file);

    cout << "Done!\n";

    return 0;
}