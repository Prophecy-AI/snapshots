// Backward Propagation Optimizer
// Transfers solutions from larger N to smaller N by removing boundary trees

#include <bits/stdc++.h>
using namespace std;

constexpr int MAX_N = 200;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;

const long double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
const long double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

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

    void calc_bounds() {
        for (int i = 0; i < n; i++) getPoly(x[i], y[i], a[i], pl[i]);
        gx0 = gy0 = 1e9L; gx1 = gy1 = -1e9L;
        for (int i = 0; i < n; i++) {
            if (pl[i].x0 < gx0) gx0 = pl[i].x0;
            if (pl[i].x1 > gx1) gx1 = pl[i].x1;
            if (pl[i].y0 < gy0) gy0 = pl[i].y0;
            if (pl[i].y1 > gy1) gy1 = pl[i].y1;
        }
    }

    long double side() const { return max(gx1 - gx0, gy1 - gy0); }

    void remove_tree(int idx) {
        for (int i = idx; i < n - 1; i++) {
            x[i] = x[i + 1];
            y[i] = y[i + 1];
            a[i] = a[i + 1];
        }
        n--;
    }
};

Cfg configs[MAX_N + 1];
long double best_sides[MAX_N + 1];

void parse_csv(const string& fn) {
    ifstream f(fn);
    string line;
    getline(f, line);
    
    map<int, vector<tuple<int, long double, long double, long double>>> data;
    
    while (getline(f, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string id, xs, ys, ds;
        getline(ss, id, ',');
        getline(ss, xs, ',');
        getline(ss, ys, ',');
        getline(ss, ds, ',');
        
        int n = stoi(id.substr(0, 3));
        int i = stoi(id.substr(4));
        
        auto parseVal = [](const string& s) -> long double {
            if (s[0] == 's') return stold(s.substr(1));
            return stold(s);
        };
        
        data[n].push_back({i, parseVal(xs), parseVal(ys), parseVal(ds)});
    }
    
    for (auto& [n, trees] : data) {
        sort(trees.begin(), trees.end());
        configs[n].n = n;
        for (int i = 0; i < n; i++) {
            configs[n].x[i] = get<1>(trees[i]);
            configs[n].y[i] = get<2>(trees[i]);
            configs[n].a[i] = get<3>(trees[i]);
        }
        configs[n].calc_bounds();
        best_sides[n] = configs[n].side();
    }
}

void save_csv(const string& fn) {
    ofstream f(fn);
    f << fixed << setprecision(18);
    f << "id,x,y,deg\n";
    for (int n = 1; n <= MAX_N; n++) {
        for (int i = 0; i < n; i++) {
            f << setfill('0') << setw(3) << n << "_" << i << ",";
            f << "s" << configs[n].x[i] << ",s" << configs[n].y[i] << ",s" << configs[n].a[i] << "\n";
        }
    }
}

long double calc_total_score() {
    long double s = 0;
    for (int n = 1; n <= MAX_N; n++) {
        s += best_sides[n] * best_sides[n] / n;
    }
    return s;
}

vector<int> get_bbox_touching_tree_indices(const Cfg& cfg) {
    vector<int> touching_indices;
    long double eps = 0.01L;
    
    for (int i = 0; i < cfg.n; i++) {
        const Poly& p = cfg.pl[i];
        bool touches = false;
        
        if (fabsl(p.x0 - cfg.gx0) < eps) touches = true;
        if (fabsl(p.x1 - cfg.gx1) < eps) touches = true;
        if (fabsl(p.y0 - cfg.gy0) < eps) touches = true;
        if (fabsl(p.y1 - cfg.gy1) < eps) touches = true;
        
        if (touches) touching_indices.push_back(i);
    }
    
    if (touching_indices.empty()) {
        for (int i = 0; i < cfg.n; i++) touching_indices.push_back(i);
    }
    
    return touching_indices;
}

void backward_propagation() {
    cout << "Starting Backward Propagation...\n";
    cout << fixed << setprecision(8) << "Initial score: " << calc_total_score() << "\n\n";

    int total_improvements = 0;

    for (int n = MAX_N; n >= 2; n--) {
        Cfg candidate = configs[n];

        while (candidate.n > 1) {
            int target_size = candidate.n - 1;
            long double best_current_side = best_sides[target_size];
            long double best_new_side = 1e9L;
            int best_tree_to_delete = -1;

            vector<int> touching_indices = get_bbox_touching_tree_indices(candidate);

            for (int tree_idx : touching_indices) {
                Cfg test_candidate = candidate;
                test_candidate.remove_tree(tree_idx);
                test_candidate.calc_bounds();

                long double test_side = test_candidate.side();

                if (test_side < best_new_side) {
                    best_new_side = test_side;
                    best_tree_to_delete = tree_idx;
                }
            }

            if (best_tree_to_delete != -1) {
                candidate.remove_tree(best_tree_to_delete);
                candidate.calc_bounds();

                if (best_new_side < best_current_side - 1e-12L) {
                    cout << "improved " << candidate.n << " from n=" << n << " " << best_current_side << " -> " << best_new_side << "\n";
                    configs[target_size] = candidate;
                    best_sides[target_size] = best_new_side;
                    total_improvements++;
                }
            } else {
                break;
            }
        }
    }

    cout << "\n\nBackward Propagation Complete!\n";
    cout << "Total improvements: " << total_improvements << "\n";
    cout << fixed << setprecision(12) << "Final score: " << calc_total_score() << "\n";
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
