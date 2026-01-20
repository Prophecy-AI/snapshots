// sa_small_n.cpp - Focus on small N values with exhaustive search
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
static constexpr int NV = 15;
static constexpr double PI = 3.1415926535897932384626433832795;
static const double TX0[NV] = {
    0.0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125
};
static const double TY0[NV] = {
    0.8, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, -0.2, -0.2, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5
};
struct Bounds { double minx, miny, maxx, maxy; };
struct Poly { double px[NV], py[NV]; Bounds b; };

static inline void build_poly(double cx, double cy, double angle_deg, Poly& out) {
    double rad = angle_deg * (PI / 180.0);
    double s = sin(rad), c = cos(rad);
    double minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
    for (int i = 0; i < NV; i++) {
        double x = TX0[i] * c - TY0[i] * s + cx;
        double y = TX0[i] * s + TY0[i] * c + cy;
        out.px[i] = x; out.py[i] = y;
        minx = min(minx, x); maxx = max(maxx, x);
        miny = min(miny, y); maxy = max(maxy, y);
    }
    out.b = {minx, miny, maxx, maxy};
}

static inline long double orient(long double ax, long double ay, long double bx, long double by, long double cx, long double cy) {
    return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
}
static inline bool on_segment(long double ax, long double ay, long double bx, long double by, long double px, long double py) {
    return min(ax,bx) <= px && px <= max(ax,bx) && min(ay,by) <= py && py <= max(ay,by);
}
static inline bool point_on_edge(long double px, long double py, const Poly& q) {
    for (int i = 0; i < NV; i++) {
        int j = (i + 1) % NV;
        long double ax = q.px[i], ay = q.py[i], bx = q.px[j], by = q.py[j];
        long double o = orient(ax,ay,bx,by,px,py);
        if (o == 0 && on_segment(ax,ay,bx,by,px,py)) return true;
    }
    return false;
}
static inline bool pip_strict(long double px, long double py, const Poly& q) {
    if (px < q.b.minx || px > q.b.maxx || py < q.b.miny || py > q.b.maxy) return false;
    if (point_on_edge(px, py, q)) return false;
    bool in = false;
    int j = NV - 1;
    for (int i = 0; i < NV; i++) {
        long double xi = q.px[i], yi = q.py[i], xj = q.px[j], yj = q.py[j];
        if ((yi > py) != (yj > py)) {
            long double xint = (xj - xi) * (py - yi) / (yj - yi) + xi;
            if (px < xint) in = !in;
        }
        j = i;
    }
    return in;
}
enum class SegHit { NONE, TOUCH, PROPER };
static inline SegHit seg_intersect_type(long double ax, long double ay, long double bx, long double by,
                                        long double cx, long double cy, long double dx, long double dy) {
    long double o1 = orient(ax,ay,bx,by,cx,cy), o2 = orient(ax,ay,bx,by,dx,dy);
    long double o3 = orient(cx,cy,dx,dy,ax,ay), o4 = orient(cx,cy,dx,dy,bx,by);
    auto sgn = [](long double v)->int { return (v > 0) - (v < 0); };
    int s1 = sgn(o1), s2 = sgn(o2), s3 = sgn(o3), s4 = sgn(o4);
    if (s1*s2 < 0 && s3*s4 < 0) return SegHit::PROPER;
    if (s1 == 0 && on_segment(ax,ay,bx,by,cx,cy)) return SegHit::TOUCH;
    if (s2 == 0 && on_segment(ax,ay,bx,by,dx,dy)) return SegHit::TOUCH;
    if (s3 == 0 && on_segment(cx,cy,dx,dy,ax,ay)) return SegHit::TOUCH;
    if (s4 == 0 && on_segment(cx,cy,dx,dy,bx,by)) return SegHit::TOUCH;
    return SegHit::NONE;
}
static inline bool overlap(const Poly& a, const Poly& b) {
    if (a.b.maxx < b.b.minx || a.b.minx > b.b.maxx || a.b.maxy < b.b.miny || a.b.miny > b.b.maxy) return false;
    for (int i = 0; i < NV; i++) {
        if (pip_strict(a.px[i], a.py[i], b)) return true;
        if (pip_strict(b.px[i], b.py[i], a)) return true;
    }
    for (int i = 0; i < NV; i++) {
        int ni = (i + 1) % NV;
        for (int j = 0; j < NV; j++) {
            int nj = (j + 1) % NV;
            if (seg_intersect_type(a.px[i],a.py[i],a.px[ni],a.py[ni],b.px[j],b.py[j],b.px[nj],b.py[nj]) == SegHit::PROPER) return true;
        }
    }
    return false;
}

static inline double parse_s(string s) {
    while (!s.empty() && isspace(s[0])) s.erase(s.begin());
    while (!s.empty() && isspace(s.back())) s.pop_back();
    if (!s.empty() && s[0] == 's') s.erase(s.begin());
    return s.empty() ? 0.0 : stod(s);
}

struct Tree { string id; double x, y, deg; };
using GroupMap = map<string, vector<Tree>>;

GroupMap load_csv(const string& path) {
    ifstream f(path);
    GroupMap gm;
    string line;
    getline(f, line);
    while (getline(f, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string id, xs, ys, ds;
        getline(ss, id, ','); getline(ss, xs, ','); getline(ss, ys, ','); getline(ss, ds, ',');
        Tree t; t.id = id; t.x = parse_s(xs); t.y = parse_s(ys); t.deg = parse_s(ds);
        string gid = id.substr(0, id.find('_'));
        gm[gid].push_back(t);
    }
    return gm;
}

void save_csv(const string& path, const GroupMap& gm) {
    ofstream f(path);
    f << setprecision(20) << "id,x,y,deg\n";
    vector<string> keys;
    for (auto& kv : gm) keys.push_back(kv.first);
    sort(keys.begin(), keys.end(), [](const string& a, const string& b) {
        return stoll(a) < stoll(b);
    });
    for (auto& gid : keys) {
        for (auto& t : gm.at(gid)) {
            f << t.id << ",s" << t.x << ",s" << t.y << ",s" << t.deg << "\n";
        }
    }
}

double calc_side(const vector<Tree>& trees) {
    if (trees.empty()) return 0;
    double minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
    for (auto& t : trees) {
        Poly p; build_poly(t.x, t.y, t.deg, p);
        minx = min(minx, p.b.minx); miny = min(miny, p.b.miny);
        maxx = max(maxx, p.b.maxx); maxy = max(maxy, p.b.maxy);
    }
    return max(maxx - minx, maxy - miny);
}

bool has_overlap(const vector<Tree>& trees) {
    vector<Poly> polys(trees.size());
    for (size_t i = 0; i < trees.size(); i++) {
        build_poly(trees[i].x, trees[i].y, trees[i].deg, polys[i]);
    }
    for (size_t i = 0; i < polys.size(); i++) {
        for (size_t j = i + 1; j < polys.size(); j++) {
            if (overlap(polys[i], polys[j])) return true;
        }
    }
    return false;
}

// Exhaustive search for N=1: try all angles
double optimize_n1(vector<Tree>& trees) {
    double best_side = calc_side(trees);
    double best_angle = trees[0].deg;
    // For N=1, the bounding box depends only on the angle
    // Try many angles
    for (double angle = 0; angle < 360; angle += 0.1) {
        trees[0].deg = angle;
        trees[0].x = 0; trees[0].y = 0;
        double side = calc_side(trees);
        if (side < best_side - 1e-10) {
            best_side = side;
            best_angle = angle;
        }
    }
    trees[0].deg = best_angle;
    trees[0].x = 0; trees[0].y = 0;
    return best_side;
}

// Multi-start SA for small N
double optimize_small_n(vector<Tree>& trees, int restarts = 1000, int iters = 100000) {
    int n = trees.size();
    double best_side = calc_side(trees);
    vector<Tree> best_trees = trees;
    
    mt19937 rng(42);
    uniform_real_distribution<double> pos_dist(-5, 5);
    uniform_real_distribution<double> ang_dist(0, 360);
    uniform_real_distribution<double> unit(0, 1);
    
    for (int r = 0; r < restarts; r++) {
        // Random restart
        vector<Tree> cur = trees;
        for (int i = 0; i < n; i++) {
            cur[i].x = pos_dist(rng);
            cur[i].y = pos_dist(rng);
            cur[i].deg = ang_dist(rng);
        }
        
        // Check for overlaps
        if (has_overlap(cur)) continue;
        
        double cur_side = calc_side(cur);
        
        // SA
        double T = 1.0;
        double cooling = pow(0.0001, 1.0 / iters);
        
        for (int it = 0; it < iters; it++) {
            int idx = rng() % n;
            double old_x = cur[idx].x, old_y = cur[idx].y, old_deg = cur[idx].deg;
            
            cur[idx].x += (unit(rng) - 0.5) * 0.1;
            cur[idx].y += (unit(rng) - 0.5) * 0.1;
            cur[idx].deg += (unit(rng) - 0.5) * 5;
            
            if (has_overlap(cur)) {
                cur[idx].x = old_x; cur[idx].y = old_y; cur[idx].deg = old_deg;
                T *= cooling;
                continue;
            }
            
            double new_side = calc_side(cur);
            double delta = new_side - cur_side;
            
            if (delta < 0 || unit(rng) < exp(-delta * 100 / T)) {
                cur_side = new_side;
                if (cur_side < best_side - 1e-10) {
                    best_side = cur_side;
                    best_trees = cur;
                }
            } else {
                cur[idx].x = old_x; cur[idx].y = old_y; cur[idx].deg = old_deg;
            }
            T *= cooling;
        }
    }
    
    trees = best_trees;
    return best_side;
}

int main(int argc, char** argv) {
    string in = "input.csv", out = "output.csv";
    int max_n = 10;
    
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "-i" && i+1 < argc) in = argv[++i];
        else if (a == "-o" && i+1 < argc) out = argv[++i];
        else if (a == "-n" && i+1 < argc) max_n = stoi(argv[++i]);
    }
    
    cout << "Loading: " << in << endl;
    GroupMap groups = load_csv(in);
    
    double total_improvement = 0;
    
    for (int n = 1; n <= max_n; n++) {
        string gid = to_string(n);
        while (gid.length() < 3) gid = "0" + gid;
        
        if (groups.find(gid) == groups.end()) continue;
        
        auto& trees = groups[gid];
        double orig_side = calc_side(trees);
        
        double new_side;
        if (n == 1) {
            new_side = optimize_n1(trees);
        } else {
            new_side = optimize_small_n(trees, 500, 50000);
        }
        
        double improvement = orig_side - new_side;
        total_improvement += improvement;
        
        cout << "N=" << n << ": " << fixed << setprecision(6) << orig_side << " -> " << new_side;
        if (improvement > 1e-10) {
            cout << " (improved by " << improvement << ")";
        }
        cout << endl;
    }
    
    cout << "\nTotal improvement: " << fixed << setprecision(6) << total_improvement << endl;
    
    save_csv(out, groups);
    cout << "Saved to: " << out << endl;
    
    return 0;
}
