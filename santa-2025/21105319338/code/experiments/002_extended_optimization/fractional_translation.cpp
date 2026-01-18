// Fractional Translation Optimizer
// Very fine position adjustments to squeeze out small improvements
// Compile: g++ -O3 -march=native -std=c++17 -fopenmp -o frac_trans fractional_translation.cpp

#include <bits/stdc++.h>
#include <omp.h>
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

inline bool pip(long double px, long double py, const Poly& q) {
    bool in = false;
    int j = NV - 1;
    for (int i = 0; i < NV; i++) {
        if ((q.py[i] > py) != (q.py[j] > py) &&
            px < (q.px[j] - q.px[i]) * (py - q.py[i]) / (q.py[j] - q.py[i]) + q.px[i])
            in = !in;
        j = i;
    }
    return in;
}

inline bool segInt(long double ax, long double ay, long double bx, long double by,
                   long double cx, long double cy, long double dx, long double dy) {
    long double d1 = (dx-cx)*(ay-cy) - (dy-cy)*(ax-cx);
    long double d2 = (dx-cx)*(by-cy) - (dy-cy)*(bx-cx);
    long double d3 = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
    long double d4 = (bx-ax)*(dy-ay) - (by-ay)*(dx-ax);
    return ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0));
}

inline bool overlap(const Poly& a, const Poly& b) {
    if (a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0) return false;
    for (int i = 0; i < NV; i++) {
        if (pip(a.px[i], a.py[i], b)) return true;
        if (pip(b.px[i], b.py[i], a)) return true;
    }
    for (int i = 0; i < NV; i++) {
        int ni = (i + 1) % NV;
        for (int j = 0; j < NV; j++) {
            int nj = (j + 1) % NV;
            if (segInt(a.px[i], a.py[i], a.px[ni], a.py[ni],
                      b.px[j], b.py[j], b.px[nj], b.py[nj])) return true;
        }
    }
    return false;
}

struct Cfg {
    int n;
    long double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];
    long double gx0, gy0, gx1, gy1;

    void upd(int i) { getPoly(x[i], y[i], a[i], pl[i]); }
    void updAll() { for (int i = 0; i < n; i++) upd(i); updGlobal(); }

    void updGlobal() {
        gx0 = gy0 = 1e9L; gx1 = gy1 = -1e9L;
        for (int i = 0; i < n; i++) {
            if (pl[i].x0 < gx0) gx0 = pl[i].x0;
            if (pl[i].x1 > gx1) gx1 = pl[i].x1;
            if (pl[i].y0 < gy0) gy0 = pl[i].y0;
            if (pl[i].y1 > gy1) gy1 = pl[i].y1;
        }
    }

    bool hasOvl(int i) const {
        for (int j = 0; j < n; j++)
            if (i != j && overlap(pl[i], pl[j])) return true;
        return false;
    }

    long double side() const { return max(gx1 - gx0, gy1 - gy0); }
};

Cfg configs[MAX_N + 1];
long double best_sides[MAX_N + 1];

void parse_csv(const string& fn) {
    ifstream f(fn);
    string line;
    getline(f, line);
    
    map<int, vector<tuple<int, long double, long double, long double>>> data;
    
    while (getline(f, line)) {
        size_t p1 = line.find(',');
        size_t p2 = line.find(',', p1+1);
        size_t p3 = line.find(',', p2+1);
        
        string id = line.substr(0, p1);
        string xs = line.substr(p1+2, p2-p1-2);
        string ys = line.substr(p2+2, p3-p2-2);
        string ds = line.substr(p3+2);
        
        size_t us = id.find('_');
        int n = stoi(id.substr(0, us));
        int idx = stoi(id.substr(us+1));
        
        data[n].push_back({idx, stold(xs), stold(ys), stold(ds)});
    }
    
    for (auto& [n, trees] : data) {
        sort(trees.begin(), trees.end());
        configs[n].n = n;
        for (int i = 0; i < n; i++) {
            configs[n].x[i] = get<1>(trees[i]);
            configs[n].y[i] = get<2>(trees[i]);
            configs[n].a[i] = get<3>(trees[i]);
        }
        configs[n].updAll();
        best_sides[n] = configs[n].side();
    }
}

void save_csv(const string& fn) {
    ofstream f(fn);
    f << fixed << setprecision(15);
    f << "id,x,y,deg\n";
    for (int n = 1; n <= MAX_N; n++) {
        for (int i = 0; i < n; i++) {
            f << setfill('0') << setw(3) << n << "_" << i << ",s" 
              << configs[n].x[i] << ",s" << configs[n].y[i] << ",s" << configs[n].a[i] << "\n";
        }
    }
}

long double calc_total_score() {
    long double total = 0;
    for (int n = 1; n <= MAX_N; n++) {
        long double s = best_sides[n];
        total += s * s / n;
    }
    return total;
}

void fractional_translation() {
    cout << "Starting Fractional Translation...\n";
    cout << fixed << setprecision(10) << "Initial score: " << calc_total_score() << "\n";

    // Very fine step sizes
    const long double frac_steps[] = {0.001L, 0.0005L, 0.0002L, 0.0001L, 0.00005L, 0.00002L, 0.00001L};
    const int dx[] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dy[] = {0, 0, 1, -1, 1, -1, 1, -1};
    
    int total_improvements = 0;
    
    for (int pass = 0; pass < 3; pass++) {
        int pass_improvements = 0;
        
        #pragma omp parallel for schedule(dynamic) reduction(+:pass_improvements)
        for (int n = 2; n <= MAX_N; n++) {
            Cfg& c = configs[n];
            long double bs = c.side();
            bool improved = true;
            
            while (improved) {
                improved = false;
                
                for (int i = 0; i < c.n; i++) {
                    for (long double step : frac_steps) {
                        for (int d = 0; d < 8; d++) {
                            long double ox = c.x[i], oy = c.y[i];
                            c.x[i] = ox + dx[d] * step;
                            c.y[i] = oy + dy[d] * step;
                            c.upd(i);
                            
                            if (!c.hasOvl(i)) {
                                c.updGlobal();
                                long double ns = c.side();
                                if (ns < bs - 1e-15L) {
                                    bs = ns;
                                    improved = true;
                                    pass_improvements++;
                                } else {
                                    c.x[i] = ox;
                                    c.y[i] = oy;
                                    c.upd(i);
                                    c.updGlobal();
                                }
                            } else {
                                c.x[i] = ox;
                                c.y[i] = oy;
                                c.upd(i);
                            }
                        }
                    }
                }
            }
            
            #pragma omp critical
            {
                if (bs < best_sides[n]) {
                    best_sides[n] = bs;
                }
            }
        }
        
        total_improvements += pass_improvements;
        cout << "Pass " << (pass + 1) << ": " << pass_improvements << " improvements, score: " 
             << calc_total_score() << "\n";
    }

    cout << "\nFractional Translation Complete!\n";
    cout << "Total improvements: " << total_improvements << "\n";
    cout << fixed << setprecision(10) << "Final score: " << calc_total_score() << "\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: ./frac_trans input.csv output.csv\n";
        return 1;
    }

    string input_file = argv[1];
    string output_file = argv[2];

    cout << "Fractional Translation Optimizer\n";
    cout << "=================================\n";
    cout << "Loading " << input_file << "...\n";

    parse_csv(input_file);

    cout << "Loaded " << MAX_N << " configurations\n";

    fractional_translation();

    cout << "Saving to " << output_file << "...\n";
    save_csv(output_file);

    cout << "Done!\n";

    return 0;
}
