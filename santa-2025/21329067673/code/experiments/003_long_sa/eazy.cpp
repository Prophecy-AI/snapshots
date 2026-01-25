#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <chrono>
#include <random>
#include <complex>
#include <omp.h>

using namespace std;

typedef complex<double> cd;

// --- CONSTANTS ---
constexpr int MAX_N = 205;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;
const double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
const double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

thread_local mt19937_64 rng(random_device{}());
inline double rf() { return uniform_real_distribution<double>(0, 1)(rng); }
inline int ri(int n) { return uniform_int_distribution<int>(0, n - 1)(rng); }

struct Pt { double x, y; };
struct Poly { 
    Pt p[NV];
    double x0, y0, x1, y1;
    void bbox() {
        x0 = x1 = p[0].x; y0 = y1 = p[0].y;
        for (int i = 1; i < NV; i++) {
            x0 = min(x0, p[i].x); x1 = max(x1, p[i].x);
            y0 = min(y0, p[i].y); y1 = max(y1, p[i].y);
        }
    }
};

// --- GEOMETRY ENGINE ---
bool contains(const Poly& poly, Pt pt) {
    bool inside = false;
    for (int i = 0, j = NV - 1; i < NV; j = i++) {
        if (((poly.p[i].y > pt.y) != (poly.p[j].y > pt.y)) &&
            (pt.x < (poly.p[j].x - poly.p[i].x) * (pt.y - poly.p[i].y) / (poly.p[j].y - poly.p[i].y) + poly.p[i].x))
            inside = !inside;
    }
    return inside;
}

inline bool overlap(const Poly& a, const Poly& b) {
    if (a.x1 < b.x0 - 1e-13 || b.x1 < a.x0 - 1e-13 || a.y1 < b.y0 - 1e-13 || b.y1 < a.y0 - 1e-13) return false;
    auto ccw = [](Pt p, Pt q, Pt r) { 
        long double v = (long double)(q.y - p.y) * (r.x - q.x) - (long double)(q.x - p.x) * (r.y - q.y);
        return (v > 1e-20L) ? 1 : (v < -1e-20L ? -1 : 0); 
    };
    for (int i = 0; i < NV; i++) {
        for (int j = 0; j < NV; j++) {
            Pt p1 = a.p[i], q1 = a.p[(i+1)%NV], p2 = b.p[j], q2 = b.p[(j+1)%NV];
            if (ccw(p1, q1, p2) != ccw(p1, q1, q2) && ccw(p2, q2, p1) != ccw(p2, q2, q1)) return true;
        }
    }
    return contains(a, b.p[0]) || contains(b, a.p[0]);
}

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

struct Cfg {
    int n;
    double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];
    void upd(int i) { pl[i] = getPoly(x[i], y[i], a[i]); }
    void updAll() { for (int i = 0; i < n; i++) upd(i); }
    double side() {
        double xmin = 1e18, xmax = -1e18, ymin = 1e18, ymax = -1e18;
        for (int i = 0; i < n; i++) {
            xmin = min(xmin, pl[i].x0); xmax = max(xmax, pl[i].x1);
            ymin = min(ymin, pl[i].y0); ymax = max(ymax, pl[i].y1);
        }
        return max(xmax - xmin, ymax - ymin);
    }
    bool check_valid() {
        for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (overlap(pl[i], pl[j])) return false;
        return true;
    }
};

// --- CALCULUS: Square Potential Gradient ---
void apply_square_pressure(Cfg& cur, int i, double S, double scale) {
    double x_mid = (cur.pl[i].x0 + cur.pl[i].x1) / 2.0;
    double y_mid = (cur.pl[i].y0 + cur.pl[i].y1) / 2.0;
    double L = S / 2.0;
    auto get_grad = [&](double pos) {
        double d1 = L - pos;
        double d2 = L + pos;
        if (d1 < 1e-9) d1 = 1e-9;
        if (d2 < 1e-9) d2 = 1e-9;
        return (1.0 / d1) - (1.0 / d2);
    };
    double gx = get_grad(x_mid);
    double gy = get_grad(y_mid);
    cur.x[i] -= gx * scale * 0.01; 
    cur.y[i] -= gy * scale * 0.01;
}

// --- MASTER HYBRID CALCULUS CYCLE ---
double run_powerhouse_cycle(Cfg& cur, int iter, double scale) {
    double start_s = cur.side();
    double best_s = start_s;
    
    for (int it = 0; it < iter; it++) {
        if (it > 0 && it % 40000 == 0) {
            Cfg temp = cur;
            double mx=0, my=0; for(int k=0; k<cur.n; k++){ mx+=cur.x[k]; my+=cur.y[k]; }
            mx/=cur.n; my/=cur.n;
            double f = (it % 80000 == 0) ? 0.999999 : 1.0000001; 
            for(int k=0; k<cur.n; k++){ cur.x[k]=mx+(cur.x[k]-mx)*f; cur.y[k]=my+(cur.y[k]-my)*f; }
            cur.updAll();
            if (cur.check_valid() && cur.side() < best_s + 1e-15) best_s = cur.side();
            else cur = temp;
        }

        int i = ri(cur.n);
        double ox = cur.x[i], oy = cur.y[i], oa = cur.a[i];

        double r_move = rf();
        if (r_move < 0.3) {
            cur.x[i] += (rf()-0.5) * scale;
            cur.y[i] += (rf()-0.5) * scale;
        } else if (r_move < 0.6) {
            cd z(cur.x[i], cur.y[i]);
            cd rot = exp(cd(0, (rf()-0.5) * scale * 0.2));
            z *= rot;
            cur.x[i] = z.real(); cur.y[i] = z.imag();
        } else if (r_move < 0.9) {
            cur.a[i] += (rf()-0.5) * scale * 45.0;
        } else {
            apply_square_pressure(cur, i, best_s, scale);
        }

        cur.upd(i);
        bool hit = false;
        for (int j = 0; j < cur.n; j++) if (i != j && overlap(cur.pl[i], cur.pl[j])) { hit = true; break; }

        if (!hit && cur.side() <= best_s + 1e-15) {
            if (cur.side() < best_s - 1e-15) best_s = cur.side();
        } else {
            cur.x[i]=ox; cur.y[i]=oy; cur.a[i]=oa; cur.upd(i);
        }
    }
    return start_s - best_s;
}

// --- IO ---
map<int, Cfg> loadCSV(string fn) {
    map<int, Cfg> res; ifstream f(fn); string ln, h; if(!f) return res;
    getline(f, h);
    while (getline(f, ln)) {
        stringstream ss(ln); string id, sx, sy, sa;
        if(!getline(ss, id, ',')) continue;
        getline(ss, sx, ','); getline(ss, sy, ','); getline(ss, sa, ',');
        int n = stoi(id.substr(0, 3)), idx = stoi(id.substr(4));
        auto p = [](string s) { 
            size_t st = s.find_first_of("0123456789.-"); 
            return (st == string::npos) ? 0.0 : stod(s.substr(st)); 
        };
        res[n].n = n; res[n].x[idx] = p(sx); res[n].y[idx] = p(sy); res[n].a[idx] = p(sa);
    }
    for (auto& pair : res) pair.second.updAll();
    return res;
}

void saveCSV(string fn, map<int, Cfg>& res) {
    ofstream f(fn); f << "id,x,y,deg" << endl;
    for (int n = 1; n <= 200; n++) {
        if (!res.count(n)) continue;
        for (int i = 0; i < n; i++)
            f << setfill('0') << setw(3) << n << "_" << i << ",s" << fixed << setprecision(18) 
              << res[n].x[i] << ",s" << res[n].y[i] << ",s" << res[n].a[i] << "\n";
    }
}

int main(int argc, char** argv) {
    string input_file = "submission_best.csv";
    string output_file = "submission_eazy.csv";
    int time_per_n = 10;  // seconds per N
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "-o" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "-t" && i + 1 < argc) time_per_n = stoi(argv[++i]);
    }
    
    cout << "Loading " << input_file << "..." << endl;
    auto res = loadCSV(input_file);
    if(res.empty()) { cerr << "Failed to load!" << endl; return 1; }
    
    // Calculate initial score
    double init_score = 0;
    for (auto& [n, c] : res) init_score += c.side() * c.side() / n;
    cout << "Initial score: " << fixed << setprecision(6) << init_score << endl;
    
    vector<int> keys; for(auto const& [n, g] : res) keys.push_back(n);
    sort(keys.rbegin(), keys.rend());

    vector<double> scales = {1e-3, 1e-5, 1e-7};
    for (double sc : scales) {
        cout << "\n>>> POWERHOUSE PHASE | Scale: " << sc << " <<<" << endl;
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < (int)keys.size(); i++) {
            int n = keys[i];
            auto id_start = chrono::steady_clock::now();
            int fails = 0;
            while (true) {
                if (chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - id_start).count() >= time_per_n) break;
                double gain = run_powerhouse_cycle(res[n], 100000, sc);
                if (gain > 1e-12) fails = 0;
                else if (++fails >= 2) break; 
            }
            #pragma omp critical
            cout << "[N=" << n << "] Side: " << fixed << setprecision(12) << res[n].side() << endl;
        }
    }
    
    // Calculate final score
    double final_score = 0;
    for (auto& [n, c] : res) final_score += c.side() * c.side() / n;
    cout << "\nFinal score: " << fixed << setprecision(6) << final_score << endl;
    cout << "Improvement: " << fixed << setprecision(6) << (init_score - final_score) << endl;
    
    saveCSV(output_file, res);
    cout << "Saved to " << output_file << endl;
    return 0;
}
