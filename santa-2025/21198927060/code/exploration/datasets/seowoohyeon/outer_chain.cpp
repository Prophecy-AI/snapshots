#include <bits/stdc++.h>
using namespace std;

constexpr int MAX_N = 200;
constexpr int NV    = 15;
constexpr double PI = 3.14159265358979323846;

// 크리스마스 트리 기준 좌표 (smartmanoj / 기존 코드와 동일)
alignas(64) const double TX[NV] = {
    0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,
    -0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125
};
alignas(64) const double TY[NV] = {
    0.8,0.5,0.5,0.25,0.25,0,0,-0.2,
    -0.2,0,0,0.25,0.25,0.5,0.5
};

struct Poly {
    double px[NV], py[NV];
    double x0, y0, x1, y1;  // bbox
};

inline void getPoly(double cx, double cy, double deg, Poly &q) {
    double rad = deg * (PI / 180.0);
    double s = sin(rad), c = cos(rad);
    double minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
    for (int i = 0; i < NV; i++) {
        double x = TX[i] * c - TY[i] * s + cx;
        double y = TX[i] * s + TY[i] * c + cy;
        q.px[i] = x;
        q.py[i] = y;
        if (x < minx) minx = x;
        if (x > maxx) maxx = x;
        if (y < miny) miny = y;
        if (y > maxy) maxy = y;
    }
    q.x0 = minx; q.y0 = miny;
    q.x1 = maxx; q.y1 = maxy;
}

struct Cfg {
    int n = 0;
    double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly   pl[MAX_N];
    double gx0, gy0, gx1, gy1;  // 전체 bbox

    void upd(int i) {
        getPoly(x[i], y[i], a[i], pl[i]);
    }

    void updAll() {
        if (n <= 0) {
            gx0 = gy0 = 0.0;
            gx1 = gy1 = 0.0;
            return;
        }
        gx0 = gy0 = 1e9;
        gx1 = gy1 = -1e9;
        for (int i = 0; i < n; i++) {
            upd(i);
            if (pl[i].x0 < gx0) gx0 = pl[i].x0;
            if (pl[i].x1 > gx1) gx1 = pl[i].x1;
            if (pl[i].y0 < gy0) gy0 = pl[i].y0;
            if (pl[i].y1 > gy1) gy1 = pl[i].y1;
        }
    }

    double side() const {
        return max(gx1 - gx0, gy1 - gy0);
    }

    double score() const {
        if (n <= 0) return 1e18;
        double s = side();
        return s * s / n;
    }

    // n에서 특정 index 트리를 제거해 n-1 cfg 생성
    Cfg removeTree(int removeIdx) const {
        Cfg c;
        if (n <= 0) return c;
        c.n = n - 1;
        int dst = 0;
        for (int i = 0; i < n; i++) {
            if (i == removeIdx) continue;
            c.x[dst] = x[i];
            c.y[dst] = y[i];
            c.a[dst] = a[i];
            dst++;
        }
        c.updAll();
        return c;
    }

    // "정사각형 변에 가장 가까운" 트리 index
    int outermostIndexBySquare() const {
        if (n <= 0) return -1;

        double s  = side();
        double cx = 0.5 * (gx0 + gx1);
        double cy = 0.5 * (gy0 + gy1);
        double half = 0.5 * s;

        double sx0 = cx - half;
        double sx1 = cx + half;
        double sy0 = cy - half;
        double sy1 = cy + half;

        int bestIdx = 0;
        double bestD = 1e18;

        for (int i = 0; i < n; i++) {
            double dLeft   = pl[i].x0 - sx0;
            double dRight  = sx1 - pl[i].x1;
            double dBottom = pl[i].y0 - sy0;
            double dTop    = sy1 - pl[i].y1;
            double dEdge   = min(min(dLeft, dRight), min(dBottom, dTop));

            if (dEdge < bestD - 1e-12) {
                bestD = dEdge;
                bestIdx = i;
            } else if (fabs(dEdge - bestD) <= 1e-12) {
                double tx  = 0.5 * (pl[i].x0 + pl[i].x1);
                double ty  = 0.5 * (pl[i].y0 + pl[i].y1);
                double bx  = 0.5 * (pl[bestIdx].x0 + pl[bestIdx].x1);
                double by  = 0.5 * (pl[bestIdx].y0 + pl[bestIdx].y1);
                double d2c = (tx - cx) * (tx - cx) + (ty - cy) * (ty - cy);
                double d2b = (bx - cx) * (bx - cx) + (by - cy) * (by - cy);
                if (d2c > d2b) {
                    bestIdx = i;
                }
            }
        }
        return bestIdx;
    }
};

// CSV 로드: smartmanoj 형식 (id,x,y,deg) → n별 Cfg
map<int, Cfg> loadCSV(const string &fn) {
    map<int, Cfg> cfg;
    ifstream f(fn);
    if (!f) {
        cerr << "Cannot open " << fn << "\n";
        return cfg;
    }
    string ln;
    // header skip
    if (!getline(f, ln)) return cfg;

    map<int, vector<tuple<int,double,double,double>>> data;

    while (getline(f, ln)) {
        if (ln.empty()) continue;
        size_t p1 = ln.find(',');
        if (p1 == string::npos) continue;
        size_t p2 = ln.find(',', p1 + 1);
        if (p2 == string::npos) continue;
        size_t p3 = ln.find(',', p2 + 1);
        if (p3 == string::npos) continue;

        string id = ln.substr(0, p1);
        string xs = ln.substr(p1 + 1, p2 - p1 - 1);
        string ys = ln.substr(p2 + 1, p3 - p2 - 1);
        string ds = ln.substr(p3 + 1);

        if (!xs.empty() && xs[0] == 's') xs = xs.substr(1);
        if (!ys.empty() && ys[0] == 's') ys = ys.substr(1);
        if (!ds.empty() && ds[0] == 's') ds = ds.substr(1);

        if (id.size() < 5) continue; // "003_0" 형태 가정
        int n   = stoi(id.substr(0, 3));
        int idx = stoi(id.substr(4));

        double x = stod(xs);
        double y = stod(ys);
        double d = stod(ds);
        data[n].push_back({idx, x, y, d});
    }

    for (auto &kv : data) {
        int n = kv.first;
        auto &v = kv.second;
        Cfg c;
        c.n = n;
        for (auto &t : v) {
            int idx;
            double x, y, d;
            tie(idx, x, y, d) = t;
            if (0 <= idx && idx < n) {
                c.x[idx] = x;
                c.y[idx] = y;
                c.a[idx] = d;
            }
        }
        c.updAll();
        cfg[n] = c;
    }

    return cfg;
}

// CSV 저장
void saveCSV(const string &fn, const map<int, Cfg> &cfg) {
    ofstream f(fn);
    f << fixed << setprecision(15);
    f << "id,x,y,deg\n";
    for (int n = 1; n <= 200; n++) {
        auto it = cfg.find(n);
        if (it == cfg.end()) continue;
        const Cfg &c = it->second;
        for (int i = 0; i < c.n; i++) {
            f << setfill('0') << setw(3) << n
              << "_" << i
              << ",s" << c.x[i]
              << ",s" << c.y[i]
              << ",s" << c.a[i] << "\n";
        }
    }
}

int main(int argc, char **argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // 기본은 현재 작업 디렉토리의 submission.csv를 in-place로 최적화
    string in  = "submission.csv";
    string out = "submission.csv";

    // 옵션으로 경로 바꾸고 싶으면 -i / -o 사용
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "-i" && i + 1 < argc) in = argv[++i];
        else if (a == "-o" && i + 1 < argc) out = argv[++i];
    }

    auto cfg = loadCSV(in);
    if (cfg.empty()) {
        cerr << "No configs loaded from " << in << "\n";
        return 1;
    }

    map<int, Cfg> best = cfg;

    double initSum = 0.0;
    for (auto &kv : best) initSum += kv.second.score();

    cout << fixed << setprecision(6);
    cout << "Loaded " << best.size() << " configs from " << in << "\n";
    cout << "Initial total score: " << initSum << "\n\n";

    const int MAX_DEPTH = 20;  // 한 시작 n에서 최대 20번 연속 outer 제거

    // 각 startN에서, outermostIndexBySquare를 연속으로 최대 20번까지 적용해서
    // n -> n-1 -> ... 식으로 체인으로 내려가며 best[n-1], best[n-2], ... 갱신
    for (int startN = 200; startN >= 2; --startN) {
        if (!best.count(startN)) continue;

        Cfg chain = best[startN];
        chain.updAll();

        cout << "### start from n=" << setw(3) << startN
             << " (score=" << chain.score() << ")\n";

        for (int depth = 1; depth <= MAX_DEPTH; ++depth) {
            int curN = chain.n;
            if (curN <= 1) break;
            int targetN = curN - 1;
            if (targetN < 1) break;

            int outerIdx = chain.outermostIndexBySquare();
            if (outerIdx < 0 || outerIdx >= chain.n) {
                cout << "  [break] invalid outerIdx at curN=" << curN << "\n";
                break;
            }

            Cfg cand = chain.removeTree(outerIdx);
            double candScore = cand.score();

            bool hadBase = best.count(targetN);
            double baseScore = hadBase ? best[targetN].score() : 1e18;

            if (!hadBase || candScore < baseScore - 1e-12) {
                best[targetN] = cand;
                double diff = hadBase ? (baseScore - candScore) : 0.0;
                double pct  = (hadBase && baseScore > 0.0) ? diff / baseScore * 100.0 : 0.0;

                cout << "  [update depth=" << setw(2) << depth << "] "
                     << "n=" << setw(3) << targetN
                     << " improved from chain starting n=" << setw(3) << startN
                     << " (curN=" << curN << ", idx=" << outerIdx << "): ";
                if (hadBase) {
                    cout << baseScore << " -> " << candScore
                         << " (" << pct << "%)\n";
                } else {
                    cout << "(no previous) -> " << candScore << "\n";
                }
            } else {
                cout << "  [keep   depth=" << setw(2) << depth << "] "
                     << "n=" << setw(3) << targetN
                     << " not improved from chain starting n=" << setw(3) << startN
                     << " (curN=" << curN << ", idx=" << outerIdx << ") "
                     << "base=" << baseScore << " cand=" << candScore << "\n";
            }

            // 다음 단계는 방금 제거된 cand에서 다시 outer 제거 → 재귀적 체인
            chain = cand;
        }
    }

    double finSum = 0.0;
    for (auto &kv : best) finSum += kv.second.score();

    cout << "\n========================================\n";
    cout << "Initial total score: " << initSum << "\n";
    cout << "Final   total score: " << finSum << "\n";
    cout << "Improve           : " << (initSum - finSum)
         << " (" << ((initSum - finSum) / initSum * 100.0) << "%)\n";
    cout << "========================================\n";

    saveCSV(out, best);
    cout << "Saved to " << out << "\n";

    return 0;
}
