"""
Debug dense block generation - see actual scores produced.
"""
import numpy as np
import pandas as pd
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.optimize import minimize
from numba import njit
import math

getcontext().prec = 25
scale_factor = Decimal('1e18')

TX = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125])
TY = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5])

class ChristmasTree:
    def __init__(self, center_x='0', center_y='0', angle='0'):
        self.center_x = Decimal(str(center_x))
        self.center_y = Decimal(str(center_y))
        self.angle = Decimal(str(angle))
        sf = Decimal('1e18')
        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h
        initial_polygon = Polygon([
            (Decimal('0.0') * sf, tip_y * sf),
            (top_w / Decimal('2') * sf, tier_1_y * sf),
            (top_w / Decimal('4') * sf, tier_1_y * sf),
            (mid_w / Decimal('2') * sf, tier_2_y * sf),
            (mid_w / Decimal('4') * sf, tier_2_y * sf),
            (base_w / Decimal('2') * sf, base_y * sf),
            (trunk_w / Decimal('2') * sf, base_y * sf),
            (trunk_w / Decimal('2') * sf, trunk_bottom_y * sf),
            (-(trunk_w / Decimal('2')) * sf, trunk_bottom_y * sf),
            (-(trunk_w / Decimal('2')) * sf, base_y * sf),
            (-(base_w / Decimal('2')) * sf, base_y * sf),
            (-(mid_w / Decimal('4')) * sf, tier_2_y * sf),
            (-(mid_w / Decimal('2')) * sf, tier_2_y * sf),
            (-(top_w / Decimal('4')) * sf, tier_1_y * sf),
            (-(top_w / Decimal('2')) * sf, tier_1_y * sf),
        ])
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x * sf),
                                          yoff=float(self.center_y * sf))

def gen_block(x_len, y_len, deg1, deg2, shift_x1, shift_y1, shift_x2, shift_y, sign):
    x = 0
    y0 = 0
    eps = 0.00000000000001
    data = []
    k = 0
    for i in range(x_len):
        if i % 2 == 1:
            angle = deg2
            x += shift_x1 + eps
            y0 += shift_y1 + eps
        else:
            angle = deg1
            x += shift_x2 + eps
            y0 += -shift_y1 - eps * sign
        y = y0
        for j in range(y_len):
            y += shift_y + eps
            data.append(['_'+str(k), 's' + str(x), 's' + str(y), 's' + str(angle)])
            k = k + 1
    return pd.DataFrame(columns=['id', 'x', 'y', 'deg'], data=data)

def find_shift_y1(deg, shift_x1):
    tree0 = ChristmasTree(0, 0, deg)
    def fun_min(shift_y1_list):
        tree1 = ChristmasTree(shift_x1, shift_y1_list[0], deg - 180)
        if tree1.polygon.intersects(tree0.polygon):
            return 1000
        if tree1.polygon.bounds[3] < tree0.polygon.bounds[1] + 0.3 * 1e18:
            return 1000
        return shift_y1_list[0]
    res = minimize(fun_min, [20], method='Powell')
    return res.x[0]

def find_shift_x2(deg, shift_x1, shift_y1):
    tree00 = ChristmasTree(0, 0, deg)
    tree01 = ChristmasTree(shift_x1, shift_y1, deg - 180)
    pair0 = unary_union([tree00.polygon, tree01.polygon])
    def fun_min(shift_x2_list):
        tree10 = ChristmasTree(shift_x1 + shift_x2_list[0], 0, deg)
        tree11 = ChristmasTree(shift_x1 * 2 + shift_x2_list[0], shift_y1, deg - 180)
        pair1 = unary_union([tree10.polygon, tree11.polygon])
        if pair1.intersects(pair0):
            return 1000
        if pair1.bounds[2] < pair0.bounds[0] + 0.3 * 1e18:
            return 1000
        return shift_x2_list[0]
    res = minimize(fun_min, [20], method='Powell')
    return res.x[0]

def find_shift_y2(deg, shift_x1, shift_y1, shift_x2):
    tree00 = ChristmasTree(0, 0, deg)
    tree01 = ChristmasTree(shift_x1, shift_y1, deg - 180)
    tree02 = ChristmasTree(shift_x1 + shift_x2, 0, deg)
    tree03 = ChristmasTree(shift_x1 * 2 + shift_x2, shift_y1, deg - 180)
    layer0 = unary_union([tree00.polygon, tree01.polygon, tree02.polygon, tree03.polygon])
    def fun_min(shift_y2_list):
        tree10 = ChristmasTree(0, shift_y2_list[0], deg)
        tree11 = ChristmasTree(shift_x1, shift_y1 + shift_y2_list[0], deg - 180)
        tree12 = ChristmasTree(shift_x1 + shift_x2, shift_y2_list[0], deg)
        tree13 = ChristmasTree(shift_x1 * 2 + shift_x2, shift_y1 + shift_y2_list[0], deg - 180)
        layer1 = unary_union([tree10.polygon, tree11.polygon, tree12.polygon, tree13.polygon])
        if layer1.intersects(layer0):
            return 1000
        if layer1.bounds[3] < layer0.bounds[1] + 0.3 * 1e18:
            return 1000
        return shift_y2_list[0]
    res = minimize(fun_min, [20], method='Powell')
    return res.x[0]

def gen_dense_block1(x_len, y_len, deg, d):
    shift_x1 = np.abs(d * np.sin(deg * np.pi / 360))
    shift_y1 = find_shift_y1(deg, shift_x1)
    shift_x2 = find_shift_x2(deg, shift_x1, shift_y1)
    shift_y2 = find_shift_y2(deg, shift_x1, shift_y1, shift_x2)
    return gen_block(x_len, y_len, deg, deg-180, shift_x1, shift_y1, shift_x2, shift_y2, 1)

@njit
def compute_bbox_score(xs, ys, angles, tx, ty):
    n = len(xs)
    V = len(tx)
    mnx = 1e300
    mny = 1e300
    mxx = -1e300
    mxy = -1e300
    for i in range(n):
        r = angles[i] * math.pi / 180.0
        c = math.cos(r)
        s = math.sin(r)
        xi = xs[i]
        yi = ys[i]
        for j in range(V):
            X = c * tx[j] - s * ty[j] + xi
            Y = s * tx[j] + c * ty[j] + yi
            if X < mnx: mnx = X
            if X > mxx: mxx = X
            if Y < mny: mny = Y
            if Y > mxy: mxy = Y
    side = max(mxx - mnx, mxy - mny)
    return side * side / n

def strip(v):
    return float(str(v).replace("s", ""))

def df_to_arrays(df):
    xs = np.array([strip(v) for v in df['x']])
    ys = np.array([strip(v) for v in df['y']])
    angles = np.array([strip(v) for v in df['deg']])
    return xs, ys, angles

# Load baseline
baseline_df = pd.read_csv('/home/submission/submission.csv')
baseline_df['N'] = baseline_df['id'].str.split('_').str[0].astype(int)

# Test specific configuration from the kernel (N=168)
print("Testing the exact configuration from the kernel (N=168):")
print("  gen_dense_block1(12, 14, 248.19859051364818, 1.1014707194584321)")

df = gen_dense_block1(12, 14, 248.19859051364818, 1.1014707194584321)
print(f"  Generated {len(df)} trees")

# Take first 168 trees
df_168 = df.head(168)
xs, ys, angles = df_to_arrays(df_168)
dense_score = compute_bbox_score(xs, ys, angles, TX, TY)

# Baseline for N=168
g = baseline_df[baseline_df['N'] == 168]
xs_b, ys_b, angles_b = df_to_arrays(g)
baseline_score = compute_bbox_score(xs_b, ys_b, angles_b, TX, TY)

print(f"  Dense block score for N=168: {dense_score:.6f}")
print(f"  Baseline score for N=168: {baseline_score:.6f}")
print(f"  Difference: {dense_score - baseline_score:.6f}")

# Test a few more N values
print("\nTesting various N values:")
for n in [50, 100, 150, 178, 200]:
    # Find best grid for this N
    best_score = float('inf')
    best_config = None
    
    for x_len in range(2, 20):
        for y_len in range(2, 20):
            if x_len * y_len < n:
                continue
            if x_len * y_len > n + 20:
                continue
            
            try:
                df = gen_dense_block1(x_len, y_len, 248.19859051364818, 1.1014707194584321)
                df = df.head(n)
                if len(df) < n:
                    continue
                xs, ys, angles = df_to_arrays(df)
                score = compute_bbox_score(xs, ys, angles, TX, TY)
                if score < best_score:
                    best_score = score
                    best_config = (x_len, y_len)
            except:
                continue
    
    # Baseline
    g = baseline_df[baseline_df['N'] == n]
    xs_b, ys_b, angles_b = df_to_arrays(g)
    baseline_score = compute_bbox_score(xs_b, ys_b, angles_b, TX, TY)
    
    print(f"  N={n}: Dense={best_score:.6f} (grid={best_config}), Baseline={baseline_score:.6f}, Diff={best_score - baseline_score:+.6f}")
