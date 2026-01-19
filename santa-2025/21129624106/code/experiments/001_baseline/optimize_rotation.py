import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar
from decimal import Decimal, getcontext

# Decimal precision
getcontext().prec = 30
scale_factor = Decimal("1")

ROT_EPSILON = 1e-7
ROT_ANGLE_MAX = 89.999
ROT_GROUP_MAX = 200

class ChristmasTree:
    def __init__(self, center_x="0", center_y="0", angle="0"):
        self.center_x = Decimal(str(center_x))
        self.center_y = Decimal(str(center_y))
        self.angle = Decimal(str(angle))

        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w  = Decimal("0.4")
        top_w  = Decimal("0.25")
        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                (Decimal("0.0") * scale_factor, tip_y * scale_factor),
                (top_w / Decimal("2") * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal("4") * scale_factor, tier_1_y * scale_factor),
                (mid_w / Decimal("2") * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal("4") * scale_factor, tier_2_y * scale_factor),
                (base_w / Decimal("2") * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal("2") * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal("2") * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal("2")) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal("2")) * scale_factor, base_y * scale_factor),
                (-(base_w / Decimal("2")) * scale_factor, base_y * scale_factor),
                (-(mid_w / Decimal("4")) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal("2")) * scale_factor, tier_2_y * scale_factor),
                (-(top_w / Decimal("4")) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal("2")) * scale_factor, tier_1_y * scale_factor),
            ]
        )

        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor),
        )

    def clone(self):
        return ChristmasTree(center_x=str(self.center_x), center_y=str(self.center_y), angle=str(self.angle))

def get_tree_list_side_lenght(tree_list):
    all_polygons = [t.polygon for t in tree_list]
    bounds = unary_union(all_polygons).bounds
    return Decimal(max(bounds[2] - bounds[0], bounds[3] - bounds[1])) / scale_factor

def get_total_score(dict_of_side_length):
    score = Decimal("0")
    for k, v in dict_of_side_length.items():
        score += v ** 2 / Decimal(str(k))
    return score

def parse_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["x"] = df["x"].astype(str).str.strip().str.lstrip("s")
    df["y"] = df["y"].astype(str).str.strip().str.lstrip("s")
    df["deg"] = df["deg"].astype(str).str.strip().str.lstrip("s")
    df[["group_id", "item_id"]] = df["id"].str.split("_", n=2, expand=True)

    dict_of_tree_list = {}
    dict_of_side_length = {}

    for group_id, group_data in df.groupby("group_id"):
        tree_list = [
            ChristmasTree(center_x=row["x"], center_y=row["y"], angle=row["deg"])
            for _, row in group_data.iterrows()
        ]
        dict_of_tree_list[group_id] = tree_list
        dict_of_side_length[group_id] = get_tree_list_side_lenght(tree_list)

    return dict_of_tree_list, dict_of_side_length

def calculate_bbox_side_at_angle(angle_deg, points):
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix_T = np.array([[c, s], [-s, c]])
    rotated_points = points.dot(rot_matrix_T)
    min_xy = np.min(rotated_points, axis=0)
    max_xy = np.max(rotated_points, axis=0)
    return max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])

def optimize_rotation(trees, angle_max=ROT_ANGLE_MAX, epsilon=ROT_EPSILON):
    all_points = []
    for tree in trees:
        all_points.extend(list(tree.polygon.exterior.coords))
    points_np = np.array(all_points)

    hull_points = points_np[ConvexHull(points_np).vertices]
    initial_side = calculate_bbox_side_at_angle(0, hull_points)

    res = minimize_scalar(
        lambda a: calculate_bbox_side_at_angle(a, hull_points),
        bounds=(0.001, float(angle_max)),
        method="bounded",
    )

    found_angle_deg = float(res.x)
    found_side = float(res.fun)

    improvement = initial_side - found_side
    if improvement > float(epsilon):
        best_angle_deg = found_angle_deg
        best_side = Decimal(str(found_side)) / scale_factor
    else:
        best_angle_deg = 0.0
        best_side = Decimal(str(initial_side)) / scale_factor

    return best_side, best_angle_deg

def apply_rotation(trees, angle_deg):
    if not trees or abs(angle_deg) < 1e-12:
        return [t.clone() for t in trees]

    bounds = [t.polygon.bounds for t in trees]
    min_x = min(b[0] for b in bounds)
    min_y = min(b[1] for b in bounds)
    max_x = max(b[2] for b in bounds)
    max_y = max(b[3] for b in bounds)
    rotation_center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])

    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[c, -s], [s, c]])

    points = np.array([[float(t.center_x), float(t.center_y)] for t in trees])
    shifted = points - rotation_center
    rotated = shifted.dot(rot_matrix.T) + rotation_center

    rotated_trees = []
    for i in range(len(trees)):
        new_tree = ChristmasTree(
            Decimal(rotated[i, 0]),
            Decimal(rotated[i, 1]),
            Decimal(trees[i].angle + Decimal(str(angle_deg))),
        )
        rotated_trees.append(new_tree)

    return rotated_trees

def write_submission(dict_of_tree_list, out_file):
    rows = []
    for group_name, tree_list in dict_of_tree_list.items():
        for item_id, tree in enumerate(tree_list):
            rows.append(
                {
                    "id": f"{group_name}_{item_id}",
                    "x": f"s{tree.center_x}",
                    "y": f"s{tree.center_y}",
                    "deg": f"s{tree.angle}",
                }
            )
    pd.DataFrame(rows).to_csv(out_file, index=False)

def fix_direction(in_csv, out_csv, passes=1):
    dict_of_tree_list, dict_of_side_length = parse_csv(in_csv)
    current_score = get_total_score(dict_of_side_length)
    print(f"Initial score before rotation: {current_score}")

    for _ in range(int(passes)):
        changed = False
        for group_id_main in range(ROT_GROUP_MAX, 0, -1): # Changed to 0 to include all
            gid = f"{int(group_id_main):03d}"
            if gid not in dict_of_tree_list:
                continue

            trees = dict_of_tree_list[gid]
            best_side, best_angle_deg = optimize_rotation(trees)
            if best_side < dict_of_side_length[gid]:
                dict_of_tree_list[gid] = apply_rotation(trees, best_angle_deg)
                dict_of_side_length[gid] = best_side
                changed = True

        new_score = get_total_score(dict_of_side_length)
        print(f"Score after pass: {new_score}")
        if new_score >= current_score or not changed:
            current_score = new_score
            break
        current_score = new_score

    write_submission(dict_of_tree_list, out_csv)
    return float(current_score)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python optimize_rotation.py input.csv output.csv [passes]")
    else:
        passes = 1
        if len(sys.argv) > 3:
            passes = int(sys.argv[3])
        fix_direction(sys.argv[1], sys.argv[2], passes)
