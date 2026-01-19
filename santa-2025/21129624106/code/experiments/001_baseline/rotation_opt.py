# ============================================================
# 3-HOUR FAST IMPROVEMENT RUNNER (bbox3 + smart filtering)
# - Coarse short runs to find promising (n,r)
# - Only runs expensive fix_direction + overlap validation on winners
# - Escalates timeout only for top candidates
# - Keeps best submission; reverts on regressions
# ============================================================

import os
import re
import csv
import time
import shutil
import subprocess
from glob import glob
from datetime import datetime
from decimal import Decimal, getcontext
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar


# ----------------------------
# USER CONFIG
# ----------------------------
BASELINE_CSV = "/kaggle/input/santa-submission/submission.csv"
BBOX3_BIN_IN = "/kaggle/input/santa-submission/bbox3"

WORK_SUBMISSION = "submission.csv"
WORK_BBOX3_BIN = "./bbox3"

OUT_DIR = "fast3h_out"
LOG_FILE = "fast3h.log"

TOTAL_BUDGET_SEC = 3 * 3600  # 3 hours hard budget (wall time)

# Only accept a candidate if it beats current best by at least this much.
# Increase if you want fewer expensive validations.
MIN_IMPROVEMENT_TO_PROCESS = 1e-10

# Rotation tightening controls (cheap-ish, but not free)
ROT_EPSILON = 1e-7
ROT_ANGLE_MAX = 89.999
ROT_GROUP_MAX = 200

# Decimal precision
getcontext().prec = 30
scale_factor = Decimal("1")


# ============================================================
# Core geometry model + scoring (grounded in your snippet)
# ============================================================

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

    # strip 's' prefix (matches your snippet)
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


# ============================================================
# Rotation tightening (fix_direction), grounded in your snippet
# ============================================================

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

    for _ in range(int(passes)):
        changed = False
        for group_id_main in range(ROT_GROUP_MAX, 2, -1):
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
        if new_score >= current_score or not changed:
            current_score = new_score
            break
        current_score = new_score

    write_submission(dict_of_tree_list, out_csv)
    return float(current_score)


# ============================================================
# Overlap check + targeted repair (grounded in your snippet)
# ============================================================

def has_overlap(trees):
    if len(trees) <= 1:
        return False

    polygons = [t.polygon for t in trees]
    tree_index = STRtree(polygons)

    for i, poly in enumerate(polygons):
        candidates = tree_index.query(poly)

        for cand in candidates:
            # Shapely 2.x may return indices, Shapely 1.8 may return geometries
            if isinstance(cand, (int, np.integer)):
                j = int(cand)
                if j == i:
                    continue
                other = polygons[j]
            else:
                if cand is poly:
                    continue
                other = cand

            if poly.intersects(other) and not poly.touches(other):
                return True

    return False


def score_and_validate_submission(file_path, max_n=ROT_GROUP_MAX):
    dict_of_tree_list, dict_of_side_length = parse_csv(file_path)

    failed = []
    for n in range(1, max_n + 1):
        gid = f"{n:03d}"
        trees = dict_of_tree_list.get(gid)
        if not trees:
            continue
        if has_overlap(trees):
            failed.append(n)

    total_score = float(get_total_score(dict_of_side_length))
    return {"total_score": total_score, "failed_overlap_n": failed, "ok": (len(failed) == 0)}


def load_groups(filename):
    groups = {}
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            full_id = row[0]
            group = full_id.split("_")[0]
            groups.setdefault(group, []).append(row)
    return header, groups


def replace_group(target_file, donor_file, group_id, output_file=None):
    if output_file is None:
        output_file = target_file

    header_t, groups_t = load_groups(target_file)
    _, groups_d = load_groups(donor_file)

    if group_id not in groups_d:
        raise ValueError(f"Donor file has no group {group_id}")

    groups_t[group_id] = groups_d[group_id]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header_t)
        for g in sorted(groups_t.keys(), key=lambda x: int(x)):
            for row in groups_t[g]:
                writer.writerow(row)


def repair_overlaps_in_place(submission_path, donor_path=BASELINE_CSV):
    res = score_and_validate_submission(submission_path, max_n=ROT_GROUP_MAX)
    if res["ok"]:
        return res

    for n in res["failed_overlap_n"]:
        replace_group(submission_path, donor_path, f"{n:03d}", submission_path)

    # one quick tighten after repair
    fix_direction(submission_path, submission_path, passes=1)
    return score_and_validate_submission(submission_path, max_n=ROT_GROUP_MAX)


# ============================================================
# bbox3 runner (fast: parse stdout, only process winners)
# ============================================================

FINAL_SCORE_RE = re.compile(r"Final\s+(?:Total\s+)?Score:\s*([0-9]+(?:\.[0-9]+)?)")


def parse_bbox3_final_score(stdout: str):
    m = FINAL_SCORE_RE.search(stdout or "")
    return float(m.group(1)) if m else None


def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def ensure_workspace():
    os.makedirs(OUT_DIR, exist_ok=True)

    shutil.copy(BASELINE_CSV, WORK_SUBMISSION)
    shutil.copy(BBOX3_BIN_IN, WORK_BBOX3_BIN)
    subprocess.run(["chmod", "+x", WORK_BBOX3_BIN], check=False)


def run_bbox3(timeout_sec, n_value, r_value):
    return subprocess.run(
        [WORK_BBOX3_BIN, "-n", str(n_value), "-r", str(r_value)],
        capture_output=True,
        text=True,
        timeout=int(timeout_sec),
    )


def save_snapshot(tag):
    snap = os.path.join(OUT_DIR, f"{tag}.csv")
    shutil.copy(WORK_SUBMISSION, snap)
    return snap


def fast3h_main():
    ensure_workspace()
    start = time.time()

    def time_left():
        return TOTAL_BUDGET_SEC - (time.time() - start)

    log("=" * 70)
    log(f"START {datetime.now().isoformat(timespec='seconds')}")
    log(f"BUDGET {TOTAL_BUDGET_SEC}s (~3h)")
    log("=" * 70)

    # Initial tighten + validate once (sets a strong baseline quickly)
    log("Initial fix_direction passes=1 ...")
    best_score = fix_direction(WORK_SUBMISSION, WORK_SUBMISSION, passes=1)
    val0 = repair_overlaps_in_place(WORK_SUBMISSION, donor_path=BASELINE_CSV)
    best_score = min(best_score, val0["total_score"])
    best_path = os.path.join(OUT_DIR, "best_submission.csv")
    shutil.copy(WORK_SUBMISSION, best_path)
    log(f"Initial best_score = {best_score:.14f} | overlap_ok={val0['ok']}")

    # ------------------------------------------------------------
    # 3-hour plan:
    # Phase A: many short runs to find promising settings (cheap)
    # Phase B: medium runs on top candidates
    # Phase C: long runs on the best few
    # ------------------------------------------------------------
    phaseA = {
        "timeout": 120,   # 2 min each
        "n_values": [1000, 1200, 1500, 1800, 2000],
        "r_values": [30, 60, 90],
        "top_k": 6,
        "fix_passes": 1,
    }
    phaseB = {
        "timeout": 600,   # 10 min each
        "top_k": 3,
        "fix_passes": 2,
    }
    phaseC = {
        "timeout": 1200,  # 20 min each
        "top_k": 2,
        "fix_passes": 3,
    }

    # Store candidates as list of dicts: {"n":..., "r":..., "score":...}
    candidates = []

    # ---------------- Phase A ----------------
    log("\n--- PHASE A (short runs) ---")
    for r in phaseA["r_values"]:
        for n in phaseA["n_values"]:
            if time_left() < 300:  # keep 5 min buffer
                log("Stopping Phase A due to low remaining time.")
                break

            log(f"[A] timeout={phaseA['timeout']} n={n} r={r} | time_left={time_left():.0f}s")

            try:
                res = run_bbox3(phaseA["timeout"], n, r)
            except subprocess.TimeoutExpired:
                log(f"[A] TIMEOUT n={n} r={r}")
                continue

            bbox_final = parse_bbox3_final_score(res.stdout)
            if bbox_final is None:
                log(f"[A] Could not parse Final Score for n={n} r={r}. Skipping.")
                continue

            # Only consider if bbox3 already beats best by a tiny margin
            if bbox_final < best_score - MIN_IMPROVEMENT_TO_PROCESS:
                log(f"[A] Promising: bbox3_final={bbox_final:.14f} < best={best_score:.14f}")
                candidates.append({"n": n, "r": r, "score": bbox_final})
            else:
                log(f"[A] Not better: bbox3_final={bbox_final:.14f} best={best_score:.14f}")

    candidates.sort(key=lambda x: x["score"])
    candidates = candidates[: phaseA["top_k"]]
    log(f"[A] Selected top {len(candidates)} candidates: {candidates}")

    # Process Phase A winners (run fix + validate only here)
    for c in list(candidates):
        if time_left() < 600:
            log("Not enough time to process Phase A winners further.")
            break

        log(f"[A->PROC] n={c['n']} r={c['r']} | fix_passes={phaseA['fix_passes']}")
        # re-run bbox3 to regenerate submission for this candidate (ensures file matches)
        try:
            run_bbox3(phaseA["timeout"], c["n"], c["r"])
        except subprocess.TimeoutExpired:
            continue

        fix_direction(WORK_SUBMISSION, WORK_SUBMISSION, passes=phaseA["fix_passes"])
        val = repair_overlaps_in_place(WORK_SUBMISSION, donor_path=BASELINE_CSV)

        cur = val["total_score"]
        save_snapshot(f"A_n{c['n']}_r{c['r']}_score{cur:.12f}")

        if val["ok"] and cur < best_score - MIN_IMPROVEMENT_TO_PROCESS:
            best_score = cur
            shutil.copy(WORK_SUBMISSION, best_path)
            log(f"[A->PROC] NEW BEST = {best_score:.14f}")
        else:
            shutil.copy(best_path, WORK_SUBMISSION)

    # ---------------- Phase B ----------------
    # Use the (possibly updated) candidates list, but refresh based on best_score
    candidates.sort(key=lambda x: x["score"])
    candidates = candidates[: phaseB["top_k"]]
    log("\n--- PHASE B (medium runs on top candidates) ---")
    log(f"[B] Candidates: {candidates}")

    for c in candidates:
        if time_left() < (phaseB["timeout"] + 600):
            log("Stopping Phase B due to low remaining time.")
            break

        log(f"[B] timeout={phaseB['timeout']} n={c['n']} r={c['r']}")
        try:
            res = run_bbox3(phaseB["timeout"], c["n"], c["r"])
        except subprocess.TimeoutExpired:
            log(f"[B] TIMEOUT n={c['n']} r={c['r']}")
            continue

        bbox_final = parse_bbox3_final_score(res.stdout) or 1e99
        if bbox_final >= best_score - MIN_IMPROVEMENT_TO_PROCESS:
            log(f"[B] bbox3_final not better ({bbox_final:.14f} vs best {best_score:.14f}), skipping fix/validate.")
            shutil.copy(best_path, WORK_SUBMISSION)
            continue

        fix_direction(WORK_SUBMISSION, WORK_SUBMISSION, passes=phaseB["fix_passes"])
        val = repair_overlaps_in_place(WORK_SUBMISSION, donor_path=BASELINE_CSV)
        cur = val["total_score"]

        save_snapshot(f"B_n{c['n']}_r{c['r']}_score{cur:.12f}")

        if val["ok"] and cur < best_score - MIN_IMPROVEMENT_TO_PROCESS:
            best_score = cur
            shutil.copy(WORK_SUBMISSION, best_path)
            log(f"[B] NEW BEST = {best_score:.14f}")
        else:
            shutil.copy(best_path, WORK_SUBMISSION)

    # ---------------- Phase C ----------------
    log("\n--- PHASE C (long runs on best few) ---")
    # Build a small set: take the same candidates, plus a quick neighborhood sweep around best n
    # (kept small to protect time)
    base_candidates = candidates[:]
    extra = []
    if base_candidates:
        best_c = base_candidates[0]
        n0, r0 = best_c["n"], best_c["r"]
        for dn in (-100, -50, 50, 100):
            extra.append({"n": max(1, n0 + dn), "r": r0})
        for dr in (-10, -5, 5, 10):
            extra.append({"n": n0, "r": max(1, r0 + dr)})

    # dedupe
    seen = set()
    phaseC_list = []
    for c in (base_candidates + extra):
        key = (int(c["n"]), int(c["r"]))
        if key not in seen:
            seen.add(key)
            phaseC_list.append({"n": key[0], "r": key[1], "score": c.get("score", 1e99)})

    phaseC_list = phaseC_list[: phaseC["top_k"]]
    log(f"[C] Candidates: {phaseC_list}")

    for c in phaseC_list:
        if time_left() < (phaseC["timeout"] + 600):
            log("Stopping Phase C due to low remaining time.")
            break

        log(f"[C] timeout={phaseC['timeout']} n={c['n']} r={c['r']}")
        try:
            res = run_bbox3(phaseC["timeout"], c["n"], c["r"])
        except subprocess.TimeoutExpired:
            log(f"[C] TIMEOUT n={c['n']} r={c['r']}")
            continue

        bbox_final = parse_bbox3_final_score(res.stdout) or 1e99
        if bbox_final >= best_score - MIN_IMPROVEMENT_TO_PROCESS:
            log(f"[C] bbox3_final not better ({bbox_final:.14f} vs best {best_score:.14f}), skipping fix/validate.")
            shutil.copy(best_path, WORK_SUBMISSION)
            continue

        fix_direction(WORK_SUBMISSION, WORK_SUBMISSION, passes=phaseC["fix_passes"])
        val = repair_overlaps_in_place(WORK_SUBMISSION, donor_path=BASELINE_CSV)
        cur = val["total_score"]

        save_snapshot(f"C_n{c['n']}_r{c['r']}_score{cur:.12f}")

        if val["ok"] and cur < best_score - MIN_IMPROVEMENT_TO_PROCESS:
            best_score = cur
            shutil.copy(WORK_SUBMISSION, best_path)
            log(f"[C] NEW BEST = {best_score:.14f}")
        else:
            shutil.copy(best_path, WORK_SUBMISSION)

    # Finalize
    shutil.copy(best_path, WORK_SUBMISSION)
    final_val = score_and_validate_submission(WORK_SUBMISSION, max_n=ROT_GROUP_MAX)

    log("\n" + "=" * 70)
    log(f"END {datetime.now().isoformat(timespec='seconds')}")
    log(f"BEST SCORE {best_score:.14f}")
    log(f"FINAL overlap_ok={final_val['ok']} failed={final_val['failed_overlap_n']}")
    log("=" * 70)

    # Zip outputs
    files = glob("*.csv") + glob("*.log") + glob(f"{OUT_DIR}/*.csv")
    zip_name = f"fast3h_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.zip"
    with ZipFile(zip_name, "w", compression=ZIP_DEFLATED, compresslevel=9) as z:
        for fn in files:
            z.write(fn)
    print("Saved:", zip_name)


# ----------------------------
# RUN
# ----------------------------
fast3h_main()