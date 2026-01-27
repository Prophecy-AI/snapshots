import pandas as pd
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
import time, random
import multiprocessing as mp

# --- Global settings ---
getcontext().prec = 50
scale_factor = Decimal('1e18')
scale_factor_float = 1e18

# ====== Hyperparameters: more aggressive = larger ======
PASSES = 6                  # Multiple synchronized passes (parallel-friendly)
EPS_IMPROVE = 1e-12
BOUND_EPS = 1.0

DEPTH = 10                   # Can be larger, but should be paired with BEAM/MAX_STATES
BEAM = 10
MAX_STATES = 4000

RAND_TRIES = 8
RAND_K = 50
RANDOM_SEED = 42

# ====== Parallel settings ======
PROCESSES = max(1, mp.cpu_count() - 2)  # Leave some CPU for the OS
CHUNKSIZE = 8                           # map chunk size; tune per machine


# --- Core class definition ---
class ChristmasTree:
    def __init__(self, center_x='0', center_y='0', angle='0'):
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)
        self.polygon = self._create_polygon()
        self.bounds = self.polygon.bounds  # (minx, miny, maxx, maxy)

    def _create_polygon(self):
        trunk_w = Decimal('0.15'); trunk_h = Decimal('0.2')
        base_w = Decimal('0.7'); base_y = Decimal('0.0')
        mid_w  = Decimal('0.4'); tier_2_y = Decimal('0.25')
        top_w  = Decimal('0.25'); tier_1_y = Decimal('0.5')
        tip_y  = Decimal('0.8'); trunk_bottom_y = -trunk_h

        coords = [
            (Decimal('0.0'), tip_y),
            (top_w / 2, tier_1_y), (top_w / 4, tier_1_y),
            (mid_w / 2, tier_2_y), (mid_w / 4, tier_2_y),
            (base_w / 2, base_y), (trunk_w / 2, base_y),
            (trunk_w / 2, trunk_bottom_y), (-(trunk_w / 2), trunk_bottom_y),
            (-(trunk_w / 2), base_y), (-(base_w / 2), base_y),
            (-(mid_w / 4), tier_2_y), (-(mid_w / 2), tier_2_y),
            (-(top_w / 4), tier_1_y), (-(top_w / 2), tier_1_y),
        ]

        scaled_coords = [(float(x) * scale_factor_float, float(y) * scale_factor_float) for x, y in coords]
        initial_polygon = Polygon(scaled_coords)

        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        return affinity.translate(
            rotated,
            xoff=float(self.center_x) * scale_factor_float,
            yoff=float(self.center_y) * scale_factor_float
        )

    def clone(self) -> "ChristmasTree":
        new_tree = ChristmasTree.__new__(ChristmasTree)
        new_tree.center_x = self.center_x
        new_tree.center_y = self.center_y
        new_tree.angle = self.angle
        new_tree.polygon = self.polygon
        new_tree.bounds = self.bounds
        return new_tree


# --- Bounds utilities ---
def get_bounds_side(bounds_list):
    if not bounds_list:
        return 0.0
    min_x = min(b[0] for b in bounds_list)
    min_y = min(b[1] for b in bounds_list)
    max_x = max(b[2] for b in bounds_list)
    max_y = max(b[3] for b in bounds_list)
    return max(max_x - min_x, max_y - min_y) / scale_factor_float

def compute_touching_candidates(bounds_list, eps=BOUND_EPS):
    n = len(bounds_list)
    if n == 0:
        return []
    min_x = min(b[0] for b in bounds_list)
    min_y = min(b[1] for b in bounds_list)
    max_x = max(b[2] for b in bounds_list)
    max_y = max(b[3] for b in bounds_list)
    cand = []
    for i, b in enumerate(bounds_list):
        if (abs(b[0] - min_x) < eps or abs(b[1] - min_y) < eps or
            abs(b[2] - max_x) < eps or abs(b[3] - max_y) < eps):
            cand.append(i)
    if not cand:
        cand = list(range(n))
    return cand


def choose_removal_beam_lookahead(bounds_list, depth, beam, max_states, rand_tries, rand_k, seed):
    """
    depth-step lookahead + beam search + random perturbations.
    Returns (best_first_idx, side_after_first_remove).
    """
    rng = random.Random(seed)
    n0 = len(bounds_list)
    if n0 <= 1:
        return None, 0.0

    def run_once(shuffle=True, limit_k=rand_k):
        base_cands = compute_touching_candidates(bounds_list)
        if shuffle:
            rng.shuffle(base_cands)
        if limit_k and len(base_cands) > limit_k:
            base_cands = base_cands[:limit_k]

        # First layer
        first_layer = []
        for idx in base_cands:
            reduced = bounds_list[:idx] + bounds_list[idx+1:]
            s1 = get_bounds_side(reduced)
            # (score_now, reduced_bounds, first_idx, first_s1)
            first_layer.append((s1, reduced, idx, s1))

        if not first_layer:
            return None, float("inf")

        first_layer.sort(key=lambda x: x[0])
        frontier = first_layer[:min(beam, len(first_layer))]

        # best_key: (future_best, first_s1)
        best_key = (frontier[0][0], frontier[0][3])
        best_first = frontier[0][2]
        best_s1 = frontier[0][3]

        states_used = len(frontier)

        # Expand to depth
        for _d in range(2, max(2, depth + 1)):
            new_frontier = []
            for score_now, bds, first_idx, first_s1 in frontier:
                if len(bds) <= 1:
                    key = (0.0, first_s1)
                    if key < best_key:
                        best_key, best_first, best_s1 = key, first_idx, first_s1
                    continue

                cands = compute_touching_candidates(bds)
                if shuffle:
                    rng.shuffle(cands)
                if limit_k and len(cands) > limit_k:
                    cands = cands[:limit_k]

                for j in cands:
                    nb = bds[:j] + bds[j+1:]
                    s = get_bounds_side(nb)
                    new_frontier.append((s, nb, first_idx, first_s1))
                    states_used += 1
                    if states_used >= max_states:
                        break
                if states_used >= max_states:
                    break

            if not new_frontier:
                break

            new_frontier.sort(key=lambda x: x[0])
            frontier = new_frontier[:min(beam, len(new_frontier))]

            cur_best = frontier[0]
            key = (cur_best[0], cur_best[3])
            if key < best_key:
                best_key, best_first, best_s1 = key, cur_best[2], cur_best[3]

            if states_used >= max_states:
                break

        return best_first, best_s1

    # First, do one "deterministic" run
    best_first, best_s1 = run_once(shuffle=False, limit_k=None)
    best_key = (best_s1, best_s1)

    # Multiple randomized runs: keep the best
    for _ in range(rand_tries):
        first, s1 = run_once(shuffle=True, limit_k=rand_k)
        if first is None:
            continue
        key = (s1, s1)
        if key < best_key:
            best_key = key
            best_first = first
            best_s1 = s1

    return best_first, best_s1


# --- Parallel worker: process one N and propose an improvement for N-1 ---
def worker_propose(args):
    """
    Input:
      (N, bounds_list, prev_best, depth, beam, max_states, rand_tries, rand_k, base_seed)
    Output:
      (target_gid, source_gid, remove_idx, new_side) or None
    """
    (N, bounds_list, prev_best, depth, beam, max_states, rand_tries, rand_k, base_seed) = args

    if bounds_list is None or len(bounds_list) <= 1:
        return None

    target_gid = f"{N-1:03d}"
    source_gid = f"{N:03d}"

    seed = (base_seed * 1000003) ^ (N * 9176) ^ (len(bounds_list) * 131)
    best_idx, best_s1 = choose_removal_beam_lookahead(
        bounds_list, depth, beam, max_states, rand_tries, rand_k, seed
    )

    if best_idx is None:
        return None

    if best_s1 < prev_best - EPS_IMPROVE:
        return (target_gid, source_gid, best_idx, best_s1)

    return None


# --- IO ---
def parse_csv(csv_path):
    print(f'Loading csv: {csv_path}')
    result = pd.read_csv(csv_path)

    for col in ['x', 'y', 'deg']:
        if result[col].dtype == object:
            result[col] = result[col].astype(str).str.strip().str.lstrip('s')

    result[['group_id', 'item_id']] = result['id'].astype(str).str.split('_', n=2, expand=True)

    dict_of_tree_list = {}
    dict_of_side_length = {}

    for group_id, group_data in result.groupby('group_id'):
        trees = []
        for _, row in group_data.iterrows():
            trees.append(ChristmasTree(center_x=row['x'], center_y=row['y'], angle=row['deg']))
        gid = f"{int(group_id):03d}"
        dict_of_tree_list[gid] = trees
        dict_of_side_length[gid] = get_bounds_side([t.bounds for t in trees])

    return dict_of_tree_list, dict_of_side_length


def save_dict_to_csv(dict_of_tree_list, output_path):
    print(f"Saving to {output_path}...")
    data = []
    sorted_keys = sorted(dict_of_tree_list.keys(), key=lambda x: int(x))
    for group_id in sorted_keys:
        trees = dict_of_tree_list[group_id]
        for i, tree in enumerate(trees):
            data.append({
                'id': f"{group_id}_{i}",
                'x': f"s{tree.center_x}",
                'y': f"s{tree.center_y}",
                'deg': f"s{tree.angle}"
            })
    pd.DataFrame(data)[['id', 'x', 'y', 'deg']].to_csv(output_path, index=False)
    print("Save complete.")


def main():
    INPUT_CSV = '/kaggle/working/submission.csv'
    OUTPUT_CSV = '/kaggle/working/submission.csv'

    try:
        dict_of_tree_list, dict_of_side_length = parse_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Not found {INPUT_CSV}")
        return

    start_time = time.time()
    print(
        f"MP HARD optimization: PROCESSES={PROCESSES}, PASSES={PASSES}, "
        f"DEPTH={DEPTH}, BEAM={BEAM}, MAX_STATES={MAX_STATES}, RAND_TRIES={RAND_TRIES}, RAND_K={RAND_K}"
    )

    # Windows compatibility: use spawn
    ctx = mp.get_context("spawn")

    changed_total = 0

    with ctx.Pool(processes=PROCESSES) as pool:
        for pass_id in range(1, PASSES + 1):
            print(f"\n=== PASS {pass_id}/{PASSES} ===")
            # Snapshot: all proposals in this pass are based on the same frozen data (parallel-safe)
            snap_tree_list = {k: v for k, v in dict_of_tree_list.items()}
            snap_side = {k: v for k, v in dict_of_side_length.items()}

            tasks = []
            base_seed = RANDOM_SEED + pass_id * 10007

            # Propose from N=200..3 to improve N-1
            for N in range(200, 2, -1):
                gidN = f"{N:03d}"
                gidPrev = f"{N-1:03d}"
                if gidN not in snap_tree_list or gidPrev not in snap_side:
                    continue
                bounds_list = [t.bounds for t in snap_tree_list[gidN]]
                prev_best = snap_side[gidPrev]
                tasks.append((N, bounds_list, prev_best, DEPTH, BEAM, MAX_STATES, RAND_TRIES, RAND_K, base_seed))

            # Compute proposals in parallel
            proposals = pool.map(worker_propose, tasks, chunksize=CHUNKSIZE)

            # For each target_gid, keep the best proposal (multiple N may target the same N-1)
            best_for_target = {}  # target_gid -> (new_side, source_gid, remove_idx)
            for p in proposals:
                if p is None:
                    continue
                target_gid, source_gid, remove_idx, new_side = p
                cur = best_for_target.get(target_gid)
                if cur is None or new_side < cur[0]:
                    best_for_target[target_gid] = (new_side, source_gid, remove_idx)

            # Apply improvements (single unified write-back)
            changed_this_pass = 0
            for target_gid, (new_side, source_gid, remove_idx) in best_for_target.items():
                old_side = dict_of_side_length.get(target_gid, float("inf"))
                if new_side < old_side - EPS_IMPROVE:
                    # Remove remove_idx from the snapshot solution of source_gid to form the new solution for target_gid
                    src_list = snap_tree_list[source_gid]
                    new_list = src_list[:remove_idx] + src_list[remove_idx+1:]
                    dict_of_tree_list[target_gid] = new_list
                    dict_of_side_length[target_gid] = new_side
                    print(f"[Group {target_gid}] Improved! {old_side:.6f} -> {new_side:.6f} (from {source_gid}, rm={remove_idx})")
                    changed_this_pass += 1
                    changed_total += 1

            print(f"PASS {pass_id} changes: {changed_this_pass}")
            if changed_this_pass == 0:
                print("No changes -> early stop.")
                break

    print(f"\nTotal changes: {changed_total}")
    print(f"Total time: {time.time() - start_time:.2f}s")
    save_dict_to_csv(dict_of_tree_list, OUTPUT_CSV)


if __name__ == '__main__':
    main()
