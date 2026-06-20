import argparse
import itertools
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

from ICVRP import build_qubo, run_ma_qaoa as run_icvrp_ma_qaoa
from ICVRP import run_standard_qaoa as run_icvrp_standard_qaoa
from OTSP import build_qubo_ostp, run_ma_qaoa as run_otsp_ma_qaoa
from OTSP import run_standard_qaoa as run_otsp_standard_qaoa
from scripts.Preprocess import constrained_kmeans, find_entry_exit


ICVRP_VAR_LIST = [
    "x_D_C1", "x_D_C2", "x_D_C3",
    "x_C1_D", "x_C1_C2", "x_C1_C3",
    "x_C2_D", "x_C2_C1", "x_C2_C3",
    "x_C3_D", "x_C3_C1", "x_C3_C2",
]

OTSP_VAR_LIST = [f"x_{i}_{j}" for i in range(4) for j in range(4) if i != j]


def save_pickle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_customers(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def prob_tag(prob):
    if isinstance(prob, str) and prob.startswith("prob_"):
        return prob
    return f"prob_00{int(prob)}"


def distance_matrix(points):
    n = len(points)
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                w[i, j] = np.sqrt(np.sum((points[i] - points[j]) ** 2))
    return w


def cluster_distance_matrix(customers, point_indices):
    points = customers[np.sort(point_indices)]
    return distance_matrix(points)


def icvrp_distance_matrix(customers, centroids):
    depot = np.mean(customers, axis=0)
    return distance_matrix([depot] + list(centroids))


def calculate_icvrp_distance(bitstring, var_list, w, label_to_index):
    dist = 0.0
    for i, bit in enumerate(bitstring):
        if bit == "1":
            _, src, dst = var_list[i].split("_")
            dist += w[label_to_index[src], label_to_index[dst]]
    return dist


def calculate_energy(bitstring, Q, g, c):
    x = np.array([int(bit) for bit in bitstring])
    return x.T @ Q @ x + g @ x + c


def check_icvrp_feasible(bitstring, var_list):
    x = np.array([int(bit) for bit in bitstring])
    checks = [
        x[var_list.index("x_D_C1")] + x[var_list.index("x_C2_C1")] + x[var_list.index("x_C3_C1")] == 1,
        x[var_list.index("x_D_C2")] + x[var_list.index("x_C1_C2")] + x[var_list.index("x_C3_C2")] == 1,
        x[var_list.index("x_D_C3")] + x[var_list.index("x_C1_C3")] + x[var_list.index("x_C2_C3")] == 1,
        x[var_list.index("x_C1_D")] + x[var_list.index("x_C1_C2")] + x[var_list.index("x_C1_C3")] == 1,
        x[var_list.index("x_C2_D")] + x[var_list.index("x_C2_C1")] + x[var_list.index("x_C2_C3")] == 1,
        x[var_list.index("x_C3_D")] + x[var_list.index("x_C3_C1")] + x[var_list.index("x_C3_C2")] == 1,
        x[var_list.index("x_D_C1")] + x[var_list.index("x_D_C2")] + x[var_list.index("x_D_C3")] == 2,
        x[var_list.index("x_C1_D")] + x[var_list.index("x_C2_D")] + x[var_list.index("x_C3_D")] == 2,
    ]
    return all(checks)


def otsp_path_from_bitstring(bitstring, var_list, n, start_idx, end_idx):
    succ = {}
    for bit, var in zip(bitstring, var_list):
        if bit != "1":
            continue
        _, src, dst = var.split("_")
        src, dst = int(src), int(dst)
        if src in succ:
            return None
        succ[src] = dst

    path = [start_idx]
    current = start_idx
    seen = {start_idx}
    while current != end_idx:
        if current not in succ:
            return None
        current = succ[current]
        if current in seen:
            return None
        seen.add(current)
        path.append(current)

    if len(path) != n or set(path) != set(range(n)):
        return None
    return path


def check_otsp_feasible(bitstring, var_list, n, start_idx, end_idx):
    x = {var: int(bit) for var, bit in zip(var_list, bitstring)}
    in_deg = [0] * n
    out_deg = [0] * n
    for var, value in x.items():
        if value != 1:
            continue
        _, src, dst = var.split("_")
        src, dst = int(src), int(dst)
        out_deg[src] += 1
        in_deg[dst] += 1

    for node in range(n):
        expected_out = 0 if node == end_idx else 1
        expected_in = 0 if node == start_idx else 1
        if out_deg[node] != expected_out or in_deg[node] != expected_in:
            return False

    return otsp_path_from_bitstring(bitstring, var_list, n, start_idx, end_idx) is not None


def calculate_otsp_distance(bitstring, var_list, w):
    dist = 0.0
    for bit, var in zip(bitstring, var_list):
        if bit == "1":
            _, src, dst = var.split("_")
            dist += w[int(src), int(dst)]
    return dist


def nearest_feasible(bitstring, var_list, is_feasible, score):
    bits = np.array([int(bit) for bit in bitstring])
    best = None
    best_score = float("inf")
    flip_count = 0

    for n_flips in range(1, 3):
        for positions in itertools.combinations(range(len(var_list)), n_flips):
            flip_count += 1
            candidate = bits.copy()
            for pos in positions:
                candidate[pos] = 1 - candidate[pos]
            candidate_str = "".join(map(str, candidate))
            if is_feasible(candidate_str):
                candidate_score = score(candidate_str)
                if candidate_score < best_score:
                    best = candidate_str
                    best_score = candidate_score
                break
        if best is not None:
            break

    return best, best_score, flip_count


def post_process_counts(counts, var_list, is_feasible, distance_fn, energy_fn, prob, num_states=100):
    feasible = []
    for bitstring, probability in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:num_states]:
        if probability <= prob:
            continue
        if is_feasible(bitstring):
            feasible.append((bitstring, distance_fn(bitstring), energy_fn(bitstring), 0))
            continue
        repaired, repair_distance, flips = nearest_feasible(bitstring, var_list, is_feasible, distance_fn)
        if repaired is not None:
            feasible.append((repaired, repair_distance, energy_fn(repaired), flips))

    if not feasible:
        return [], []

    min_distance = min(item[1] for item in feasible)
    bitstrings = []
    flips = []
    for bitstring, distance, _, flip_count in feasible:
        if distance == min_distance and bitstring not in bitstrings:
            bitstrings.append(bitstring)
            flips.append(flip_count)
    return bitstrings, flips


def run_raw_icvrp(customers, centroids, p, run_seed):
    w = icvrp_distance_matrix(customers, centroids)
    return {
        "std_counts": [run_icvrp_standard_qaoa(w, p=p, seed=run_seed)],
        "ma_counts": [run_icvrp_ma_qaoa(w, p=p, seed=run_seed)],
        "distance_matrix": w,
    }


def run_raw_otsp(customers, assignments, centroids, p, run_seed):
    depot = np.mean(customers, axis=0)
    results = {"cluster1": [], "cluster2": [], "cluster3": []}
    contexts = {}

    for cluster_id in range(3):
        point_indices = np.where(assignments == cluster_id)[0]
        point_indices_sorted = np.sort(point_indices)
        orig_to_local = {point_idx: idx for idx, point_idx in enumerate(point_indices_sorted)}
        dist_mat = cluster_distance_matrix(customers, point_indices)
        entry_orig, exit_orig = find_entry_exit(point_indices, customers, cluster_id, centroids, depot)
        entry_local = orig_to_local[entry_orig]
        exit_local = orig_to_local[exit_orig]

        std_counts = run_otsp_standard_qaoa(4, dist_mat, entry_local, exit_local, p=p, seed=run_seed)
        ma_counts = run_otsp_ma_qaoa(4, dist_mat, entry_local, exit_local, p=p, seed=run_seed)

        results[f"cluster{cluster_id + 1}"].append({
            "std_qaoa": [std_counts],
            "ma_qaoa": [ma_counts],
        })
        contexts[cluster_id + 1] = {
            "dist_mat": dist_mat,
            "entry_local": entry_local,
            "exit_local": exit_local,
        }

    return results, contexts


def post_process_icvrp(raw_icvrp, data_id, prob):
    w = raw_icvrp["distance_matrix"]
    var_list, label_to_index, Q, g, c, _, _, _, _ = build_qubo(w)

    def process(counts):
        return post_process_counts(
            counts,
            var_list,
            lambda bitstring: check_icvrp_feasible(bitstring, var_list),
            lambda bitstring: calculate_icvrp_distance(bitstring, var_list, w, label_to_index),
            lambda bitstring: calculate_energy(bitstring, Q, g, c),
            prob,
        )

    std_bits, std_flips = process(raw_icvrp["std_counts"][0])
    ma_bits, ma_flips = process(raw_icvrp["ma_counts"][0])
    return (
        {"dict_prob": {data_id: std_bits}, "dict_flip": {data_id: std_flips}},
        {"dict_prob": {data_id: ma_bits}, "dict_flip": {data_id: ma_flips}},
    )


def post_process_otsp(raw_otsp, contexts, run_seed, prob):
    std_posts = {}
    ma_posts = {}
    for cluster in range(1, 4):
        context = contexts[cluster]
        dist_mat = context["dist_mat"]
        entry_local = context["entry_local"]
        exit_local = context["exit_local"]
        var_list, _, Q, g, c, _, _, _, _ = build_qubo_ostp(4, dist_mat, entry_local, exit_local)
        cluster_result = raw_otsp[f"cluster{cluster}"][0]

        def process(counts):
            return post_process_counts(
                counts,
                var_list,
                lambda bitstring: check_otsp_feasible(bitstring, var_list, 4, entry_local, exit_local),
                lambda bitstring: calculate_otsp_distance(bitstring, var_list, dist_mat),
                lambda bitstring: calculate_energy(bitstring, Q, g, c),
                prob,
            )

        std_bits, std_flips = process(cluster_result["std_qaoa"][0])
        ma_bits, ma_flips = process(cluster_result["ma_qaoa"][0])
        std_posts[cluster] = {run_seed: {"dict_prob": std_bits, "dict_flip": std_flips}}
        ma_posts[cluster] = {run_seed: {"dict_prob": ma_bits, "dict_flip": ma_flips}}

    return std_posts, ma_posts


def write_pipeline_outputs(root, data_id, run_seed, p, prob_label, raw_icvrp, raw_otsp, icvrp_posts, otsp_posts):
    paths = {}
    raw_icvrp_path = root / "results_raw" / f"results_dataset_{data_id}" / f"p_{p}" / f"run_{run_seed}.pkl"
    raw_otsp_path = root / "results_raw_otsp" / f"results_dataset_{data_id}" / f"p_{p}" / f"run_{run_seed}.pkl"
    save_pickle(raw_icvrp_path, raw_icvrp)
    save_pickle(raw_otsp_path, raw_otsp)
    paths["raw_icvrp"] = raw_icvrp_path
    paths["raw_otsp"] = raw_otsp_path

    for variant, post in (("std", icvrp_posts[0]), ("ma", icvrp_posts[1])):
        path = root / "results_post" / f"results_dataset_{variant}_qaoa" / f"p_{p}" / prob_label / f"run_{run_seed}.pkl"
        save_pickle(path, post)
        paths[f"post_icvrp_{variant}"] = path

    for variant, posts in (("std", otsp_posts[0]), ("ma", otsp_posts[1])):
        for cluster, post in posts.items():
            path = (
                root / "results_post_otsp" / f"results_dataset_{variant}_qaoa"
                / f"p_{p}" / f"{prob_label}_cluster{cluster}" / f"dataset_{data_id}.pkl"
            )
            save_pickle(path, post)
            paths[f"post_otsp_{variant}_cluster{cluster}"] = path

    return paths


def merge_outputs(root, customers, assignments, centroids, data_id, run_seed, p, prob_label, variant, icvrp_post, otsp_post):
    sys.path.insert(0, str(root / "scripts"))
    from merge_otsp_icvrp import cluster_context, merge_run

    depot = np.mean(customers, axis=0)
    context = cluster_context(customers, assignments, centroids, depot)
    icvrp_bits = icvrp_post["dict_prob"].get(data_id, [])
    otsp_bits = {
        cluster: otsp_post[cluster][run_seed]["dict_prob"]
        for cluster in range(1, 4)
    }
    best_solution = merge_run(customers, context, icvrp_bits, otsp_bits)
    best_solution["run_seed"] = run_seed

    result = {
        "data_id": data_id,
        "variant": variant,
        "p": p,
        "prob": prob_label,
        "run_start": run_seed,
        "run_end": run_seed,
        "merged_run_count": 1,
        "skipped_run_count": 0,
        "best_solution": best_solution,
    }
    out_path = root / "results_merged" / f"results_dataset_{variant}_qaoa" / f"p_{p}" / prob_label / f"dataset_{data_id}.pkl"
    save_pickle(out_path, result)
    return out_path, result


def parse_args():
    parser = argparse.ArgumentParser(description="Run OTSP, ICVRP, post-processing, and merge for one dataset.")
    parser.add_argument("--data", default="data_example/dataset0.pkl")
    parser.add_argument("--data_id", type=int, default=0)
    parser.add_argument("--run_seed", type=int, default=0)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--prob", type=int, default=1)
    parser.add_argument("--variant", choices=["std", "ma"], default="ma")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path.cwd()
    data_path = (root / args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    prob_threshold = args.prob / 1000
    label = prob_tag(args.prob)
    customers = load_customers(data_path)
    assignments, centroids = constrained_kmeans(customers, random_state=args.run_seed)

    print(f"Running OTSP for {data_path}")
    raw_otsp, otsp_contexts = run_raw_otsp(customers, assignments, centroids, args.p, args.run_seed)

    print("Running ICVRP")
    raw_icvrp = run_raw_icvrp(customers, centroids, args.p, args.run_seed)

    print("Post-processing ICVRP and OTSP")
    icvrp_posts = post_process_icvrp(raw_icvrp, args.data_id, prob_threshold)
    otsp_posts = post_process_otsp(raw_otsp, otsp_contexts, args.run_seed, prob_threshold)
    output_paths = write_pipeline_outputs(
        root, args.data_id, args.run_seed, args.p, label, raw_icvrp, raw_otsp, icvrp_posts, otsp_posts
    )

    variant_index = 0 if args.variant == "std" else 1
    print(f"Merging {args.variant} outputs")
    merged_path, merged = merge_outputs(
        root,
        customers,
        assignments,
        centroids,
        args.data_id,
        args.run_seed,
        args.p,
        label,
        args.variant,
        icvrp_posts[variant_index],
        otsp_posts[variant_index],
    )

    output_paths["merged"] = merged_path
    print(f"Saved merged output: {merged_path}")
    print(f"Best total distance: {merged['best_solution']['total_distance']}")


if __name__ == "__main__":
    main()
