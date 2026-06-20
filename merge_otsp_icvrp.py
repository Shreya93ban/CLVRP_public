import argparse
import itertools
import math
import os
import pickle
from pathlib import Path

import numpy as np

from scripts.Preprocess import constrained_kmeans, find_entry_exit


ICVRP_VAR_LIST = [
    "x_D_C1",
    "x_D_C2",
    "x_D_C3",
    "x_C1_D",
    "x_C1_C2",
    "x_C1_C3",
    "x_C2_D",
    "x_C2_C1",
    "x_C2_C3",
    "x_C3_D",
    "x_C3_C1",
    "x_C3_C2",
]

OTSP_VAR_LIST = [f"x_{i}_{j}" for i in range(4) for j in range(4) if i != j]


def prob_tag(prob):
    """Return the shared probability folder name, e.g. prob_001."""
    if isinstance(prob, str):
        prob = prob.strip()
        if prob.startswith("prob_"):
            return prob
        prob = int(prob)

    return f"prob_00{int(prob)}"


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def selected_edges(bitstring, var_list):
    if len(bitstring) != len(var_list):
        raise ValueError(
            f"Bitstring length {len(bitstring)} does not match {len(var_list)} variables."
        )

    edges = []
    for bit, var in zip(bitstring, var_list):
        if bit == "1":
            _, src, dst = var.split("_")
            edges.append((src, dst))
    return edges


def icvrp_routes(bitstring):
    """Decode an ICVRP bitstring into depot-to-depot cluster routes."""
    successors = {src: dst for src, dst in selected_edges(bitstring, ICVRP_VAR_LIST)}
    starts = [dst for src, dst in selected_edges(bitstring, ICVRP_VAR_LIST) if src == "D"]
    routes = []

    for start in starts:
        route = ["D", start]
        seen = {start}
        current = start

        while current != "D":
            if current not in successors:
                raise ValueError(f"ICVRP route from {start} stops at {current}.")
            current = successors[current]
            if current != "D":
                if current in seen:
                    raise ValueError(f"ICVRP route from {start} contains a cycle.")
                seen.add(current)
            route.append(current)

        routes.append(route)

    return routes


def otsp_local_path(bitstring, start_local):
    """Decode a 4-node OTSP bitstring into a local node path."""
    successors = {}
    for bit, var in zip(bitstring, OTSP_VAR_LIST):
        if bit == "1":
            _, src, dst = var.split("_")
            successors[int(src)] = int(dst)

    path = [start_local]
    seen = {start_local}
    current = start_local

    while current in successors:
        current = successors[current]
        if current in seen:
            raise ValueError("OTSP path contains a cycle.")
        seen.add(current)
        path.append(current)

    if len(path) != 4:
        raise ValueError(f"OTSP path visits {len(path)} nodes, expected 4.")

    return path


def route_distance(points):
    return sum(
        math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        for i in range(len(points) - 1)
    )


def cluster_context(customers, assignments, centroids, depot):
    context = {}
    for cluster in range(3):
        point_indices = np.where(assignments == cluster)[0]
        point_indices_sorted = np.sort(point_indices)
        orig_to_local = {
            int(orig): int(local) for local, orig in enumerate(point_indices_sorted)
        }
        local_to_orig = {
            int(local): int(orig) for local, orig in enumerate(point_indices_sorted)
        }
        entry_orig, exit_orig = find_entry_exit(
            point_indices, customers, cluster, centroids, depot
        )
        entry_orig = int(entry_orig)
        exit_orig = int(exit_orig)

        context[cluster + 1] = {
            "point_indices": point_indices_sorted.tolist(),
            "entry_orig": entry_orig,
            "exit_orig": exit_orig,
            "entry_local": orig_to_local[entry_orig],
            "exit_local": orig_to_local[exit_orig],
            "local_to_orig": local_to_orig,
        }

    return context


def load_icvrp_bits(root, variant, p, prob, run_seed, data_id):
    path = (
        root
        / "results_post"
        / f"results_dataset_{variant}_qaoa"
        / f"p_{p}"
        / prob_tag(prob)
        / f"run_{run_seed}.pkl"
    )
    data = load_pickle(path)
    return data["dict_prob"].get(data_id, [])


def load_otsp_bits(root, variant, p, prob, run_seed, data_id):
    cluster_bits = {}
    for cluster in range(1, 4):
        path = (
            root
            / "results_post_otsp"
            / f"results_dataset_{variant}_qaoa"
            / f"p_{p}"
            / f"{prob_tag(prob)}_cluster{cluster}"
            / f"dataset_{data_id}.pkl"
        )
        data = load_pickle(path)
        run_data = data.get(run_seed, {})
        cluster_bits[cluster] = run_data.get("dict_prob", [])
    return cluster_bits


def merge_candidate(customers, context, icvrp_bit, otsp_choice):
    depot = np.mean(customers, axis=0)
    cluster_paths = {}

    for cluster, bitstring in otsp_choice.items():
        cluster_info = context[cluster]
        local_path = otsp_local_path(bitstring, cluster_info["entry_local"])
        customer_path = [cluster_info["local_to_orig"][local] for local in local_path]
        cluster_paths[cluster] = {
            "otsp_bitstring": bitstring,
            "local_path": local_path,
            "customer_path": customer_path,
            "entry_orig": cluster_info["entry_orig"],
            "exit_orig": cluster_info["exit_orig"],
        }

    merged_routes = []
    for cluster_route in icvrp_routes(icvrp_bit):
        customer_route = []
        for node in cluster_route:
            if node == "D":
                continue
            cluster = int(node[1:])
            customer_route.extend(cluster_paths[cluster]["customer_path"])

        route_points = [depot] + [customers[idx] for idx in customer_route] + [depot]
        merged_routes.append(
            {
                "cluster_route": cluster_route,
                "customer_route": customer_route,
                "distance": route_distance(route_points),
            }
        )

    all_customers = [
        customer for route in merged_routes for customer in route["customer_route"]
    ]
    expected_customers = set(range(len(customers)))
    if len(all_customers) != len(expected_customers):
        raise ValueError("Merged route does not contain the expected customer count.")
    if set(all_customers) != expected_customers:
        raise ValueError("Merged route does not cover every customer exactly once.")

    return {
        "icvrp_bitstring": icvrp_bit,
        "cluster_paths": cluster_paths,
        "routes": merged_routes,
        "total_distance": sum(route["distance"] for route in merged_routes),
    }


def merge_run(customers, context, icvrp_bits, otsp_bits):
    for cluster, bits in otsp_bits.items():
        if not bits:
            raise ValueError(f"Missing OTSP solution for cluster {cluster}.")

    best_merge = None
    best_distance = float("inf")
    checked_combinations = 0
    invalid_combinations = 0

    otsp_clusters = sorted(otsp_bits)
    otsp_choices = list(
        itertools.product(*(otsp_bits[cluster] for cluster in otsp_clusters))
    )

    for icvrp_bit in icvrp_bits:
        for otsp_choice_tuple in otsp_choices:
            checked_combinations += 1
            otsp_choice = dict(zip(otsp_clusters, otsp_choice_tuple))

            try:
                candidate = merge_candidate(customers, context, icvrp_bit, otsp_choice)
            except ValueError:
                invalid_combinations += 1
                continue

            if candidate["total_distance"] < best_distance:
                best_distance = candidate["total_distance"]
                best_merge = candidate

    if best_merge is None:
        raise ValueError("No valid merged route found from candidate combinations.")

    best_merge["checked_combinations"] = checked_combinations
    best_merge["invalid_combinations"] = invalid_combinations
    return best_merge


def merge_dataset(args):
    root = Path(args.root).resolve()
    customers = load_pickle(root / "datasets" / f"dataset{args.data_id}.pkl")

    best_solution = None
    best_distance = float("inf")
    skipped = {}
    merged_run_count = 0
    checked_combinations = 0
    invalid_combinations = 0

    for run_seed in range(args.run_start, args.run_end + 1):
        assignments, centroids = constrained_kmeans(customers, random_state=run_seed)
        depot = np.mean(customers, axis=0)
        context = cluster_context(customers, assignments, centroids, depot)

        try:
            icvrp_bits = load_icvrp_bits(
                root, args.variant, args.p, args.prob, run_seed, args.data_id
            )
            otsp_bits = load_otsp_bits(
                root, args.variant, args.p, args.prob, run_seed, args.data_id
            )

            if not icvrp_bits:
                raise ValueError("Missing ICVRP solution.")

            run_solution = merge_run(customers, context, icvrp_bits, otsp_bits)
            run_solution["run_seed"] = run_seed
            merged_run_count += 1
            checked_combinations += run_solution["checked_combinations"]
            invalid_combinations += run_solution["invalid_combinations"]

            if run_solution["total_distance"] < best_distance:
                best_distance = run_solution["total_distance"]
                best_solution = run_solution

        except (FileNotFoundError, KeyError, ValueError) as exc:
            if args.strict:
                raise
            skipped[run_seed] = str(exc)

    result = {
        "data_id": args.data_id,
        "variant": args.variant,
        "p": args.p,
        "prob": prob_tag(args.prob),
        "run_start": args.run_start,
        "run_end": args.run_end,
        "merged_run_count": merged_run_count,
        "skipped_run_count": len(skipped),
        "checked_combinations": checked_combinations,
        "invalid_combinations": invalid_combinations,
        "best_solution": best_solution,
        "skipped_runs": skipped,
    }

    out_path = (
        root
        / args.output
        / f"results_dataset_{args.variant}_qaoa"
        / f"p_{args.p}"
        / prob_tag(args.prob)
        / f"dataset_{args.data_id}.pkl"
    )
    save_pickle(out_path, result)
    return out_path, merged_run_count, len(skipped), best_solution


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge post-processed ICVRP and OTSP solutions for one dataset."
    )
    parser.add_argument("--data_id", type=int, required=True)
    parser.add_argument("--variant", choices=["std", "ma"], default="ma")
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--prob", default="1")
    parser.add_argument("--run_start", type=int, default=0)
    parser.add_argument("--run_end", type=int, default=49)
    parser.add_argument("--root", default=os.getcwd())
    parser.add_argument("--output", default="results_merged")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    output_path, n_merged, n_skipped, best = merge_dataset(parse_args())
    print(f"Saved best merged summary from {n_merged} merged runs to {output_path}")
    if best is not None:
        print(f"Best run={best['run_seed']} total_distance={best['total_distance']}")
    if n_skipped:
        print(f"Skipped {n_skipped} runs with missing or invalid inputs")
