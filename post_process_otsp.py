import numpy as np
import math
from itertools import combinations
import pickle
import os
from  OTSP import build_qubo_ostp
from scripts.Preprocess import find_entry_exit, constrained_kmeans


def calculate_distance(bitstring, var_list, w):
    """Actual route distance (sum of selected edge lengths) from a bitstring."""
    dist = 0.0
    for k, b in enumerate(bitstring):
        if b == '1':
            _, i, j = var_list[k].split('_')
            i, j = int(i), int(j)
            dist += w[i, j]
    return dist

def calculate_energy_from_bitstring(bitstring, Q, g, c):
    """QUBO energy for a given bitstring."""
    x = np.array([int(b) for b in bitstring])
    return x.T @ Q @ x + g @ x + c


def check_otsp_feasibility(bitstring, var_list, n, start_idx, end_idx, check_connected=True):
    """
    Check feasibility of an OSTP assignment.

    Parameters
    ----------
    var_list : list[str]
        Variable names like ['x_0_1', 'x_0_2', ...]
    bitstring : str | list | np.ndarray | dict
        Assignment for variables.
        Supported forms:
          - bitstring string like '010101'
          - list/array of 0/1 with same order as var_list
          - dict mapping variable name -> 0/1
    n : int
        Number of nodes.
    start_idx : int
        Start node.
    end_idx : int
        End node.
    check_connected : bool
        If True, also check that the chosen edges form one single
        start-to-end Hamiltonian path, not disconnected subtours.

    Returns
    -------
    feasible : bool
    info : dict
        Diagnostic information.
    """

    # -------------------------
    # Parse assignment
    # -------------------------
    if isinstance(bitstring, dict):
        x = {v: int(bitstring.get(v, 0)) for v in var_list}
    elif isinstance(bitstring, str):
        if len(bitstring) != len(var_list):
            raise ValueError("Bitstring length does not match var_list length.")
        x = {v: int(b) for v, b in zip(var_list, bitstring)}
    else:
        if len(bitstring) != len(var_list):
            raise ValueError("Assignment length does not match var_list length.")
        x = {v: int(b) for v, b in zip(var_list, bitstring)}

    # -------------------------
    # Compute in/out degrees
    # -------------------------
    out_deg = [0] * n
    in_deg = [0] * n
    selected_edges = []

    for v, val in x.items():
        if val not in (0, 1):
            raise ValueError(f"Variable {v} has non-binary value {val}.")
        if val == 1:
            _, i, j = v.split('_')
            i, j = int(i), int(j)
            out_deg[i] += 1
            in_deg[j] += 1
            selected_edges.append((i, j))

    # -------------------------
    # Check degree constraints
    # -------------------------
    violations = []

    for i in range(n):
        expected_out = 0 if i == end_idx else 1
        expected_in = 0 if i == start_idx else 1

        if out_deg[i] != expected_out:
            violations.append(
                f"Node {i}: out-degree {out_deg[i]} != expected {expected_out}"
            )
        if in_deg[i] != expected_in:
            violations.append(
                f"Node {i}: in-degree {in_deg[i]} != expected {expected_in}"
            )

    degree_feasible = (len(violations) == 0)

    # -------------------------
    # Optional connectivity/path check
    # -------------------------
    connected_feasible = True
    path = None

    if check_connected and degree_feasible:
        succ = {}
        for i, j in selected_edges:
            if i in succ:
                connected_feasible = False
                violations.append(f"Node {i} has multiple successors.")
            succ[i] = j

        visited = []
        current = start_idx
        seen = set()

        while current in succ:
            if current in seen:
                connected_feasible = False
                violations.append("Cycle detected while following path from start.")
                break
            seen.add(current)
            nxt = succ[current]
            visited.append((current, nxt))
            current = nxt

        path_nodes = [start_idx]
        for i, j in visited:
            path_nodes.append(j)

        path = path_nodes

        # Must end at end_idx
        if current != end_idx:
            connected_feasible = False
            violations.append(
                f"Path starting at {start_idx} ends at {current}, not at {end_idx}."
            )

        # Must visit all nodes exactly once
        if len(path_nodes) != n:
            connected_feasible = False
            violations.append(
                f"Path contains {len(path_nodes)} nodes, expected {n}."
            )

        if len(set(path_nodes)) != len(path_nodes):
            connected_feasible = False
            violations.append("Path repeats nodes.")

        if set(path_nodes) != set(range(n)):
            connected_feasible = False
            missing = sorted(set(range(n)) - set(path_nodes))
            extra = sorted(set(path_nodes) - set(range(n)))
            if missing:
                violations.append(f"Missing nodes from main path: {missing}")
            if extra:
                violations.append(f"Unexpected nodes in path: {extra}")

    feasible = degree_feasible and (connected_feasible if check_connected else True)

    info = {
        "feasible": feasible,
        "degree_feasible": degree_feasible,
        "connected_feasible": connected_feasible,
        "in_deg": in_deg,
        "out_deg": out_deg,
        "selected_edges": selected_edges,
        "path": path,
        "violations": violations,
    }

    return feasible, info


def find_nearest_feasible(w, bitstring, var_list, n, start_idx, end_idx, label_to_index, Q, g, c):
    """
    Flip bits until a feasible solution is found, within one or two bit-flips.
    Returns (feasible_bitstring (if found), distance, energy, flip counts).
    """
    bitstring_array = np.array([int(b) for b in bitstring])
    best_feasible = None
    best_dist = float('inf')
    best_energy = float('inf')

    flip_count = 0
    for num_flips in range(1, 3):
        for flip_positions in combinations(range(len(var_list)), num_flips):
            flip_count += 1
            candidate = bitstring_array.copy()
            for pos in flip_positions:
                candidate[pos] = 1 - candidate[pos]
            cand_str = ''.join(map(str, candidate))
            if check_otsp_feasibility(cand_str, var_list, n, start_idx, end_idx,)[0]:
                d = calculate_distance(cand_str, var_list, w)
                if d < best_dist:
                    best_dist = d
                    best_feasible = cand_str
                    best_energy = calculate_energy_from_bitstring(cand_str, Q, g, c)
                break

            if best_feasible is not None:
                break
    return best_feasible, best_dist, best_energy, flip_count


def one_to_three_local_search(n, w, start_idx, end_idx,counts, prob=0.001, num_states=100):
    """
    Returns a list of all possible feasible solutions within one and two bit-flips.

    """
    feas_solns_3 = []
    top100 = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:num_states]
    top_100_counts = dict(top100)
    var_list, label_to_index, Q, g, c, _, _, _, _ = build_qubo_ostp(n, w, start_idx, end_idx)

    for bitstring in top_100_counts.keys():
        # print(bitstring)
        if top_100_counts[bitstring] > prob:
            if check_otsp_feasibility(bitstring, var_list, n, start_idx, end_idx,)[0]:
                found_dist = calculate_distance(bitstring, var_list, w)
                found_energy = calculate_energy_from_bitstring(bitstring, Q, g, c)
                feas_solns_3.append((bitstring, found_dist, found_energy, 0))
            else:
                std_post_bit, std_post_dist, std_post_energy, flip_counts = find_nearest_feasible(w, bitstring, var_list, n, start_idx, end_idx,label_to_index, Q, g, c)
                if flip_counts < 79 and std_post_bit is not None:
                    feas_solns_3.append((std_post_bit, std_post_dist, std_post_energy, flip_counts))

                # else:
                #     feas_solns_3.append((None, None, None, flip_counts))

    return feas_solns_3


def probable_bitstrings(feas_soln_data):

    '''
    Returns a list of all feasible solutions found within one and two bit-flips 
    that have the lowest distance, along with the number of bit-flips.

    '''

    bitstring = [i[0] for i in feas_soln_data]
    # energy =[ i[1] for i in feas_soln_data]
    distance = [i[2] for i in feas_soln_data]
    flips = [i[3] for i in feas_soln_data]

    lowest_distance = min(distance)
    probable_bitstring = []
    for i in range(len(distance)):
        if distance[i] == lowest_distance:
            probable_bitstring.append(bitstring[i])
            flips.append(flips[i])
    unique_lst = list(dict.fromkeys(probable_bitstring))
    return unique_lst, flips


if __name__ == "__main__":
    path = os.getcwd()
    prob_2 = 10
    prob_1 = prob_2/1000
    result_folder = os.path.join(path, 'results_raw_otsp')
    folder_path = os.path.join(path, 'datasets')
    result_post_folder = os.path.join(path, 'results_post_otsp')
    for dat_id in range(1, 100):
        fn = 'dataset' + str(dat_id)
        file_path = os.path.join(folder_path, f"{fn}.pkl")
        with open(file_path, "rb") as f:
            customers = pickle.load(f)

        cluster_std_runs = {cluster: {} for cluster in range(3)}
        cluster_ma_runs = {cluster: {} for cluster in range(3)}

        for run_seed in range(50):
            assignments, centroids = constrained_kmeans(customers, random_state=run_seed)
            depot = np.mean(customers, axis=0)

            run_folder = os.path.join(result_folder, f"results_dataset_{dat_id}", "p_3")
            file_path = os.path.join(run_folder, f"run_{run_seed}.pkl")

            with open(file_path, "rb") as f:
                results = pickle.load(f)

            for cluster in range(3):
                keys = 'cluster' + str(cluster + 1)
                results_key = results[keys][0]
                std_counts = results_key['std_qaoa'][0]
                ma_counts = results_key['ma_qaoa'][0]

                point_indices = np.where(assignments == cluster)[0]
                point_indices_sorted = np.sort(point_indices)
                cluster_points_sorted = customers[point_indices_sorted]

                dist_mat = np.zeros((4, 4))
                for i in range(4):
                    for j in range(4):
                        if i != j:
                            dist_mat[i][j] = math.hypot(
                                cluster_points_sorted[i][0] - cluster_points_sorted[j][0],
                                cluster_points_sorted[i][1] - cluster_points_sorted[j][1]
                            )

                orig_to_local = {p_idx: j for j, p_idx in enumerate(point_indices_sorted)}
                entry_orig, exit_orig = find_entry_exit(
                    point_indices, customers, cluster, centroids, depot
                )
                entry_local = orig_to_local[entry_orig]
                exit_local = orig_to_local[exit_orig]

                feas_soln_local_std = one_to_three_local_search(
                    4, dist_mat, entry_local, exit_local, std_counts, prob=prob_1
                )
                feas_soln_local_ma = one_to_three_local_search(
                    4, dist_mat, entry_local, exit_local, ma_counts, prob=prob_1
                )

                if feas_soln_local_std:
                    probable_std, flips_std = probable_bitstrings(feas_soln_local_std)
                else:
                    probable_std, flips_std = [], []

                if feas_soln_local_ma:
                    probable_ma, flips_ma = probable_bitstrings(feas_soln_local_ma)
                else:
                    probable_ma, flips_ma = [], []

                cluster_std_runs[cluster][run_seed] = {
                    'dict_prob': probable_std,
                    'dict_flip': flips_std
                }
                cluster_ma_runs[cluster][run_seed] = {
                    'dict_prob': probable_ma,
                    'dict_flip': flips_ma
                }

        for cluster in range(3):
            tag = f"prob_00{prob_2}_cluster{cluster+1}"

            std_folder = os.path.join(
                result_post_folder, "results_dataset_std_qaoa", "p_3", tag
            )
            os.makedirs(std_folder, exist_ok=True)
            std_out_path = os.path.join(std_folder, f"dataset_{dat_id}.pkl")
            with open(std_out_path, "wb") as f:
                pickle.dump(cluster_std_runs[cluster], f, protocol=pickle.HIGHEST_PROTOCOL)
                print(std_out_path)

            ma_folder = os.path.join(
                result_post_folder, "results_dataset_ma_qaoa", "p_3", tag
            )
            os.makedirs(ma_folder, exist_ok=True)
            ma_out_path = os.path.join(ma_folder, f"dataset_{dat_id}.pkl")
            with open(ma_out_path, "wb") as f:
                pickle.dump(cluster_ma_runs[cluster], f, protocol=pickle.HIGHEST_PROTOCOL)
                print(ma_out_path)
