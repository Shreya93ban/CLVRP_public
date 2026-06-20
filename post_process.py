import numpy as np
from itertools import combinations
import pickle
import os
from ICVRP import build_qubo
from scripts.Preprocess import constrained_kmeans


def calculate_distance(bitstring, var_list, w, label_to_index):
    """Actual route distance (sum of edge lengths) from a bitstring."""
    dist = 0.0
    for i, v in enumerate(bitstring):
        if v == '1':
            from_node = var_list[i].split('_')[1]
            to_node = var_list[i].split('_')[2]
            dist += w[label_to_index[from_node], label_to_index[to_node]]
    return dist


def calculate_energy_from_bitstring(bitstring, Q, g, c):
    """QUBO energy for a given bitstring."""
    x = np.array([int(b) for b in bitstring])
    return x.T @ Q @ x + g @ x + c


def check_feasibility(bitstring, var_list):
    """Check if a 12‑bit string satisfies all VRP constraints."""
    sol_array = np.array([int(b) for b in bitstring])
    constraints = [
        sol_array[var_list.index('x_D_C1')] + sol_array[var_list.index('x_C2_C1')] + sol_array[var_list.index('x_C3_C1')] == 1,
        sol_array[var_list.index('x_D_C2')] + sol_array[var_list.index('x_C1_C2')] + sol_array[var_list.index('x_C3_C2')] == 1,
        sol_array[var_list.index('x_D_C3')] + sol_array[var_list.index('x_C1_C3')] + sol_array[var_list.index('x_C2_C3')] == 1,
        sol_array[var_list.index('x_C1_D')] + sol_array[var_list.index('x_C1_C2')] + sol_array[var_list.index('x_C1_C3')] == 1,
        sol_array[var_list.index('x_C2_D')] + sol_array[var_list.index('x_C2_C1')] + sol_array[var_list.index('x_C2_C3')] == 1,
        sol_array[var_list.index('x_C3_D')] + sol_array[var_list.index('x_C3_C1')] + sol_array[var_list.index('x_C3_C2')] == 1,
        sol_array[var_list.index('x_D_C1')] + sol_array[var_list.index('x_D_C2')] + sol_array[var_list.index('x_D_C3')] == 2,
        sol_array[var_list.index('x_C1_D')] + sol_array[var_list.index('x_C2_D')] + sol_array[var_list.index('x_C3_D')] == 2
    ]
    return all(constraints)


def find_nearest_feasible(bitstring, var_list, w, label_to_index, Q, g, c):
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
            if check_feasibility(cand_str, var_list):
                d = calculate_distance(cand_str, var_list, w, label_to_index)
                if d < best_dist:
                    best_dist = d
                    best_feasible = cand_str
                    best_energy = calculate_energy_from_bitstring(cand_str, Q, g, c)
                break

            if best_feasible is not None:
                break
    return best_feasible, best_dist, best_energy, flip_count


def one_to_three_local_search(w, counts, prob=0.001, num_states=100):
    """
    Returns a list of all possible feasible solutions within one and two bit-flips.

    """
    feas_solns_3 = []
    top100 = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:num_states]
    top_100_counts = dict(top100)
    var_list, label_to_index, Q, g, c, _, _, _, _ = build_qubo(w)

    for bitstring in top_100_counts.keys():
        # print(bitstring)
        if top_100_counts[bitstring] > prob:
            if check_feasibility(bitstring, var_list):
                found_dist = calculate_distance(bitstring, var_list, w, label_to_index)
                found_energy = calculate_energy_from_bitstring(bitstring, Q, g, c)
                feas_solns_3.append((bitstring, found_dist, found_energy, 0))
            else:
                std_post_bit, std_post_dist, std_post_energy, flip_counts = find_nearest_feasible(bitstring, var_list, w, label_to_index, Q, g, c)
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
    prob_2 = 9
    prob_1 = prob_2/1000
    result_folder = os.path.join(path, 'results_raw')
    folder_path = os.path.join(path, 'datasets')
    result_post_folder = os.path.join(path, 'results_post')
    for run_seed in range(0, 50): 

        feas_soln_local_std = []
        feas_soln_local_ma = []

        for dat_id in range(100):

            # raw_result_loading
            run_folder = os.path.join(result_folder, f"results_dataset_{dat_id}", "p_3")
            file_path = os.path.join(run_folder, f"run_{run_seed}.pkl")
            with open(file_path, "rb") as f:
                results = pickle.load(f)

            std_counts = results['std_counts'][0]
            ma_counts = results['ma_counts'][0]

            # data_loading
            fn = 'dataset' + str(dat_id)
            file_path = os.path.join(folder_path, f"{fn}.pkl")
            with open(file_path, "rb") as f:
                customers = pickle.load(f)

            # Qubo formation
            assignments, centroids = constrained_kmeans(customers, random_state=run_seed)
            depot = np.mean(customers, axis=0)

            points_icvrp = [depot] + list(centroids)
            w = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    w[i, j] = np.sqrt(np.sum((points_icvrp[i] - points_icvrp[j])**2))

            # Local 1-3 bit-flip search

            # Std QAOA
            feas_soln_local = one_to_three_local_search(w, std_counts, prob=prob_1)
            feas_soln_local_std.append(feas_soln_local)

            # MA QAOA
            feas_soln_local_ma1 = one_to_three_local_search(w, ma_counts, prob=prob_1)
            feas_soln_local_ma.append(feas_soln_local_ma1)

            # Probable bitstrings STD QAOA       
            found_feasible_soln = [i for i in range(len(feas_soln_local_std)) if feas_soln_local_std[i] != []]
            all_probable = []
            flips = []
            for k in found_feasible_soln:
                feas_soln_data = feas_soln_local_std[k]
                all_probable.append(probable_bitstrings(feas_soln_data)[0])
                flips.append(probable_bitstrings(feas_soln_data)[1])

            dict_prob = dict(zip(found_feasible_soln, all_probable))
            dict_flip = dict(zip(found_feasible_soln, flips))

            result_std = {
                'dict_prob': dict_prob,
                'dict_flip': dict_flip
            }           
            # print('Std QAOA is done, with flips = ', flips)
            # Probable bitstrings MA-QAOA       
            found_feasible_soln_ma = [i for i in range(len(feas_soln_local_ma)) if feas_soln_local_ma[i] != []]
            all_probable_ma = []
            flips_ma = []
            for k in found_feasible_soln_ma:
                feas_soln_data_ma = feas_soln_local_ma[k]
                all_probable_ma.append(probable_bitstrings(feas_soln_data_ma)[0])
                flips_ma.append(probable_bitstrings(feas_soln_data_ma)[1])

            dict_prob_ma = dict(zip(found_feasible_soln_ma, all_probable_ma))
            dict_flip_ma = dict(zip(found_feasible_soln_ma, flips_ma))

            result_ma = {
                'dict_prob': dict_prob_ma,
                'dict_flip': dict_flip_ma
            }
            # print('MA QAOA is done, with flips = ', flips_ma)
        # Saving data std
        tag=f"prob_00{prob_2}"
        run_folder = os.path.join(result_post_folder, f"results_dataset_std_qaoa", "p_3", tag)

        os.makedirs(run_folder, exist_ok=True)

        out_path = os.path.join(run_folder, f"run_{run_seed}.pkl")

        with open(out_path, "wb") as f:
            pickle.dump(result_std, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(out_path)

        # Saving data ma
        run_folder = os.path.join(result_post_folder, f"results_dataset_ma_qaoa", "p_3", tag)
        os.makedirs(run_folder, exist_ok=True)

        out_path = os.path.join(run_folder, f"run_{run_seed}.pkl")

        with open(out_path, "wb") as f:
            pickle.dump(result_ma, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(out_path)
