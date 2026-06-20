import os
import numpy as np
import pickle
from scripts.Preprocess import constrained_kmeans
from ICVRP import build_qubo
from post_process import calculate_distance, calculate_energy_from_bitstring, check_feasibility


def post_data_load(result_folder, run_seed, pr, variant, p=3):
    if pr < 10:
        run_folder = os.path.join(
            result_folder, "results_dataset_"+variant+"_qaoa",
            "p_"+str(p),
            'prob_00'+str(pr)
            )
    else:
        run_folder = os.path.join(
            result_folder, "results_dataset_"+variant+"_qaoa",
            "p_"+str(p),
            'prob_0'+str(int(pr/10))
            )
    file_path = os.path.join(run_folder, f"run_{run_seed}.pkl")
    with open(file_path, "rb") as f:
        results = pickle.load(f)
        dict_prob = results['dict_prob']
        dict_flip = results['dict_flip']

    return dict_prob, dict_flip



def compute_metrics(data_id, pr):
    path= os.getcwd()
    folder_path = os.path.join(path, 'results_post', 'p_'+str(pr))
    file_path= fn = 'metrics_icvrp_dataid_' + str(data_id)
    file_path = os.path.join(folder_path, f"{fn}.pkl")
    with open(file_path, "rb") as f:
        res1 = pickle.load(f)
    distance_data = res1['distance_dataset']
    distances_dataset=[]
    for elem in distance_data.values():
        bits= [j for j in elem.keys()]
        id = np.argmin([j for j in elem.values()])
        distances_dataset.append((bits[id], [j for j in elem.values()][id]))



    num_feas= len(distances_dataset)
    mean_dist= np.mean([i[1] for i in distances_dataset])
    sem_dist= np.std([i[1] for i in distances_dataset])/np.sqrt(len(distances_dataset))
    min_dist_id= np.argmin([i[1] for i in distances_dataset])
    min_dist = [i[1] for i in distances_dataset][min_dist_id]
    min_bit= [i[0] for i in distances_dataset][min_dist_id]

    return num_feas, mean_dist, sem_dist, min_dist_id, min_bit, min_dist


import itertools
import numpy as np
from post_process import calculate_distance

def find_min_distance_feasible_bitstrings(
    var_list,
    w,
    label_to_index,
    feasibility_fn,
    tol=1e-2
):
    """
    Iterate over all bitstrings, exclude the all-zero string,
    keep only feasible ones, and return all feasible bitstrings
    with minimum distance.

    Parameters
    ----------
    var_list : list[str]
    w : np.ndarray
    label_to_index : dict
    feasibility_fn : callable
        Function taking a bitstring and returning True/False
    tol : float
        Tolerance for distance ties

    Returns
    -------
    best_bitstrings : list[str]
    best_distance : float
    """
    n = len(var_list)
    best_bitstrings = []
    best_distance = float("inf")

    for bits in itertools.product("01", repeat=n):
        bitstring = "".join(bits)

        if "1" not in bitstring:
            continue

        if not feasibility_fn(bitstring, var_list):
            continue

        distance = calculate_distance(bitstring, var_list, w, label_to_index)

        if distance < best_distance - tol:
            best_distance = distance
            best_bitstrings = [bitstring]
        elif abs(distance - best_distance) <= tol:
            best_bitstrings.append(bitstring)

    return best_bitstrings, best_distance

# num_feas, mean_dist, sem_dist, min_dist_id, min_bit, min_dist= compute_metrics(data_id, pr)


if __name__ == "__main__":

    path = os.getcwd()
    folder_path = os.path.join(path, 'datasets')
    result_folder = os.path.join(path, 'results_post')
    out_folder = os.path.join(result_folder, 'approx_ratio')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for pr in range(1, 10):
        for data_id in range(100):
            runs_distance = []
            runs_energy = []
            success_runs = []
            fn = 'dataset' + str(data_id)
            file_path = os.path.join(folder_path, f"{fn}.pkl")
            with open(file_path, "rb") as f:
                customers = pickle.load(f)

            assignments, centroids = constrained_kmeans(customers)
            depot = np.mean(customers, axis=0)
            points_icvrp = [depot] + list(centroids)

            
            w = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    w[i, j] = np.sqrt(np.sum((points_icvrp[i] - points_icvrp[j])**2))

            var_list, label_to_index, Q, g, c, _, _, _, _ = build_qubo(w)

            for run_seed in range(50):
                bits_std, flips_std = post_data_load(
                    result_folder, run_seed, pr,
                    'ma')

                

                if data_id in [i for i in bits_std.keys()]:
                
                    bit_list = bits_std[data_id]
                    # flip_list= flips_std[i]
                    dist_list = []
                    energy_list = []
                    for bitstring in bit_list:
                        distance = calculate_distance(bitstring, var_list, w,label_to_index)
                        energy = calculate_energy_from_bitstring(
                                                                bitstring, Q, g,
                                                                c)
                        dist_list.append(distance)
                        energy_list.append(energy)
                    dict_distance = dict(zip(bit_list, dist_list))
                    dict_energy = dict(zip(bit_list, energy_list))
                
                    runs_distance.append(dict_distance)
                    runs_energy.append(dict_energy)
                    
                success_runs.append(run_seed)
            
            distance_dataset = dict(zip(success_runs, runs_distance))
            energy_dataset = dict(zip(success_runs, runs_energy))
            brute_bits, brute_dist = find_min_distance_feasible_bitstrings(
                                                                        var_list,
                                                                        w,
                                                                        label_to_index,
                                                                        check_feasibility,
                                                                        tol=1e-2)

            out_res = {}
            out_res["distance_dataset"] = distance_dataset
            out_res["energy_dataset"] = energy_dataset
            out_res["bruteforce_bits"] = brute_bits
            out_res["bruteforce_dist"] = brute_dist

            fn = 'metrics_icvrp_dataid_' + str(data_id)
            folder_new = os.path.join(result_folder, 'ma', 'p_'+str(pr))
            os.makedirs(folder_new, exist_ok=True)
            file_path = os.path.join(folder_new, f"{fn}.pkl")
            
            with open(file_path, "wb") as f:
                pickle.dump(out_res, f)
            print(file_path)
            # print(var_list, w, label_to_index)
