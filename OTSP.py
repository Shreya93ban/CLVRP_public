import os
import numpy as np
import math
import pickle
import argparse


# Qiskit imports
from qiskit.primitives import Sampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms.optimizers import COBYLA


from scripts.Preprocess import constrained_kmeans, find_entry_exit
from ICVRP import build_ma_qaoa_circuit, build_standard_qaoa_circuit, energy_expectation






def build_qubo_ostp(n, dist_matrix, start_idx, end_idx, penalty=1000):
    """
    Build QUBO for an open TSP / cluster OTSP.
    
    Parameters
    ----------
    n : int
        Number of nodes.
    dist_matrix : array-like of shape (n, n)
        Distance matrix.
    start_idx : int
        Start node index.
    end_idx : int
        End node index.
    penalty : float, optional
        Penalty used in QuadraticProgramToQubo and for 2-cycle suppression.

    Returns
    -------
    (
        var_list,
        label_to_index,
        qubo_matrix,
        qubo_linear,
        qubo_constant,
        ising_op,
        ising_offset,
        zz_terms,
        z_terms
    )
    """
    label_to_index = {f"N{i}": i for i in range(n)}
    var_list = [f'x_{i}_{j}' for i in range(n) for j in range(n) if i != j]

    qp = QuadraticProgram("Cluster_OTSP")
    for v in var_list:
        qp.binary_var(v)

    # linear objective
    linear_dict = {}
    for v in var_list:
        _, i, j = v.split('_')
        i, j = int(i), int(j)
        linear_dict[v] = dist_matrix[i][j]

    # quadratic penalty for 2-cycles
    quadratic_dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            v_ij = f'x_{i}_{j}'
            v_ji = f'x_{j}_{i}'
            quadratic_dict[(v_ij, v_ji)] = penalty

    qp.minimize(linear=linear_dict, quadratic=quadratic_dict)

    # constraints
    for i in range(n):
        # outgoing degree
        coeffs_out = {f'x_{i}_{j}': 1 for j in range(n) if j != i}
        rhs_out = 0 if i == end_idx else 1
        qp.linear_constraint(coeffs_out, '=', rhs_out, f'out_{i}')

        # incoming degree
        coeffs_in = {f'x_{j}_{i}': 1 for j in range(n) if j != i}
        rhs_in = 0 if i == start_idx else 1
        qp.linear_constraint(coeffs_in, '=', rhs_in, f'in_{i}')

    converter = QuadraticProgramToQubo(penalty=penalty)
    qubo = converter.convert(qp)
    ising_op, ising_offset = qubo.to_ising()

    # extract QUBO matrix, linear vector and constant
    qubo_matrix_upper = qubo.objective.quadratic.to_array()
    qubo_matrix = qubo_matrix_upper + qubo_matrix_upper.T - np.diag(np.diag(qubo_matrix_upper))
    qubo_linear = qubo.objective.linear.to_array()
    qubo_constant = qubo.objective.constant

    # extract ZZ and Z terms for circuit building
    zz_terms, z_terms = [], []
    if hasattr(ising_op, 'paulis'):
        for pauli, coeff in zip(ising_op.paulis, ising_op.coeffs):
            coeff_real = float(np.real(coeff))
            label = pauli.to_label()
            idx = [len(label) - 1 - k for k, ch in enumerate(label) if ch == 'Z']
            if len(idx) == 2:
                zz_terms.append((coeff_real, tuple(sorted(idx))))
            elif len(idx) == 1:
                z_terms.append((coeff_real, idx[0]))

    return (
        var_list,
        label_to_index,
        qubo_matrix,
        qubo_linear,
        qubo_constant,
        ising_op,
        ising_offset,
        zz_terms,
        z_terms
    )

def run_standard_qaoa(n, w, entry, exit,  p=3, seed=None):
    """Run standard QAOA and return (bitstring, distance, energy)."""
    var_list, label_to_index, Q, g, c, ising_op, offset, zz_terms, z_terms = build_qubo_ostp(n, w, entry, exit)
    n_vars = len(var_list)
    circuit, params = build_standard_qaoa_circuit(n_vars, zz_terms, z_terms, p)

    if seed is not None:
        np.random.seed(seed)
    x0 = np.random.uniform(-0.5, 0.5, len(params))
    optimizer = COBYLA(maxiter=200)
    res = optimizer.minimize(
        lambda v: energy_expectation(circuit, params, v, ising_op, offset),
        x0
    )
    opt_params = res.x
    # sample
    circuit_measured = circuit.copy()
    circuit_measured.measure_all()
    sampler = Sampler()
    bound = circuit_measured.assign_parameters(dict(zip(params, opt_params)))
    result = sampler.run(bound, shots=10000).result()
    counts = result.quasi_dists[0].binary_probabilities()
    return counts

def run_ma_qaoa(n, w, entry, exit, p=3, seed=None):
    """Run MA‑QAOA and return (bitstring, distance, energy)."""
    var_list, label_to_index, Q, g, c, ising_op, offset, zz_terms, z_terms = build_qubo_ostp(n, w, entry, exit)
    n_vars = len(var_list)
    circuit, params = build_ma_qaoa_circuit(n_vars, zz_terms, z_terms, p)

    if seed is not None:
        np.random.seed(seed)
    x0 = np.random.uniform(-0.1, 0.1, len(params))
    optimizer = COBYLA(maxiter=200)
    res = optimizer.minimize(
        lambda v: energy_expectation(circuit, params, v, ising_op, offset),
        x0
    )
    opt_params = res.x
    circuit_measured = circuit.copy()
    circuit_measured.measure_all()
    sampler = Sampler()
    bound = circuit_measured.assign_parameters(dict(zip(params, opt_params)))
    result = sampler.run(bound, shots=10000).result()
    counts = result.quasi_dists[0].binary_probabilities()
    return counts



# code starts

if __name__ == "__main__":
    
    dat_id = 0
    run_seed = 0

    file_path = "C:/Users/shrey/OneDrive/Desktop/SOA_WORKS/VRP/CLVRP/CLVRP_public/dataset0.pkl"

    with open(file_path, "rb") as f:
        customers = pickle.load(f)


    # ----- STEP 1 & 2: Clustering + entry/exit points -----
    assignments, centroids = constrained_kmeans(customers, random_state=run_seed)

    # Depot
    depot = np.mean(customers, axis=0)

    results = {'cluster1': [],
              'cluster2': [],
              'cluster3': [],
              }

    for cluster_id in range(3):
        ind_clus_results = {'std_qaoa': [], 'ma_qaoa': []}
        point_indices = np.where(assignments == cluster_id)[0]
        point_indices_sorted = np.sort(point_indices)
        start_label = cluster_id * 4 + 1
        orig_to_local = {p_idx: j for j, p_idx in enumerate(point_indices_sorted)}

        # Extract points in sorted order
        cluster_points_sorted = customers[point_indices_sorted]

        # Build distance matrix for this cluster
        dist_mat = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                if i != j:
                    dist_mat[i][j] = math.hypot(
                        cluster_points_sorted[i][0] - cluster_points_sorted[j][0],
                        cluster_points_sorted[i][1] - cluster_points_sorted[j][1]
                    )

        # Entry / exit
        entry_orig, exit_orig = find_entry_exit(point_indices, customers,
                                                cluster_id, centroids, depot)
        entry_local = orig_to_local[entry_orig]
        exit_local = orig_to_local[exit_orig]
        
        # Standard QAOA
        std_counts = run_standard_qaoa(4, dist_mat, entry_local, exit_local, p=1, seed=run_seed)

        # MA‑QAOA
        ma_counts = run_ma_qaoa(4, dist_mat, entry_local, exit_local, p=1, seed=run_seed)

        ind_clus_results['std_qaoa'].append(std_counts)

        ind_clus_results['ma_qaoa'].append(ma_counts)

        if cluster_id == 0:
            results['cluster1'].append(ind_clus_results)
        elif cluster_id == 1:
            results['cluster2'].append(ind_clus_results)
        else:
            results['cluster3'].append(ind_clus_results)

    print(results.keys()) # We print the results for demonstation purposes. 

