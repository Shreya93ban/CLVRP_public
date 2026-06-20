import numpy as np
import warnings


# Qiskit imports
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives import Sampler, Estimator
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer

warnings.filterwarnings('ignore')


def build_qubo(w, penalty=500):
    """
    Build QUBO for the 4‑node VRP (depot + 3 centroids).
    w : 4x4 distance matrix (indices: 0=depot,1=C1,2=C2,3=C3)
    """
    label_to_index = {'D': 0, 'C1': 1, 'C2': 2, 'C3': 3}
    var_list = [
        'x_D_C1','x_D_C2','x_D_C3',
        'x_C1_D','x_C1_C2','x_C1_C3',
        'x_C2_D','x_C2_C1','x_C2_C3',
        'x_C3_D','x_C3_C1','x_C3_C2'
    ]

    qp = QuadraticProgram("VRP")
    for v in var_list:
        qp.binary_var(v)

    # linear objective
    linear_dict = {}
    for v in var_list:
        from_node = v.split('_')[1]
        to_node   = v.split('_')[2]
        linear_dict[v] = w[label_to_index[from_node], label_to_index[to_node]]
    qp.minimize(linear=linear_dict)

    # constraints
    qp.linear_constraint({'x_D_C1':1, 'x_C2_C1':1, 'x_C3_C1':1}, '=', 1, "visit_C1")
    qp.linear_constraint({'x_D_C2':1, 'x_C1_C2':1, 'x_C3_C2':1}, '=', 1, "visit_C2")
    qp.linear_constraint({'x_D_C3':1, 'x_C1_C3':1, 'x_C2_C3':1}, '=', 1, "visit_C3")
    qp.linear_constraint({'x_C1_D':1, 'x_C1_C2':1, 'x_C1_C3':1}, '=', 1, "leave_C1")
    qp.linear_constraint({'x_C2_D':1, 'x_C2_C1':1, 'x_C2_C3':1}, '=', 1, "leave_C2")
    qp.linear_constraint({'x_C3_D':1, 'x_C3_C1':1, 'x_C3_C2':1}, '=', 1, "leave_C3")
    qp.linear_constraint({'x_D_C1':1, 'x_D_C2':1, 'x_D_C3':1}, '=', 2, "depot_out")
    qp.linear_constraint({'x_C1_D':1, 'x_C2_D':1, 'x_C3_D':1}, '=', 2, "depot_in")

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
            idx = [len(label)-1-i for i,ch in enumerate(label) if ch=='Z']
            if len(idx) == 2:
                zz_terms.append((coeff_real, tuple(sorted(idx))))
            elif len(idx) == 1:
                z_terms.append((coeff_real, idx[0]))

    return (var_list, label_to_index, qubo_matrix, qubo_linear, qubo_constant,
            ising_op, ising_offset, zz_terms, z_terms)




# ----------------------------------------------------------------------
# QAOA circuit builders (standard and MA)
# ----------------------------------------------------------------------
def build_standard_qaoa_circuit(n_qubits, zz_terms, z_terms, p=1):
    qc = QuantumCircuit(n_qubits)
    params = []
    qc.h(range(n_qubits))
    for layer in range(p):
        gamma = Parameter(f"gamma_{layer}")
        params.append(gamma)
        for coeff, (i, j) in zz_terms:
            qc.rzz(2 * gamma * coeff, i, j)
        for coeff, i in z_terms:
            qc.rz(2 * gamma * coeff, i)
        beta = Parameter(f"beta_{layer}")
        params.append(beta)
        qc.rx(2 * beta, range(n_qubits))
    return qc, params

def build_ma_qaoa_circuit(n_qubits, zz_terms, z_terms, p=1):
    qc = QuantumCircuit(n_qubits)
    params = []
    qc.h(range(n_qubits))
    for layer in range(p):
        for coeff, (i, j) in zz_terms:
            g = Parameter(f"gZZ_{layer}_{i}_{j}")
            params.append(g)
            qc.cx(i, j)
            qc.rz(2 * g * coeff, j)
            qc.cx(i, j)
        for coeff, i in z_terms:
            g = Parameter(f"gZ_{layer}_{i}")
            params.append(g)
            qc.rz(2 * g * coeff, i)
        for i in range(n_qubits):
            b = Parameter(f"b_{layer}_{i}")
            params.append(b)
            qc.rx(2 * b, i)
    return qc, params


def energy_expectation(circuit, parameters, vals, ising_op, offset):
    est = Estimator()
    bound = circuit.assign_parameters(dict(zip(parameters, vals)))
    res = est.run([bound], [ising_op]).result()
    return float(np.real(res.values[0] + offset))



def run_standard_qaoa(w, p=3, seed=None):
    """Run standard QAOA and return (bitstring, distance, energy)."""
    var_list, label_to_index, Q, g, c, ising_op, offset, zz_terms, z_terms = build_qubo(w)
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

def run_ma_qaoa(w, p=3, seed=None):
    """Run MA‑QAOA and return (bitstring, distance, energy)."""
    var_list, label_to_index, Q, g, c, ising_op, offset, zz_terms, z_terms = build_qubo(w)
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


if __name__ == "__main__":
    import argparse
    import os, pickle
    from scripts.Preprocess import constrained_kmeans
  

    parser = argparse.ArgumentParser(
        description="Solve the depot-centroid ICVRP instance for one dataset."
    )
    parser.add_argument("--dat_id", type=int, default=0)
    parser.add_argument("--run_seed", type=int, default=0)
    parser.add_argument("--p", type=int, default=1)
    args = parser.parse_args()

    dat_id = 0
    run_seed = 0

    file_path = "C:/Users/shrey/OneDrive/Desktop/SOA_WORKS/VRP/CLVRP/CLVRP_public/dataset0.pkl"

    with open(file_path, "rb") as f:
        customers = pickle.load(f)
    assignments, centroids = constrained_kmeans(customers, random_state=args.run_seed)
    depot = np.mean(customers, axis=0)

    points_icvrp = [depot] + list(centroids)
    w = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            w[i, j] = np.sqrt(np.sum((points_icvrp[i] - points_icvrp[j]) ** 2))

    var_list, label_to_index, Q, g, c, _, _, _, _ = build_qubo(w)
    std_counts = run_standard_qaoa(w, p=args.p, seed=args.run_seed)
    ma_counts = run_ma_qaoa(w, p=args.p, seed=args.run_seed)

    results = {
        "dataset_id": args.dat_id,
        "run_seed": args.run_seed,
        "p": args.p,
        "assignments": assignments,
        "centroids": centroids,
        "depot": depot,
        "distance_matrix": w,
        "var_list": var_list,
        "label_to_index": label_to_index,
        "qubo_matrix": Q,
        "qubo_linear": g,
        "qubo_constant": c,
        "std_counts": std_counts,
        "ma_counts": ma_counts,
    }

    print(results.keys())

