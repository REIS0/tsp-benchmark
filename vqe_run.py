import time
from utils.parameters import *
from utils.graph import load_graph
from utils.utils import *
from utils.model import create_model
from qiskit_key import get_key
from db import Database
from qiskit_ibm_runtime.fake_provider import FakeOsaka
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV1 as Sampler
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import SamplingVQE

SERVICE_KEY = get_key()

# Create dicts for data
ANSATZ = Ansatz()

graphs = []

print("Loading graphs...")
for i in range(10):
    g = load_graph(f"graphs/graph_5_{i}")
    model = create_model(g)
    graphs.append((g, model))

print("Creating backend...")
# Ignore the warning message since we need to use the deprecated sampler
aer_sim = AerSimulator().from_backend(FakeOsaka())
noisy_sampler = Sampler(backend=aer_sim)

finished = None
with Database() as db:
    finished = db.get_finished_vqe()

def qp_to_qubo(qp):
    qp2qubo = QuadraticProgramToQubo()
    qubo = qp2qubo.convert(qp)
    qubitOp, offset = qubo.to_ising()
    return qubo, qubitOp.num_qubits


def run_vqe(qubo, optimizer, ansatz):
    print("Running VQE...")
    vqe = SamplingVQE(sampler=noisy_sampler, ansatz=ansatz, optimizer=optimizer)
    optimizer = MinimumEigenOptimizer(vqe)
    vqe_result = optimizer.solve(qubo)

    return vqe_result.samples


total_permutations = (len(optimizers.keys()) * len(ANSATZ.ansatz_list) * DEPTH) - len(finished)
current_permutation = 0

for opt_key, opt_val in optimizers.items():
    for ansatz in ANSATZ.ansatz_list:
        for depth in range(1, DEPTH + 1):
            current = (depth, ansatz, opt_key)
            if current in finished:
                continue
            current_permutation += 1
            print("-------------------------------------------")
            print(f"Running iteration {current_permutation} of {total_permutations} with current settings: ")
            print(f"Optimizer: {opt_key}")
            print(f"Ansatz: {ansatz}")
            print(f"Depth: {depth}")
            print()

            feasibility_list = []
            cost_list = []
            rank_list = []
            graph_index = 0
            exec_start = None
            exec_stop = None

            start = time.time()
            try:
                for graph, model in graphs:
                    graph_index += 1
                    print(f"Running graph: {graph_index}")
                    qubo, num_qubits = qp_to_qubo(model)
                    exec_start = time.time()
                    raw_samples = run_vqe(qubo, opt_val, ANSATZ.get_ansatz(num_qubits, ansatz, depth))
                    exec_stop = time.time()
                    print("Processing samples...")
                    samples = process_samples(raw_samples, model)
                    print(f"Graph {graph_index} finished")
                    print()

                    feasibility_list.append(get_feasibility_ratio(samples))
                    cost_list.append(get_cost_ratio(samples, graph.optimal))
                    rank_list.append(get_rank(samples, get_info(raw_samples[0], model)))
            except Exception as e:
                print("Skipped due to the following exception:")
                print(e)
                print("\nSKIP")
                continue

            print("Saving to database...")
            with Database() as db:
                db.insert_data(
                    {
                        "algorithm": "vqe",
                        "depth": depth,
                        "ansatz": ansatz,
                        "optimizer": opt_key,
                        "feasibility_ratio": sum(feasibility_list) / len(feasibility_list),
                        "cost_ratio": sum(cost_list) / len(cost_list),
                        "rank": sum(rank_list) / len(rank_list),
                        "time": exec_stop - exec_start,
                    }
                )
            end = time.time()
            print(f"Time: {end - start}s")
            print("-------------------------------------------")

print()
print("End")
