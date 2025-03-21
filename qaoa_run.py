import time

from qiskit_aer import AerSimulator
from qiskit_algorithms import QAOA
from qiskit_ibm_runtime import SamplerV1 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeOsaka
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo

from db import Database
from qiskit_key import get_key
from utils.graph import load_graph
from utils.model import create_model
from utils.parameters import *

SERVICE_KEY = get_key()

# Create dicts for data
MIXERS = Mixers()


def qp_to_qubo(qp):
    qp2qubo = QuadraticProgramToQubo()
    qubo = qp2qubo.convert(qp)
    qubitOp, offset = qubo.to_ising()
    return qubo, qubitOp.num_qubits


def run_qaoa(qubo, optimizer, mixer, reps):
    print("Running QAOA...")
    qaoa_mes = QAOA(sampler=noisy_sampler, mixer=mixer, reps=reps, optimizer=optimizer)
    qaoa = MinimumEigenOptimizer(qaoa_mes)
    qaoa_result = qaoa.solve(qubo)

    return qaoa_result.samples[0]


graphs = []

print("Loading graphs...")
for i in range(3):
    g = load_graph(f"graphs/graph_5_{i}")
    model = create_model(g)
    graphs.append((g, model))

print("Creating backend...")
# Ignore the warning message since we need to use the deprecated sampler
aer_sim = AerSimulator().from_backend(FakeOsaka())
noisy_sampler = Sampler(backend=aer_sim)

with Database() as db:
    finished = db.get_finished_qaoa()

# total_permutations = (len(optimizers.keys()) * len(MIXERS.mixer_list) * DEPTH) - len(
#     finished
# )
# current_permutation = 0

for opt_key, opt_val in optimizers.items():
    for mixer in MIXERS.mixer_list:
        for depth in range(1, DEPTH + 1):
            # current_permutation += 1
            print("-------------------------------------------")
            # print(
            #     f"Running iteration {current_permutation} of {total_permutations} with current settings: "
            # )
            print(f"Optimizer: {opt_key}")
            print(f"Mixer: {mixer}")
            print(f"Depth: {depth}")
            print()

            graph_index = 0

            start = time.time()
            try:
                for graph, model in graphs:
                    graph_index += 1
                    print(f"Running graph: {graph_index}")
                    qubo, num_qubits = qp_to_qubo(model)
                    for i in range(5):
                        iteration = i+1
                        print(f"Running iteration {iteration}")
                        if (depth, mixer, opt_key, graph_index, iteration) in finished:
                            print(f"Skipped ({depth}, {mixer}, {opt_key}, {graph_index}, {iteration})")
                            continue
                        exec_start = time.time()
                        best_sample = run_qaoa(
                            qubo, opt_val, MIXERS.get_mixer(num_qubits, mixer), depth
                        )
                        exec_stop = time.time()
                        sample_cost = int(best_sample.fval)
                        valid = model.get_feasibility_info(best_sample.x)[0]
                        print(f"Graph {graph_index} iteration {iteration} finished")
                        print("Saving to database...")
                        print()
                        with Database() as db:
                            db.insert_data(
                                {
                                    "algorithm": "qaoa",
                                    "mixer": mixer,
                                    "optimizer": opt_key,
                                    "depth": depth,
                                    "cost": sample_cost,
                                    "valid": 1 if valid else 0,
                                    "graph": graph_index,
                                    "iteration": iteration,
                                    "optimal": graph.optimal,
                                    "time": exec_stop - exec_start,
                                }
                            )
            except Exception as e:
                print("Skipped due to the following exception:")
                print(e)
                print("\nSKIP")
                continue

            end = time.time()
            print(f"Time: {end - start}s")
            print("-------------------------------------------")

print()
print("End")
