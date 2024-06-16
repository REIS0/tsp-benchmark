import networkx as nx
import multiprocessing as mp
from utils.graph import get_edge_weight


def plot_result_bitstring(bitstring: str, graph: nx.Graph, qp):
    # bitstring start from right to left
    print(f"Bitstring: {bitstring}")
    bitstr_inv = [int(bitstring[-1 - i]) for i in range(len(bitstring))]
    print(f"Bitstring invert: {bitstr_inv}")
    final_path = []
    for key, index in qp.variables_index.items():
        if bitstr_inv[index] == 1:
            final_path.append(key)
    final_path = sorted(final_path, key=lambda x: int(x[-1]))
    print(final_path)
    final_path = [int(x[2]) for x in final_path]

    weight = (
            get_edge_weight(graph, 0, final_path[0]) +
            sum([get_edge_weight(graph, final_path[i], final_path[i + 1]) for i in range(len(final_path) - 1)]) +
            get_edge_weight(graph, final_path[-1], 0)
    )
    print(f"Total weight: {weight}")

    plot_graph = nx.DiGraph()
    plot_graph.add_edges_from(
        [(0, final_path[0])] +
        [(final_path[i], final_path[i + 1]) for i in range(len(final_path) - 1)] +
        [(final_path[-1], 0)]
    )
    nx.draw_circular(plot_graph, with_labels=True)


def get_info(solution_sample, qp) -> dict:
    # qp: quadratic model before converting to qubo
    return {
        'feasible': qp.get_feasibility_info(solution_sample.x)[0],
        'cost': solution_sample.fval,
        'probability': solution_sample.probability
    }


def process_samples(samples, qp):
    pool = mp.Pool(mp.cpu_count())
    processes = [pool.apply_async(get_info, args=(s, qp)) for s in samples]
    result = [p.get() for p in processes]
    return result


def get_feasibility_ratio(samples: list) -> float:
    # higher = better
    total = len(samples)
    feasible = len([sample for sample in samples if sample['feasible']])
    return feasible / total


def get_cost_ratio(samples: list, optimal: int) -> float:
    # higher = better
    feasible = [sample for sample in samples if sample['feasible']]
    if len(feasible) == 0:
        return 0
    median = sum([sample['cost'] for sample in feasible]) / len(feasible)
    return optimal / median


def get_rank(samples: list, best_sample: dict) -> int:
    # less = better
    feasible = [sample for sample in samples if sample['feasible']]
    rank = 1
    for sample in feasible:
        if sample['probability'] > best_sample['probability']:
            rank += 1
    return rank
