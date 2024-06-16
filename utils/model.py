from qiskit_optimization import QuadraticProgram

from utils.graph import Graph
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model import Model


def __create_variables(n: int, mdl: Model) -> dict:
    x = {}
    for i in range(1, n):
        for t in range(1, n):
            x[(i, t)] = mdl.binary_var(name=f"x_{i},{t}")
    return x


# 8
def __cost_function(n: int, vars: dict, graph: Graph, mdl: Model) -> float:
    v = 0
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            v += graph.get_edge_weight(i, j) * mdl.sum(vars[(i, t)] * vars[(j, t + 1)] for t in range(1, n - 1))

    v += mdl.sum([graph.get_edge_weight(0, i) * (vars[(i, 1)] + vars[(i, n - 1)]) for i in range(1, n)])
    return v


def __constraints(n: int, vars: dict, mdl: Model) -> None:
    # multiple city at same time constraint
    for t in range(1, n):
        mdl.add_constraint(
            mdl.sum(vars[(i, t)] for i in range(1, n)) == 1,
            f"t_{t}"
        )
    # revisit city constrain
    for i in range(1, n):
        mdl.add_constraint(
            mdl.sum(vars[(i, t)] for t in range(1, n)) == 1,
            f"i_{i}"
        )


def create_model(graph: Graph) -> QuadraticProgram:
    mdl = Model("TSP")
    vars = __create_variables(graph.n_nodes, mdl)
    mdl.minimize(__cost_function(graph.n_nodes, vars, graph, mdl))
    __constraints(graph.n_nodes, vars, mdl)

    return from_docplex_mp(mdl)
