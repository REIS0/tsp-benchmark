import networkx as nx
import matplotlib.pyplot as plt
import random
import itertools


class Graph:
    def __init__(self, graph: nx.Graph):
        self.__graph = graph
        self.__optimal = find_optimal(self.graph)

    @property
    def graph(self) -> nx.Graph:
        return self.__graph

    @property
    def optimal(self) -> int:
        return self.__optimal

    @property
    def n_nodes(self) -> int:
        return self.__graph.number_of_nodes()

    def plot_graph(self) -> None:
        pos = nx.circular_layout(self.__graph)
        nx.draw_networkx_nodes(self.__graph, pos)
        nx.draw_networkx_edges(self.__graph, pos)
        nx.draw_networkx_labels(self.__graph, pos)
        nx.draw_networkx_edge_labels(self.__graph, pos, nx.get_edge_attributes(self.__graph, 'weight'), font_size=10,
                                     font_color="b")
        plt.plot()

    def get_edge_weight(self, node1: int, node2: int) -> int:
        return self.__graph.edges[str(node1), str(node2)]['weight']


def get_edge_weight(graph: nx.Graph, node1: int, node2: int) -> int:
    return graph.edges[node1, node2]['weight']


def create_graph(n_nodes: int, edge_list: list) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(edge_list)
    return graph


class GraphFactory:
    @staticmethod
    def create_graph(n_nodes: int) -> nx.Graph:
        edge_list = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                # weight = W_i,j
                edge_list.append((i, j, {"weight": random.randint(1, 20)}))
        return create_graph(n_nodes, edge_list)


def save_graph(graph: nx.Graph, filename: str) -> None:
    nx.write_gexf(graph, f"{filename}.gexf")


def load_graph(filename: str) -> Graph:
    graph = nx.read_gexf(f"{filename}.gexf")
    return Graph(graph)


def find_optimal(graph: nx.Graph) -> int:
    all_paths = []
    permutations = list(itertools.permutations([i for i in range(1, graph.number_of_nodes())]))
    for p in permutations:
        path = list(p)
        path.insert(0, 0)
        path.append(0)
        all_paths.append(path)

    min_cost = float("inf")
    for path in all_paths:
        cost = sum(graph.edges[str(i), str(j)]["weight"] for i, j in nx.utils.pairwise(path))
        if cost < min_cost:
            min_cost = cost

    return min_cost
