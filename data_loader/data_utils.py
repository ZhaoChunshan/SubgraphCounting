import torch
import networkx
import numpy as np


def load_graph(file):
    g = networkx.Graph()
    vertices, edges = [], []
    with open(file) as handle:
        for line in handle:
            if not line:
                continue
            if line[0] == "t":
                pass
            elif line[0] == "v":
                vertex_id, label = map(int, line.split()[1:3])
                vertices.append((vertex_id, {"label": label}))
            elif line[0] == "e":
                src, tgt = map(int, line.split()[1:3])
                edges.append((src, tgt))
    g.add_nodes_from(vertices)
    g.add_edges_from(edges)
    return g


def get_query_key(query_name: str):
    dense = "dense"
    if "sparse" in query_name:
        dense = "sparse"

    size_id_token = query_name[6:][query_name[6:].find("_")+1:]
    size, query_id = map(int, size_id_token.split("_"))

    return dense, size, query_id


def load_ground_truth(file):
    ground_truth = {}
    with open(file) as handle:
        for line in handle:
            if not line:
                continue
            query_name, count = line.split()
            ground_truth[query_name] = int(count)
    return ground_truth


def get_edge_index(graph: networkx.Graph):
    # Sparse representation of the graph
    adj = networkx.to_scipy_sparse_array(graph).tocoo()

    # Row index of nonzero elements
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    # Column index of nonzero elements
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)

    # Stack the tensor
    edge_index = torch.stack([row, col], dim=0)

    return edge_index


def one_hot_encoding(graph: networkx.Graph, max_label_count):
    features = torch.zeros(size=(graph.number_of_nodes(), max_label_count), dtype=torch.float)
    for v in graph.nodes():
        label = graph.nodes[v]["label"]
        features[v][label] = 1.0
    return features


def frequency_based_encoding():
    pass


def pretraining_based_encoding():
    pass


if __name__ == "__main__":
    Q4 = load_graph("../data/hprd/query_graph/query_dense_4_1.graph")
    edge_idx = get_edge_index(Q4)
    print(edge_idx)
