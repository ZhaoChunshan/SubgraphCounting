import networkx
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from data_loader.data_utils import *


class QuerySet:
    def __init__(self, data_dir, dataset_name, query_size, train_set_max_id):
        self.data_graph = QuerySet.__load_data_graph(data_dir, dataset_name)
        self.train_queries, self.test_queries = QuerySet.__get_train_test_set(
            data_dir, dataset_name, query_size, train_set_max_id)

    @staticmethod
    def __load_data_graph(data_dir, dataset_name):
        data_dir = Path(data_dir)
        dataset_dir = data_dir / dataset_name
        data_graph_path = dataset_dir / 'data_graph' / (dataset_name + ".graph")
        return load_graph(data_graph_path)

    @staticmethod
    def __get_train_test_set(data_dir, dataset_name, query_size, train_set_max_id):
        """
        a train/test set is a list of (query_graph, query_name, count)
        """
        train_set, test_set = [], []

        data_dir = Path(data_dir)
        dataset_dir = data_dir / dataset_name
        query_dir = dataset_dir / 'query_graph'
        ground_truth = load_ground_truth(dataset_dir / 'ground_truth.txt')

        for query_path in query_dir.iterdir():
            query_name = query_path.stem
            density, cur_size, query_id = get_query_key(query_name)
            if cur_size != query_size or query_name not in ground_truth:
                continue
            query = load_graph(query_path)
            data_tuple = (query, query_name, ground_truth[query_name])
            if query_id <= train_set_max_id:
                train_set.append(data_tuple)
            else:
                test_set.append(data_tuple)

        return train_set, test_set


class MPNNDataset(Dataset):
    def __init__(self, queries, data_graph: networkx.Graph):
        """
        class to prepare data for Message Passing Graph Neural Networks.
        :param queries: List containing tuple (query_graph, query_name, count),
        """
        self.label_count = 1 + max(networkx.get_node_attributes(data_graph, 'label').values())
        self.data = [self.__transform(query[0], query[2]) for query in queries]

    def __transform(self, query_graph, count):
        """
        Transform query_graph to tensors that will be fed forward to MPNN.
        Return node_features, edge_index, log(count+1)
        """
        node_features = one_hot_encoding(query_graph, self.label_count)
        edge_index = get_edge_index(query_graph)
        log_count = torch.tensor(np.log(count))
        return node_features, edge_index, log_count

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class GraphormerDataset(Dataset):
    pass


class GraphTransformerDataset(Dataset):
    pass
