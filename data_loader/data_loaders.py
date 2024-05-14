import torch
import networkx
from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.dataset import QuerySet, MPNNDataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class SubgraphCountingDataLoader(BaseDataLoader):
    """
    Subgraph counting data loader using BaseDataLoader
    """
    def __init__(self, input_type, data_dir, dataset, query_size, batch_size, shuffle=True, train_set_max_id=160,
                 validation_split=0.0, num_workers=1):
        self.query_set = QuerySet(data_dir, dataset, query_size, train_set_max_id)
        self.train_queries = self.query_set.train_queries
        self.data_graph = self.query_set.data_graph

        if input_type == "MPNNDataset":
            self.dataset = MPNNDataset(self.train_queries, self.data_graph)
            collate_fn = mpnn_collate_fn
        else:
            raise NotImplementedError("SubgraphCountingDataLoader only supports MPNNDataset.")

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn)

    def input_feature_size(self):
        return 1 + max(networkx.get_node_attributes(self.data_graph, 'label').values())


def mpnn_collate_fn(batch):
    x = [sample[0] for sample in batch]
    edge_index = [sample[1] for sample in batch]
    log_count = torch.tensor([sample[2] for sample in batch], dtype=torch.float)
    return x, edge_index, log_count
