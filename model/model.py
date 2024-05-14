import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class GCN(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GCN, self).__init__()

        self.num_layers = num_layers

        self.conv1 = GCNConv(input_size, hidden_size)

        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(GCNConv(hidden_size, hidden_size))

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        for layer in self.hidden_layers:
            x = layer(x, edge_index)
            x = F.relu(x)

        # Mean-pooling to get the graph level representation
        x = torch.mean(x, dim=0)

        # Predict result
        x = self.fc(x)

        return x


class GIN(BaseModel):
    pass


class GAT(BaseModel):
    pass


class GraphSAGE(BaseModel):
    pass
