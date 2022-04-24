import torch
import torch.nn.functional as F 
from torch.nn import Dropout, Linear, BatchNorm1d, ModuleList, Sequential, ReLU
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv, GINEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class MLP(torch.nn.Module):
    def __init__(self, model_params):
        super(MLP, self).__init__()
        feature_size = model_params["feature_size"]
        self.n_layers = model_params["model_layers"]
        dense_neurons = model_params["model_embedding_size"]
        dense_neurons_final = model_params["model_dense_neurons"]
        
        embedding_dropout_rate = model_params["model_embedding_dropout_rate"]
        linear_dropout_rate = model_params["model_linear_dropout_rate"]

        self.conv_layers = ModuleList([])

        self.initial_conv = Linear(feature_size, dense_neurons)
        self.embedding_dropout = Dropout(embedding_dropout_rate)
        self.linear_dropout = Dropout(linear_dropout_rate)

        for i in range(self.n_layers-1):
            self.conv_layers.append(Linear(dense_neurons, 
                                            dense_neurons))
        
        self.linear = Linear(dense_neurons, dense_neurons_final)
        self.out = Linear(dense_neurons_final, 1)

    def forward(self, x):
        # First Conv layer
        x = self.initial_conv(x)
        x = torch.relu(x)
        x = self.embedding_dropout(x)

        # Other Conv layers
        for i in range(self.n_layers-1):
            x = self.conv_layers[i](x)
            x = torch.relu(x)
            x = self.embedding_dropout(x)
        
        # Apply a final (linear) classifier.
        x = self.linear(x)
        x = self.linear_dropout(x)

        x = self.out(x)
        x = self.linear_dropout(x)
        x = torch.sigmoid(x)
        return x

'''
----------------------------------------------------------------------------------------
'''

class MLP_REGRESSION(torch.nn.Module):
    def __init__(self, model_params):
        super(MLP_REGRESSION, self).__init__()
        feature_size = model_params["feature_size"]
        self.n_layers = model_params["model_layers"]
        dense_neurons_1 = model_params["model_dense_neurons_1"]
        dense_neurons_2 = model_params["model_dense_neurons_2"]
        dense_neurons_3 = model_params["model_dense_neurons_3"]
        dense_neurons_4 = model_params["model_dense_neurons_4"]
        dense_neurons_5 = model_params["model_dense_neurons_5"]
        dense_neurons_final = model_params["model_dense_neurons_final"]
        linear_dropout_rate = model_params["model_linear_dropout_rate"]

        dense_neurons = [dense_neurons_1,dense_neurons_2,dense_neurons_3,dense_neurons_4,dense_neurons_5]

        self.conv_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        self.initial_conv = Linear(feature_size, dense_neurons_1)
        self.linear_dropout = Dropout(linear_dropout_rate)

        for i in range(self.n_layers-1):
            self.conv_layers.append(Linear(dense_neurons[i], 
                                            dense_neurons[i+1]))
            self.bn_layers.append(BatchNorm1d(dense_neurons[i+1]))
        
        self.linear = Linear(dense_neurons[self.n_layers-1], dense_neurons_final)
        self.out = Linear(dense_neurons_final, 1)

    def forward(self, x):
        # First Conv layer
        x = self.initial_conv(x)
        x = torch.relu(x)
        x = self.linear_dropout(x)

        # Other Conv layers
        for i in range(self.n_layers-1):
            x = self.conv_layers[i](x)
            x = torch.relu(x)
            x = self.linear_dropout(x)
        
        # Apply a final (linear) classifier.
        x = self.linear(x)
        x = self.linear_dropout(x)

        x = self.out(x)
        x = self.linear_dropout(x)
        return x

'''
----------------------------------------------------------------------------------------
'''

class MLP_UNCERTAINTY(torch.nn.Module):
    def __init__(self, model_params):
        super(MLP_UNCERTAINTY, self).__init__()
        feature_size = model_params["feature_size"]
        self.n_layers = model_params["model_layers"]
        dense_neurons_1 = model_params["model_dense_neurons_1"]
        dense_neurons_2 = model_params["model_dense_neurons_2"]
        dense_neurons_3 = model_params["model_dense_neurons_3"]
        dense_neurons_4 = model_params["model_dense_neurons_4"]
        dense_neurons_5 = model_params["model_dense_neurons_5"]
        dense_neurons_final = model_params["model_dense_neurons_final"]
        linear_dropout_rate = model_params["model_linear_dropout_rate"]

        dense_neurons = [dense_neurons_1,dense_neurons_2,dense_neurons_3,dense_neurons_4,dense_neurons_5]

        self.conv_layers = ModuleList([])

        self.initial_conv = Linear(feature_size, dense_neurons_1)
        self.linear_dropout = Dropout(linear_dropout_rate)

        for i in range(self.n_layers-1):
            self.conv_layers.append(Linear(dense_neurons[i], 
                                            dense_neurons[i+1]))
        
        self.linear = Linear(dense_neurons[self.n_layers-1], dense_neurons_final)
        self.out = Linear(dense_neurons_final, 1)

    def forward(self, x):
        # First Conv layer
        x = self.initial_conv(x)
        x = torch.relu(x)
        x = self.linear_dropout(x)

        # Other Conv layers
        for i in range(self.n_layers-1):
            x = self.conv_layers[i](x)
            x = torch.relu(x)
            x = self.linear_dropout(x)
        
        # Apply a final (linear) classifier.
        x = self.linear(x)
        x = self.linear_dropout(x)

        x = self.out(x)
        x = self.linear_dropout(x)
        return x

'''
----------------------------------------------------------------------------------------
'''

class GCN(torch.nn.Module):
    def __init__(self, model_params):
        super(GCN, self).__init__()
        feature_size = model_params["feature_size"]
        embedding_size = model_params["model_embedding_size"]
        self.n_layers = model_params["model_layers"]
        gnn_dropout_rate = model_params["model_gnn_dropout_rate"]
        linear_dropout_rate = model_params["model_linear_dropout_rate"]
        dense_neurons = model_params["model_dense_neurons"]

        self.conv_layers = ModuleList([])

        self.initial_conv = GCNConv(feature_size, embedding_size)
        self.gnn_dropout = Dropout(gnn_dropout_rate)
        self.linear_dropout = Dropout(linear_dropout_rate)

        for i in range(self.n_layers):
            self.conv_layers.append(GCNConv(embedding_size, 
                                            embedding_size))

        # Output layer *2 for global pooling
        self.linear = Linear(embedding_size*2, dense_neurons)
        self.out = Linear(dense_neurons, 1)

    def forward(self, x, edge_index, batch):
        # First Conv layer
        x = self.initial_conv(x, edge_index)
        x = torch.relu(x)
        x = self.gnn_dropout(x)

        # Other Conv layers
        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index)
            x = torch.relu(x)
            x = self.gnn_dropout(x)
       
        # Global Pooling 
        x = torch.cat([gmp(x, batch), 
                       gap(x, batch)], dim=1)
        
        # Apply a final (linear) classifier.
        x = self.linear(x)
        x = self.linear_dropout(x)

        x = self.out(x)
        x = self.linear_dropout(x)
        x = torch.sigmoid(x)
        return x

'''
----------------------------------------------------------------------------------------
'''

class GAT(torch.nn.Module):
    def __init__(self, model_params):
        super(GAT, self).__init__()
        feature_size = model_params["feature_size"]
        embedding_size = model_params["model_embedding_size"]
        self.n_layers = model_params["model_layers"]
        gnn_dropout_rate = model_params["model_gnn_dropout_rate"]
        linear_dropout_rate = model_params["model_linear_dropout_rate"]
        gat_dropout_rate = model_params["gat_dropout_rate"]
        no_of_heads = model_params["no_of_heads"]
        edge_dim = model_params['edge_dim']

        dense_neurons = model_params["model_dense_neurons"]

        self.conv_layers = ModuleList([])

        self.initial_conv = GATConv(feature_size, 
                                    embedding_size,
                                    heads=no_of_heads,
                                    dropout=gat_dropout_rate,
                                    edge_dim=edge_dim)

        self.gnn_dropout = Dropout(gnn_dropout_rate)
        self.linear_dropout = Dropout(linear_dropout_rate)

        for i in range(self.n_layers):
            self.conv_layers.append(GATConv(embedding_size*no_of_heads, 
                                            embedding_size,
                                            heads=no_of_heads,
                                            dropout=gat_dropout_rate,
                                            edge_dim=edge_dim))

        # Output layer *2 for global pooling
        self.linear = Linear(embedding_size*2*no_of_heads, dense_neurons)
        self.out = Linear(dense_neurons, 1)

    def forward(self, x, edge_attr, edge_index, batch):
        # First Conv layer
        x = self.initial_conv(x, edge_index, edge_attr)
        x = torch.relu(x)

        # Other Conv layers
        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(x)
            x = self.gnn_dropout(x)
       
        # Global Pooling 
        x = torch.cat([gmp(x, batch), 
                       gap(x, batch)], dim=1)
        
        # Apply a final (linear) classifier.
        x = self.linear(x)
        x = self.linear_dropout(x)

        x = self.out(x)
        x = self.linear_dropout(x)
        x = torch.sigmoid(x)

        return x


'''
----------------------------------------------------------------------------------------
'''

class GATv2(torch.nn.Module):
    def __init__(self, model_params):
        super(GATv2, self).__init__()
        feature_size = model_params["feature_size"]
        embedding_size = model_params["model_embedding_size"]
        self.n_layers = model_params["model_layers"]
        gnn_dropout_rate = model_params["model_gnn_dropout_rate"]
        linear_dropout_rate = model_params["model_linear_dropout_rate"]
        gat_dropout_rate = model_params["gat_dropout_rate"]
        no_of_heads = model_params["no_of_heads"]
        edge_dim = model_params['edge_dim']

        dense_neurons = model_params["model_dense_neurons"]

        self.conv_layers = ModuleList([])

        self.initial_conv = GATv2Conv(feature_size, 
                                    embedding_size,
                                    heads=no_of_heads,
                                    dropout=gat_dropout_rate,
                                    edge_dim=edge_dim)

        self.gnn_dropout = Dropout(gnn_dropout_rate)
        self.linear_dropout = Dropout(linear_dropout_rate)

        for i in range(self.n_layers):
            self.conv_layers.append(GATv2Conv(embedding_size*no_of_heads, 
                                            embedding_size,
                                            heads=no_of_heads,
                                            dropout=gat_dropout_rate,
                                            edge_dim=edge_dim))

        # Output layer *2 for global pooling
        self.linear = Linear(embedding_size*2*no_of_heads, dense_neurons)
        self.out = Linear(dense_neurons, 1)

    def forward(self, x, edge_attr, edge_index, batch):
        # First Conv layer
        x = self.initial_conv(x, edge_index, edge_attr)
        x = torch.relu(x)

        # Other Conv layers
        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(x)
            x = self.gnn_dropout(x)
       
        # Global Pooling 
        x = torch.cat([gmp(x, batch), 
                       gap(x, batch)], dim=1)
        
        # Apply a final (linear) classifier.
        x = self.linear(x)
        x = self.linear_dropout(x)

        x = self.out(x)
        x = self.linear_dropout(x)
        x = torch.sigmoid(x)

        return x


'''
----------------------------------------------------------------------------------------
'''

class TRANSFORMER(torch.nn.Module):
    def __init__(self, model_params):
        super(TRANSFORMER, self).__init__()
        feature_size = model_params["feature_size"]
        embedding_size = model_params["model_embedding_size"]
        no_of_heads = model_params["no_of_heads"]
        self.n_layers = model_params["model_layers"]
        transformer_dropout_rate = model_params["model_transformer_dropout_rate"]
        dense_neurons = model_params["model_dense_neurons"]
        edge_dim = model_params["edge_dim"]
        gnn_dropout_rate = model_params["model_gnn_dropout_rate"]
        linear_dropout_rate = model_params["model_linear_dropout_rate"]

        self.conv_layers = ModuleList([])

        # Transformation layer
        self.conv1 = TransformerConv(feature_size, 
                                    embedding_size, 
                                    heads=no_of_heads, 
                                    dropout=transformer_dropout_rate,
                                    edge_dim=edge_dim,
                                    beta=True)
        self.gnn_dropout = Dropout(gnn_dropout_rate)
        self.linear_dropout = Dropout(linear_dropout_rate)

        # Other layers
        for i in range(self.n_layers):
            self.conv_layers.append(TransformerConv(embedding_size*no_of_heads, 
                                                    embedding_size, 
                                                    heads=no_of_heads, 
                                                    dropout=transformer_dropout_rate,
                                                    edge_dim=edge_dim,
                                                    beta=True))
        # Linear layers
        self.linear = Linear(embedding_size*2*no_of_heads, dense_neurons)
        self.out = Linear(dense_neurons, 1) 

    def forward(self, x, edge_attr, edge_index, batch):
        # Initial transformation
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(x)
        
        # Global Pooling 
        x = torch.cat([gmp(x, batch), 
                       gap(x, batch)], dim=1)
        
        # Apply a final (linear) classifier.
        x = self.linear(x)
        x = self.linear_dropout(x)

        x = self.out(x)
        x = self.linear_dropout(x)
        x = torch.sigmoid(x)

        return x

'''
----------------------------------------------------------------------------------------
'''


class GINE(torch.nn.Module):
    def __init__(self, model_params):
        super(GINE, self).__init__()
        feature_size = model_params["feature_size"]
        embedding_size = model_params["model_embedding_size"]
        self.n_layers = model_params["model_layers"]
        dense_neurons = model_params["model_dense_neurons"]
        edge_dim = model_params["edge_dim"]
        gnn_dropout_rate = model_params["model_gnn_dropout_rate"]
        linear_dropout_rate = model_params["model_linear_dropout_rate"]

        self.gnn_dropout = Dropout(gnn_dropout_rate)
        self.linear_dropout = Dropout(linear_dropout_rate)
        
        self.conv_layers = ModuleList([])

        self.conv1 = GINEConv(Sequential(Linear(feature_size, embedding_size),
                           ReLU(),
                           Linear(embedding_size, embedding_size)),
                           train_eps=True,
                           edge_dim=edge_dim)
        
        # Other layers
        for i in range(self.n_layers):
            self.conv_layers.append(GINEConv(Sequential(Linear(embedding_size, embedding_size),
                           ReLU(),
                           Linear(embedding_size, embedding_size)),
                           train_eps=True,
                           edge_dim=edge_dim))
        
        # Linear layers
        self.linear = Linear(embedding_size*2, dense_neurons)
        self.out = Linear(dense_neurons, 1)
        

    
    def forward(self, x, edge_attr, edge_index, batch):
        # Initial transformation
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        
        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(x)

            # Global Pooling 
        x = torch.cat([gmp(x, batch), 
                       gap(x, batch)], dim=1)
        
        # Apply a final (linear) classifier.
        x = self.linear(x)
        x = self.linear_dropout(x)

        x = self.out(x)
        x = self.linear_dropout(x)
        x = torch.sigmoid(x)

        return x