import numpy as np

# 'model_this_run' should fall in the set [MLP,MLP_REGRESSION,GCN,GAT,GATv2,TRANSFORMER,GINE]

GLOBALPARAMETERS = {
    'no_of_epochs': 200,
    'model_this_run' : 'MLP',
    'train_size': 0.15,
    'valid_size': 0.05,
}


MLP_HYPERPARAMETERS = {
    'has_edge_info' : False,
    "batch_size": 64,
    "learning_rate": 0.000986,
    "model_layers": 2,
    "model_embedding_size" : 512,
    "model_dense_neurons" : 64,
    "model_linear_dropout_rate" : 0.0211,
    "model_embedding_dropout_rate" : 0.149

}

MLP_REGRESSION_HYPERPARAMETERS = {
    'has_edge_info' : False,
    "batch_size": 16,
    "learning_rate": 0.000933,
    "model_layers": 3,
    "model_dense_neurons_1" : 64,
    "model_dense_neurons_2" : 64,
    "model_dense_neurons_3" : 64,
    "model_dense_neurons_4" : 64,
    "model_dense_neurons_5" : 64,
    "model_dense_neurons_final" : 64,
    "model_linear_dropout_rate" : 0.01

}

MLP_UNCERTAINTY_HYPERPARAMETERS = {
    'has_edge_info' : False,
    "batch_size": 64,
    "learning_rate": 0.01,
    "model_layers": 5,
    "model_dense_neurons_1" : 64,
    "model_dense_neurons_2" : 64,
    "model_dense_neurons_3" : 64,
    "model_dense_neurons_4" : 64,
    "model_dense_neurons_5" : 64,
    "model_dense_neurons_final" : 64,
    "model_linear_dropout_rate" : 0.01

}

GCN_HYPERPARAMETERS = {
    'has_edge_info' : False,
    "batch_size": 64,
    "learning_rate": 0.000226,
    "model_embedding_size": 1024,
    "model_layers": 2,
    "model_gnn_dropout_rate": 0.0472,
    "model_linear_dropout_rate": 0.0413,
    "model_dense_neurons": 32,
}

GAT_HYPERPARAMETERS = {
    'has_edge_info' : True,
    "batch_size": 64,
    "learning_rate": 0.000087,
    "model_embedding_size": 1024,
    "model_layers": 3,
    "model_gnn_dropout_rate": 0.112,
    "model_linear_dropout_rate": 0.0586,
    "model_dense_neurons": 64,
    'no_of_heads': 3,
    'gat_dropout_rate': 0.0178,
}

GATv2_HYPERPARAMETERS = {
    'has_edge_info' : True,
    "batch_size": 64,
    "learning_rate": 0.000341,
    "model_embedding_size": 256,
    "model_layers": 5,
    "model_gnn_dropout_rate": 0.145,
    "model_linear_dropout_rate": 0.0447,
    "model_dense_neurons": 128,
    'no_of_heads': 4,
    'gat_dropout_rate': 0.0216,
}

TRANSFORMER_HYPERPARAMETERS = {
    'has_edge_info' : True,
    "batch_size":  64,
    "learning_rate": 0.0009546,  
    "model_embedding_size": 128,
    "model_attention_heads": 3,
    "model_layers": 3,
    "model_dense_neurons": 256,
    "model_gnn_dropout_rate": 0.173,
    "model_linear_dropout_rate": 0.0195,
    'model_transformer_dropout_rate': 0.0167
}

GINE_HYPERPARAMETERS = {
    'has_edge_info' : True,
    "batch_size":  64,
    "learning_rate": 0.000104,  
    "model_embedding_size": 1024,
    "model_layers": 3,
    "model_dropout_rate": 0.0168,
    "model_dense_neurons": 64,
    "model_gnn_dropout_rate": 0.168,
    "model_linear_dropout_rate": 0.0597
}