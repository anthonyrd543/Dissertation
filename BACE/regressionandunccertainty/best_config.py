import numpy as np

# 'model_this_run' should fall in the set [MLP,MLP_REGRESSION,GCN,GAT,GATv2,TRANSFORMER,GINE]

GLOBALPARAMETERS = {
    'no_of_epochs': 200,
    'model_this_run' : 'MLP_MLE',
    'train_size': 0.75,
    'valid_size': 0.05,
}


MLP_HYPERPARAMETERS = {
    'has_edge_info' : False,
    "batch_size": 16,
    "learning_rate": 0.0002007339899573065,
    "model_layers": 2,
    "model_embedding_size" : 256,
    "model_dense_neurons" : 256,
    "model_linear_dropout_rate" : 0.052950804012409605,
    "model_embedding_dropout_rate" : 0.1591852948412082

}

MLP_REGRESSION_HYPERPARAMETERS = {
    'has_edge_info' : False,
    "batch_size": 16,
    "learning_rate": 0.000933,
    "model_layers": 2,
    "model_embedding_size" : 256,
    "model_dense_neurons" : 64,
    "model_linear_dropout_rate" : 0.0598,
    "model_embedding_dropout_rate" : 0.0114

}

MLP_MLE_HYPERPARAMETERS = {
    'has_edge_info' : False,
    "batch_size": 64,
    "learning_rate": 0.000138,
    "model_layers": 3,
    "model_embedding_size" : 128,
    "model_dense_neurons" : 64,
    "model_linear_dropout_rate" : 0.078,
    "model_embedding_dropout_rate" : 0.119
}

MLP_MDN_HYPERPARAMETERS = {
    'has_edge_info' : False,
    "batch_size": 32,
    "learning_rate": 0.0000413,
    "model_layers": 3,
    "model_embedding_size" : 64,
    "model_dense_neurons" : 256,
    "model_linear_dropout_rate" : 0.181,
    "model_embedding_dropout_rate" : 0.193,
    'no_of_gaussians' : 3
}

MLP_DE_HYPERPARAMETERS = {
    'has_edge_info' : False,
    "batch_size": 16,
    "learning_rate": 0.000933,
    "model_layers": 2,
    "model_embedding_size" : 256,
    "model_dense_neurons" : 64,
    "model_linear_dropout_rate" : 0.0598,
    "model_embedding_dropout_rate" : 0.0114,
    'no_of_models' : 5
}

MLP_QR_HYPERPARAMETERS = {
    'has_edge_info' : False,
    "batch_size": 16,
    "learning_rate": 0.0000415,
    "model_layers": 4,
    "model_embedding_size" : 64,
    "model_dense_neurons" : 256,
    "model_linear_dropout_rate" : 0.0805,
    "model_embedding_dropout_rate" : 0.0144,
}

MLP_MCDO_HYPERPARAMETERS = {
    'has_edge_info' : False,
    "batch_size": 16,
    "learning_rate": 0.0002007339899573065,
    "model_layers": 2,
    "model_embedding_size" : 256,
    "model_dense_neurons" : 256,
    "model_linear_dropout_rate" : 0.05,
    "model_embedding_dropout_rate" : 0.05

}