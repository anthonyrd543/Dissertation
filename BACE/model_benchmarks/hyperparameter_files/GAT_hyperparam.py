from torch_geometric.datasets import MoleculeNet
import torch
import numpy as np
import config
from torch_geometric.data import DataLoader
import optuna
import pandas as pd

import sys
sys.path.insert(1, '/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks')

import models

no_of_epochs = config.GLOBALPARAMETERS['no_of_epochs']
train_size = config.GLOBALPARAMETERS['train_size']
valid_size = train_size + config.GLOBALPARAMETERS['valid_size']
early_stopping = config.GLOBALPARAMETERS['early_stopping']
n_iters = config.GLOBALPARAMETERS['n_iters']

class Engine:
    def __init__(self, model, model_params, optimizer, device):
        self.model = model
        self.model_params = model_params
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = torch.nn.BCELoss()

    def train(self, loader):
        self.model.train()
        # Enumerate over the data
        final_loss = 0
        for batch in loader:
            # Use GPU
            batch.to(self.device)  
            # Reset gradients
            self.optimizer.zero_grad() 
            # Passing the node features and the connection info
            if self.model_params['has_edge_info']:
                pred = self.model(batch.x.float(),
                                batch.edge_attr.float(), 
                                batch.edge_index, 
                                batch.batch)
            else:
                pred = self.model(batch.x.float(), 
                                batch.edge_index, 
                                batch.batch)
            # Calculating the loss and gradients
            loss = self.loss_fn(pred, batch.y)
            final_loss += loss.item()
            loss.backward()
            # Update using the gradients
            self.optimizer.step()   
        return final_loss / len(loader)
    
    def evaluate(self, data_loader):
            self.model.eval()
            final_loss = 0
            for batch in data_loader:
                batch.to(self.device)
                if self.model_params['has_edge_info']:
                    pred = self.model(batch.x.float(),
                                    batch.edge_attr.float(), 
                                    batch.edge_index, 
                                    batch.batch)
                else:
                    pred = self.model(batch.x.float(), 
                                    batch.edge_index, 
                                    batch.batch)
                loss = self.loss_fn(pred, batch.y)  
                final_loss += loss.item()
            return final_loss / len(data_loader)

def run_training(params, save_model = False):
    data = MoleculeNet(root=".", name="bace")
    data = data.shuffle()
    #Specify device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    
    model_params = params
    model_params['feature_size'] = data.num_features
    model_params["edge_dim"] = data[0].edge_attr.shape[1]
    model_params["has_edge_info"] = True
    
    model = models.GAT(model_params)
    print(model)

    model = model.to(device)

    # Wrap data in a data loader
    data_size = len(data)
    NUM_GRAPHS_PER_BATCH = model_params['batch_size']
    train_loader = DataLoader(data[:int(data_size * train_size)], 
                        batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
    valid_loader = DataLoader(data[int(data_size * train_size):int(data_size * (valid_size))], 
                            batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
    test_loader = DataLoader(data[int(data_size * (valid_size)):], 
                            batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_params['learning_rate'])  

    eng = Engine(model, model_params, optimizer, device)

    best_loss = np.inf
    early_stopping_iter = early_stopping
    early_stopping_counter = 0

    print("Starting training...")
    losses = []
    for epoch in range(no_of_epochs):
        loss = eng.train(train_loader)
        val_loss = eng.evaluate(valid_loader)
        losses.append(loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train Loss {loss} | Valid Loss {val_loss}")
        if val_loss < best_loss:
            best_loss = val_loss
            if save_model:
                torch.save(model.state_dict(), '/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks/hyperparameter_files/saved_models/gat_model.pt')
        else:
            early_stopping_counter +=1
        if early_stopping_counter > early_stopping_iter:
            break
    return best_loss


def objective(trial):
    params = {
        "model_layers": trial.suggest_int("model_layers", 1,7),
        "batch_size": trial.suggest_categorical('batch_size', [64,32,128]),
        "learning_rate": trial.suggest_loguniform('learning_rate', 1e-6,1e-3),
        "model_embedding_size": trial.suggest_categorical("model_embedding_size",[32, 64, 128, 256, 512, 1024]),
        "model_gnn_dropout_rate": trial.suggest_uniform('model_gnn_dropout_rate', 0.01, 0.2),
        "model_linear_dropout_rate": trial.suggest_uniform('model_linear_dropout_rate', 0.01, 0.2),
        "model_dense_neurons": trial.suggest_categorical("model_dense_neurons",[32, 64, 128, 256]),
        "no_of_heads": trial.suggest_int("no_of_heads",1,4),
        "gat_dropout_rate": trial.suggest_uniform('gat_dropout_rate', 0.01, 0.2),
    }

    best_loss = run_training(params, save_model=False)
    return best_loss

if __name__ == '__main__':
    study = optuna.create_study(direction = 'minimize')
    study.optimize(objective, n_trials=n_iters)

    print('best_trial')
    trial_ = study.best_trial

    print(trial_.values)
    print(trial_.params)

    scores = run_training(trial_.params,save_model=True)

    best_parameters = pd.DataFrame([trial_.params])
    best_parameters["Best Loss"] = trial_.values
    best_parameters.to_csv('/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks/hyperparameter_files/best_parameters/gat_best_parameters.csv')