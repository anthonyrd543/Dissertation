from torch_geometric.datasets import MoleculeNet
import torch
import numpy as np
import config
from torch_geometric.data import DataLoader
import optuna

import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

import sys
sys.path.insert(1, '/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks')

import models

no_of_epochs = config.GLOBALPARAMETERS['no_of_epochs']
train_size = config.GLOBALPARAMETERS['train_size']
valid_size = config.GLOBALPARAMETERS['valid_size']
early_stopping = config.GLOBALPARAMETERS['early_stopping']
n_iters = config.GLOBALPARAMETERS['n_iters']

class FingerprintDataset(Dataset):
    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt('/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks/fingerprints/bace_fingerprints_classification.csv', delimiter=',', dtype=np.float32, skiprows=1)

        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,[0]])
        self.n_samples = xy.shape[0]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

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
        for i, (fingerprint, labels) in enumerate(loader):
            self.optimizer.zero_grad()  
            fingerprint = fingerprint.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(fingerprint)
            loss = self.loss_fn(outputs, labels)
            # Backward and optimize
            final_loss += loss.item()
            loss.backward()
            self.optimizer.step() 
        return final_loss / len(loader)
    
    def evaluate(self, loader):
            self.model.eval()
            # Enumerate over the data
            final_loss = 0
            for i, (fingerprint, labels) in enumerate(loader):  
                fingerprint = fingerprint.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(fingerprint)
                loss = self.loss_fn(outputs, labels)
                # Backward and optimize
                final_loss += loss.item()
            return final_loss / len(loader)

def run_training(params, save_model = False):
    data = FingerprintDataset()

    #Specify device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
   
    model_params = params
    model_params['feature_size'] = data.x.shape[1]
    model_params["has_edge_info"] = False
    
    model = models.MLP(model_params)
    print(model)

    model = model.to(device)

    train = int(train_size * len(data))
    valid = int(valid_size * len(data))
    test = len(data) - train - valid
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(data, [train, valid, test])

    NUM_FINGERPRINTS_PER_BATCH = model_params['batch_size']
    train_loader = DataLoader(train_dataset, 
                        batch_size=NUM_FINGERPRINTS_PER_BATCH, shuffle=True)
    valid_loader = DataLoader(valid_dataset, 
                        batch_size=NUM_FINGERPRINTS_PER_BATCH, shuffle=True)
    test_loader = DataLoader(test_dataset, 
                        batch_size=NUM_FINGERPRINTS_PER_BATCH, shuffle=True)

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
                torch.save(model.state_dict(), '/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks/hyperparameter_files/saved_models/mlp_model.pt')
        else:
            early_stopping_counter +=1
        if early_stopping_counter > early_stopping_iter:
            break
    return best_loss


def objective(trial):
    params = {
        "model_layers": trial.suggest_int("model_layers", 1,7),
        "batch_size": trial.suggest_categorical('batch_size', [64,32,128,16]),
        "learning_rate": trial.suggest_loguniform('learning_rate', 1e-6,1e-3),
        "model_embedding_size": trial.suggest_categorical("model_embedding_size",[32, 64, 128, 256,512,1024,2048]),
        "model_linear_dropout_rate": trial.suggest_uniform('model_linear_dropout_rate', 0.01, 0.2),
        "model_embedding_dropout_rate": trial.suggest_uniform('model_embedding_dropout_rate', 0.01, 0.2),
        "model_dense_neurons": trial.suggest_categorical("model_dense_neurons",[32, 64, 128, 256])
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
    import pandas as pd

    best_parameters = pd.DataFrame([trial_.params])
    best_parameters["Best Loss"] = trial_.values
    best_parameters.to_csv('/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks/hyperparameter_files/best_parameters/mlp_best_parameters.csv')