from torch_geometric.datasets import MoleculeNet
import torch
import numpy as np
import best_config
from torch_geometric.data import DataLoader
import optuna

import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

import sys
sys.path.insert(1, '/home/rajeckidoyle/Documents/Classification/BACE_Classification/regressionandunccertainty')

import models

no_of_epochs = best_config.GLOBALPARAMETERS['no_of_epochs']
train_size = best_config.GLOBALPARAMETERS['train_size']
valid_size = train_size + best_config.GLOBALPARAMETERS['valid_size']


class FingerprintDataset(Dataset):
    def __init__(self,split):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt(f'./fingerprints/bace_reverse_split_{split}.csv', delimiter=',', dtype=np.float32, skiprows=1)

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
        self.loss_fn = torch.nn.MSELoss()

    def train(self, loader):
        self.model.train()
        # Enumerate over the data
        final_loss = 0
        for j, (fingerprint, labels) in enumerate(loader):
            self.optimizer.zero_grad()
            fingerprint = fingerprint.to(self.device)
            labels = labels.to(self.device)
            preds = self.model(fingerprint)
            loss = self.loss_fn(preds, labels)
            final_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        final_loss = final_loss / len(loader) 
        return final_loss / len(loader)
    
    def evaluate(self, loader):
            self.model.eval()
            # Enumerate over the data
            final_loss = 0
            for j, (fingerprint, labels) in enumerate(loader):
                fingerprint = fingerprint.to(self.device)
                labels = labels.to(self.device)
                preds = self.model(fingerprint)
                loss = self.loss_fn(preds, labels)
                final_loss += loss.item()
            final_loss = final_loss / len(loader)
            return final_loss / len(loader)

def run_training(params, save_model = False):
    data = FingerprintDataset()

    #Specify device
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
   
    model_params = params
    model_params['feature_size'] = data.x.shape[1]
    model_params["has_edge_info"] = False
    
    model = models.MLP_QR(model_params)
    print(model)

    model = model.to(device)

    train_dataset = FingerprintDataset(split='train')
    test_dataset = FingerprintDataset(split='test')
    

    optimizer = torch.optim.Adam(model.parameters(), lr=model_params['learning_rate'])  

    eng = Engine(model, model_params, optimizer, device)

    best_loss = np.inf

    NUM_FINGERPRINTS_PER_BATCH = model_params['batch_size']
    train_loader = DataLoader(train_dataset, 
                        batch_size=NUM_FINGERPRINTS_PER_BATCH, shuffle=True)
    test_loader = DataLoader(test_dataset, 
                        batch_size=NUM_FINGERPRINTS_PER_BATCH, shuffle=True)

    print("Starting training...")
    losses = []
    for epoch in range(no_of_epochs):
        loss = eng.train(train_loader)
        losses.append(loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train Loss {loss}")
        
    return loss

if __name__ == '__main__':
    params = best_config.MLP_QR_HYPERPARAMETERS
    run_training(params)
