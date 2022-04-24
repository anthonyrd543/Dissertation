from asyncio import LifoQueue
import torch
import torch.nn.functional as F 
from torch.nn import Dropout, Linear, BatchNorm1d, ModuleList, Sequential, ReLU

class MLP_REGRESSION(torch.nn.Module):
    def __init__(self, model_params):
        super(MLP_REGRESSION, self).__init__()
        feature_size = model_params["feature_size"]
        self.n_layers = model_params["model_layers"]
        dense_neurons = model_params["model_embedding_size"]
        dense_neurons_final = model_params["model_dense_neurons"]

        embedding_dropout_rate = model_params["model_embedding_dropout_rate"]
        linear_dropout_rate = model_params["model_linear_dropout_rate"]
        
        self.conv_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        self.initial_conv = Linear(feature_size, dense_neurons)
        self.embedding_dropout = Dropout(embedding_dropout_rate)
        self.linear_dropout = Dropout(linear_dropout_rate)

        for i in range(self.n_layers-1):
            self.conv_layers.append(Linear(dense_neurons, 
                                            dense_neurons))
            self.bn_layers.append(BatchNorm1d(dense_neurons))
        
        self.linear = Linear(dense_neurons, dense_neurons_final)
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
        x = torch.relu(x)
        x = self.linear_dropout(x)

        x = self.out(x)
        return x


'''
----------------------------------------------------------------------------------------
'''

class MLP_MLE(torch.nn.Module):
    def __init__(self, model_params):
        super(MLP_MLE, self).__init__()
        feature_size = model_params["feature_size"]
        self.n_layers = model_params["model_layers"]
        dense_neurons = model_params["model_embedding_size"]
        dense_neurons_final = model_params["model_dense_neurons"]

        embedding_dropout_rate = model_params["model_embedding_dropout_rate"]
        linear_dropout_rate = model_params["model_linear_dropout_rate"]

        self.conv_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        self.initial_conv = Linear(feature_size, dense_neurons)
        self.embedding_dropout = Dropout(embedding_dropout_rate)
        self.linear_dropout = Dropout(linear_dropout_rate)

        for i in range(self.n_layers-1):
            self.conv_layers.append(Linear(dense_neurons, 
                                            dense_neurons))
            self.bn_layers.append(BatchNorm1d(dense_neurons))
        
        self.linear_dropout = Dropout(linear_dropout_rate)

        self.readout1 = Linear(dense_neurons,dense_neurons_final)
        self.mu = Linear(dense_neurons_final, 1)

        self.readout2 = Linear(dense_neurons,dense_neurons_final)
        self.var = Linear(dense_neurons_final, 1)

    def forward(self, x):
        
        x = self.initial_conv(x)
        x = torch.relu(x)
        x = self.embedding_dropout(x)

        for i in range(self.n_layers-1):
            x = self.conv_layers[i](x)
            x = torch.relu(x)
            x = self.linear_dropout(x)
            x = self.bn_layers[i](x)

        x_mu = self.readout1(x)
        x_mu = torch.relu(x_mu)
        x_mu = self.linear_dropout(x_mu)

        mu = self.mu(x_mu)

        x_var = self.readout2(x)
        x_var = torch.relu(x_var)
        x_var = self.linear_dropout(x_var)

        # Exponential activation to enforce positive var
        var = torch.exp(self.var(x_var))
        return mu, var

'''
----------------------------------------------------------------------------------------
'''

class MLP_MDN(torch.nn.Module):
    def __init__(self, model_params):
        super(MLP_MDN, self).__init__()
        feature_size = model_params["feature_size"]
        self.n_layers = model_params["model_layers"]
        dense_neurons = model_params["model_embedding_size"]
        dense_neurons_final = model_params["model_dense_neurons"]
        self.num_gaussians = model_params["no_of_gaussians"]

        embedding_dropout_rate = model_params["model_embedding_dropout_rate"]
        linear_dropout_rate = model_params["model_linear_dropout_rate"]

        self.conv_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        self.initial_conv = Linear(feature_size, dense_neurons)
        self.embedding_dropout = Dropout(embedding_dropout_rate)
        self.linear_dropout = Dropout(linear_dropout_rate)

        for i in range(self.n_layers-1):
            self.conv_layers.append(Linear(dense_neurons, 
                                            dense_neurons))
            self.bn_layers.append(BatchNorm1d(dense_neurons))


        self.readout1 = Linear(dense_neurons, dense_neurons_final)
        self.mu = Linear(dense_neurons_final, self.num_gaussians)

        self.readout2 = Linear(dense_neurons, dense_neurons_final)
        self.sigma = Linear(dense_neurons_final, self.num_gaussians)
        
        self.readout3 = Linear(dense_neurons, dense_neurons_final)
        self.alpha = Linear(dense_neurons_final, self.num_gaussians)

    def forward(self, x):
        x = self.initial_conv(x)
        x = torch.relu(x)
        x = self.embedding_dropout(x)

        for i in range(self.n_layers-1):
            x = self.conv_layers[i](x)
            x = torch.relu(x)
            x = self.linear_dropout(x)
            x = self.bn_layers[i](x)

        x_mu = self.readout1(x)
        x_mu = torch.relu(x_mu)
        x_mu = self.linear_dropout(x_mu)

        

        x_var = self.readout2(x)
        x_var = torch.relu(x_var)
        x_var = self.linear_dropout(x_var)

        x_alpha = self.readout3(x)
        x_alpha = torch.relu(x_alpha)
        x_alpha = self.linear_dropout(x_alpha)

        mus = self.mu(x_mu)

        # Exponential activation to enforce positive var
        sigmas = torch.exp(self.sigma(x_var))

        # Softmax activation to enforce probabilities
        alphas = F.softmax(self.alpha(x_alpha), dim=1)

        return mus, sigmas, alphas

'''
----------------------------------------------------------------------------------------
'''


class MLP_DE(torch.nn.Module):
    def __init__(self, model_params):
        super(MLP_DE, self).__init__()
        feature_size = model_params["feature_size"]
        self.n_layers = model_params["model_layers"]
        dense_neurons = model_params["model_embedding_size"]
        dense_neurons_final = model_params["model_dense_neurons"]

        embedding_dropout_rate = model_params["model_embedding_dropout_rate"]
        linear_dropout_rate = model_params["model_linear_dropout_rate"]

        self.conv_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        self.initial_conv = Linear(feature_size, dense_neurons)
        self.embedding_dropout = Dropout(embedding_dropout_rate)
        self.linear_dropout = Dropout(linear_dropout_rate)

        for i in range(self.n_layers-1):
            self.conv_layers.append(Linear(dense_neurons, 
                                            dense_neurons))
            self.bn_layers.append(BatchNorm1d(dense_neurons))
        

        self.readout1 = Linear(dense_neurons,dense_neurons_final)
        self.mu = Linear(dense_neurons_final, 1)

        self.readout2 = Linear(dense_neurons,dense_neurons_final)
        self.var = Linear(dense_neurons_final, 1)

    def forward(self, x):
        
        x = self.initial_conv(x)
        x = torch.relu(x)
        x = self.embedding_dropout(x)

        for i in range(self.n_layers-1):
            x = self.conv_layers[i](x)
            x = torch.relu(x)
            x = self.linear_dropout(x)
            x = self.bn_layers[i](x)

        x_mu = self.readout1(x)
        x_mu = torch.relu(x_mu)
        x_mu = self.linear_dropout(x_mu)

        mu = self.mu(x_mu)

        x_var = self.readout2(x)
        x_var = torch.relu(x_var)
        x_var = self.linear_dropout(x_var)

        # Exponential activation to enforce positive var
        var = torch.exp(self.var(x_var))
        return mu, var

'''
----------------------------------------------------------------------------------------
'''

class MLP_QR(torch.nn.Module):
    def __init__(self, model_params):
        super(MLP_QR, self).__init__()
        feature_size = model_params["feature_size"]
        self.n_layers = model_params["model_layers"]
        dense_neurons = model_params["model_embedding_size"]
        dense_neurons_final = model_params["model_dense_neurons"]

        embedding_dropout_rate = model_params["model_embedding_dropout_rate"]
        linear_dropout_rate = model_params["model_linear_dropout_rate"]

        self.conv_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        self.initial_conv = Linear(feature_size, dense_neurons)
        self.embedding_dropout = Dropout(embedding_dropout_rate)
        self.linear_dropout = Dropout(linear_dropout_rate)

        for i in range(self.n_layers-1):
            self.conv_layers.append(Linear(dense_neurons, 
                                            dense_neurons))
            self.bn_layers.append(BatchNorm1d(dense_neurons))

        self.readout1 = Linear(dense_neurons,dense_neurons_final)
        self.lower_quantile = Linear(dense_neurons_final, 1)

        self.readout2 = Linear(dense_neurons,dense_neurons_final)
        self.median = Linear(dense_neurons_final, 1)

        self.readout3 = Linear(dense_neurons,dense_neurons_final)
        self.upper_quantile = Linear(dense_neurons_final, 1)

    def forward(self, x):
        x = self.initial_conv(x)
        x = torch.relu(x)
        x = self.embedding_dropout(x)

        for i in range(self.n_layers-1):
            x = self.conv_layers[i](x)
            x = torch.relu(x)
            x = self.linear_dropout(x)
            x = self.bn_layers[i](x)
        
        lq = self.readout1(x)
        lq = self.lower_quantile(lq)

        med = self.readout2(x)
        med = self.median(med)

        uq = self.readout3(x)
        uq = self.upper_quantile(uq)

        return torch.cat([lq, med, uq], axis=1)