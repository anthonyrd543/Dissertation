from torch_geometric.datasets import MoleculeNet
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
import models

train_data = MoleculeNet(root="/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks/split_bace/train", name="bace")
test_data = MoleculeNet(root="/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks/split_bace/test", name="bace")

#Specify device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

import best_config
from torch_geometric.data import DataLoader
#Get Model Parameters
NUM_GRAPHS_PER_BATCH = 64
train_loader = DataLoader(train_data, 
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(test_data, 
                         batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

def train(model, optimizer, loader):
    model.train()
    # Enumerate over the data
    final_loss = 0
    for batch in loader:
      # Use GPU
      batch.to(device)  
      # Reset gradients
      optimizer.zero_grad() 
      # Passing the node features and the connection info
      if model_params['has_edge_info']:
          pred = model(batch.x.float(),
                        batch.edge_attr.float(), 
                        batch.edge_index, 
                        batch.batch)
      else:
          pred = model(batch.x.float(), 
                        batch.edge_index, 
                        batch.batch)
      # Calculating the loss and gradients
      loss = loss_fn(pred, batch.y)
      final_loss += loss.item()
      loss.backward()
      # Update using the gradients
      optimizer.step()   
    return final_loss / len(loader)

def get_results(binary_results):
    accuracy = metrics.accuracy_score(binary_results["y_real"],binary_results["y_pred"])
    f1 = metrics.f1_score(binary_results["y_real"],binary_results["y_pred"])
    precision = metrics.precision_score(binary_results["y_real"],binary_results["y_pred"])
    recall = metrics.recall_score(binary_results["y_real"],binary_results["y_pred"])
    roc_auc = metrics.roc_auc_score(binary_results["y_real"],binary_results["y_pred"])
    tn, fp, fn, tp = metrics.confusion_matrix(binary_results["y_real"],binary_results["y_pred"], labels=[0,1]).ravel()
    specificity = tn / (tn+fp)

    results = {
        'Accuracy':accuracy,
        'F1': f1,
        'Precision': precision,
        'Recall': recall,
        'ROCAUC': roc_auc,
        'Specificity': specificity,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        "TP": tp,
    }
    return results

def test(model, test_loader):
    model.eval()
    true_values = []
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            batch.to(device)
            if model_params['has_edge_info']:
                pred = model(batch.x.float(),
                                batch.edge_attr.float(), 
                                batch.edge_index, 
                                batch.batch)
            else:
                pred = model(batch.x.float(), 
                                batch.edge_index, 
                                batch.batch)
            true_values += batch.y.tolist()
            predictions += pred.tolist()
    df = pd.DataFrame({'y_pred':predictions,'y_real':true_values})
    df = df.applymap(lambda x : x[0])
    output = df["y_pred"].apply(lambda x: int(round(x,0)))
    output
    binary_results = df.applymap(lambda x : int(round(x,0)))
    binary_results
    return df, binary_results, output, get_results(binary_results)

from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

models_to_test = ['GCN']

for model in models_to_test:

    model_this_run = model


    # Root mean squared error
    loss_fn = torch.nn.BCELoss()



    hyperparameters = eval('best_config.'+model_this_run+'_HYPERPARAMETERS')
    model_params = hyperparameters
    model_params['feature_size'] = train_data.num_features
    model_params['no_of_heads'] = 3
    model_params["edge_dim"] = train_data[0].edge_attr.shape[1]



    summary_columns = {
        'Accuracy':[],
        'F1': [],
        'Precision': [],
        'Recall': [],
        'ROCAUC': [],
        'Specificity': [],
        'TN': [],
        'FP': [],
        'FN': [],
        "TP": [],
    }
    summary = pd.DataFrame(summary_columns)

    best_acc = 0

    filepath = '/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks/'


    for i in range(10):
        torch.manual_seed(i)
        print(f"Starting training run {i}")
        #Create instance of model
        model = eval('models.'+model_this_run+'(model_params)')
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_params['learning_rate'])
        for epoch in range(300):
            loss = train(model, optimizer, train_loader)
        _, _, _, results = test(model, test_loader)
        if results['Accuracy'] > best_acc:
            best_acc = results['Accuracy']
            torch.save(model.state_dict(), filepath+'best_models/'+model_this_run+'_model.pt')
        results_this_run = pd.DataFrame([results])
        summary = summary.append(results_this_run)


    summary.to_csv(filepath + 'results/'+model_this_run+'summary.csv')
    summary
            

    model = eval('models.'+model_this_run+'(model_params)')
    model.load_state_dict(torch.load(f'/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks/best_models/{model_this_run}_model.pt'))
    model = model.to(device)

    df, binary_results, output, results = test(model, test_loader)

    print(model)

    import matplotlib.pyplot as plt
    labels = [0,1]
    cm = metrics.confusion_matrix(output,df["y_real"])
    metrics.ConfusionMatrixDisplay.from_predictions(binary_results["y_real"],binary_results["y_pred"], cmap='Blues')

    filepath = '/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks/results/'
    plt.savefig(filepath + model_this_run + 'confusion_matrix.jpeg', bbox_inches='tight',dpi=100)

    r_fpr, r_tpr, thresholds = metrics.roc_curve(df["y_real"].to_list(),df["y_pred"].to_list())

    roc_curve_data = pd.DataFrame({'r_fpr': r_fpr, 'r_tpr': r_tpr, 'thresholds': thresholds})

    filepath = '/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks/results/'
    roc_curve_data.to_csv(filepath + model_this_run+'rocdata.csv')

    import matplotlib.pyplot as plt

    plt.plot(r_fpr, r_tpr)
    # Title
    plt.title('ROC Plot')
    # Axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    filepath = '/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks/results/'
    plt.savefig(filepath + model_this_run + 'roc_curve.jpeg', bbox_inches='tight', dpi=100)

    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(df["y_real"].to_list(),df["y_pred"].to_list())
    metrics.PrecisionRecallDisplay.from_predictions(df["y_real"].to_list(),df["y_pred"].to_list())

    plt.title('PR Curve')

    filepath = '/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks/results/'
    plt.savefig(filepath + model_this_run + 'pr_curve.jpeg', bbox_inches='tight', dpi=100)