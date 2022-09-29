

import nni
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from schooled.datasets.sarimadataset import SarimaDataset, BATCH_SIZE

from schooled.datasets.sarimadataset import SEQ_LEN

# Hyperparameters to be tuned
# ---------------------------
# These are the hyperparameters that will be tuned.
params = {
    'num_layers':1,
    'num_1':1,
    'num_2':16,
    'num_3':16,
    'act_0':'Sigmoid',
    'act_1': 'ReLU',
    'act_2': 'ReLU',
    'lr': 0.001,
    'momentum': 0,
}

 
# Get optimized hyperparameters
# -----------------------------
# If run directly, :func:`nni.get_next_parameter` is a no-op and returns an empty dict.
# But with an NNI *experiment*, it will receive optimized hyperparameters from tuning algorithm.
optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

 
# Load dataset
# ------------

training_data = SarimaDataset(end_index=50000)
test_data = SarimaDataset(start_index=10000)

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

 
# Build model with hyperparameters
# --------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# If your input data is of shape (seq_len, batch_size, features) then you don’t need batch_first=True and your LSTM will give output of shape (seq_len, batch_size, hidden_size).
# If your input data is of shape (batch_size, seq_len, features) then you need batch_first=True and your LSTM will give output of shape (batch_size, seq_len, hidden_size).


class ShallowRegression(nn.Module):
    # https://www.crosstab.io/articles/time-series-pytorch-lstm


    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        if params['num_layers']==0:
            self.stack = nn.Sequential(
                nn.Linear(in_features=SEQ_LEN, out_features=1)
            )
        elif params['num_layers']==1:
            self.stack = nn.Sequential(
                nn.Linear(in_features=SEQ_LEN, out_features=params['num_1']),
                getattr(nn, params['act_0'])(),
                nn.Linear(in_features=params['num_1'], out_features=1)
            )
        elif params['num_layers']==2:
            self.stack = nn.Sequential(
                nn.Linear(in_features=SEQ_LEN, out_features=params['num_1']),
                getattr(nn, params['act_0'])(),
                nn.Linear(in_features=params['num_1'], out_features=params['num_2']),
                getattr(nn, params['act_1'])(),
                nn.Linear(in_features=params['num_2'], out_features=1)
            )
        elif params['num_layers']==3:
            self.stack = nn.Sequential(
                nn.Linear(in_features=SEQ_LEN,  out_features=params['num_1']),
                getattr(nn, params['act_0'])(),
                nn.Linear(in_features=params['num_1'], out_features=params['num_2']),
                getattr(nn, params['act_1'])(),
                nn.Linear(in_features=params['num_2'], out_features=params['num_3']),
                getattr(nn, params['act_2'])(),
                nn.Linear(in_features=params['num_3'], out_features=1)
            )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits

model = ShallowRegression().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])

 
# Define train and test
# ---------------------

def train(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")


def verify(data_loader, model, loss_function):

    num_batches = len(data_loader)
    total_loss = 0
    total_last_value_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            last_value = np.atleast_2d( X[:,-1]).transpose()
            last_value_output = torch.from_numpy( last_value )
            loss_value = loss_function(output, y).item()
            last_value_loss_value = loss_function( last_value_output, y).item()

            if np.isfinite(loss_value):
                total_loss += loss_value
                total_last_value_loss += last_value_loss_value
            else:
                print(loss_value)
                raise Exception('loss is not finite')

    rel_loss = total_loss / total_last_value_loss
    print(f"Rel loss: {rel_loss}")
    return rel_loss



 
# Train model and report accuracy
# -------------------------------
# Report accuracy metrics to NNI so the tuning algorithm can suggest better hyperparameters.
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    accuracy = verify(test_dataloader, model, loss_fn)
    nni.report_intermediate_result(accuracy)
nni.report_final_result(accuracy)