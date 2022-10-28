
# %%
import nni
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from schooled.learning.massageddataset import SarimaDataset, BATCH_SIZE



# %%
# Hyperparameters to be tuned
# ---------------------------
# These are the hyperparameters that will be tuned.
params = {
    'num_hidden':16,
    'lr': 0.001,
    'momentum': 0,
}

# %%
# Get optimized hyperparameters
# -----------------------------
# If run directly, :func:`nni.get_next_parameter` is a no-op and returns an empty dict.
# But with an NNI *experiment*, it will receive optimized hyperparameters from tuning algorithm.
optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

# %%
# Load dataset
# ------------
training_data = SarimaDataset(end_index=4000)
test_data = SarimaDataset(start_index=4000)

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

# %%
# Build model with hyperparameters
# --------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# If your input data is of shape (seq_len, batch_size, features) then you don’t need batch_first=True and your LSTM will give output of shape (seq_len, batch_size, hidden_size).
# If your input data is of shape (batch_size, seq_len, features) then you need batch_first=True and your LSTM will give output of shape (batch_size, seq_len, hidden_size).

class ShallowRegressionLSTM(nn.Module):
    # https://www.crosstab.io/articles/time-series-pytorch-lstm

    def __init__(self):
        super().__init__()
        self.input_size = 1  # this is the number of features
        self.hidden_units = params['num_hidden']
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out

model = ShallowRegressionLSTM().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])

# %%
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
    total_zero_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            zero_output = torch.from_numpy( np.zeros_like(output) )
            loss_value = loss_function(output, y).item()
            zero_loss_value = loss_function( zero_output, y).item()

            if np.isfinite(loss_value):
                total_loss += loss_value
                total_zero_loss += zero_loss_value
            else:
                print(loss_value)
                raise Exception('loss is not finite')

    rel_loss = total_loss / total_zero_loss
    print(f"Rel loss: {rel_loss}")
    return rel_loss



# %%
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