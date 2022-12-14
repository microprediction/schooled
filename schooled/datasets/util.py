import torch


class TimeseriesDataset(torch.utils.data.Dataset):
    # Ack: https://stackoverflow.com/users/1855792/eugene-tartakovsky
    def __init__(self, X, y, seq_len=1):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        return self.X[index:index + self.seq_len], self.y[index + self.seq_len - 1]