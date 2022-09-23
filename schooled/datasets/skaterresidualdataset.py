from torch.utils.data.dataset import Dataset
import numpy as np
from schooled.datasets.skaterresiduals import memoized_residuals

BATCH_SIZE = 3


class ResidualDataset(Dataset):
    """
        Serves examples of lagged values and next residual value
    """

    def __init__(self, seq_len=100, start_index=0, end_index=-1 ):
        self._x = memoized_residuals(seq_len=seq_len)[start_index:end_index]
        self._seq_len = seq_len
        self._len = np.shape(self._x)[0]
        super(Dataset).__init__()

    def __getitem__(self,ndx):
        x1 = self._x[ndx,:-1].astype(np.float32)
        x = np.atleast_2d(x1).transpose()
        y = np.array(self._x[ndx][-1]).astype(np.float32)
        return x,y

    def __len__(self):
        return self._len


if __name__=='__main__':
    test_dataset = ResidualDataset(end_index=1000)
    X, y = next(iter(test_dataset))
    print(np.shape(X))
    print(y)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    X, y = next(iter(train_loader))

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)