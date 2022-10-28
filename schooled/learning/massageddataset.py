from torch.utils.data.dataset import Dataset
import numpy as np
from schooled.generating.step_2_collate import load_massaged
from schooled.generating.step_1_generate import SEQ_LEN, FS_SHORT_NAMES, FS_ERROR_NAMES

# Serves a dataset class

TRAIN_SEQ_LEN = 10
COL_TO_LEARN = FS_SHORT_NAMES[1] # auto-arima, auto-arima wiggly, tsa p2, tsa p1
COL_TO_LEARN = 'y_next'
BATCH_SIZE=1000


def load_regressors_and_target(train_seq_len=TRAIN_SEQ_LEN, col_to_learn=COL_TO_LEARN):
    assert train_seq_len <= SEQ_LEN
    df = load_massaged()
    df = df.dropna(how='any')
    xy_cols = (['y_'+str(i) for i in range(SEQ_LEN-1)] + ['y_last'])[-train_seq_len:] + [col_to_learn]
    for col in xy_cols:
        if col not in df.columns:
            print(col)
            raise Exception('urgh')
    XY = df[xy_cols].values
    return XY


class SarimaDataset(Dataset):
    """
        Serves examples of lagged values and next predicted value
    """

    def __init__(self, train_seq_len=TRAIN_SEQ_LEN, start_index=0, end_index=-1):
        self._xy = load_regressors_and_target(train_seq_len=train_seq_len)[start_index:end_index]
        self._seq_len = np.shape(self._xy)[1]-1
        self._len = np.shape(self._xy)[0]
        super(Dataset).__init__()

    def __getitem__(self,ndx):
        x1 = self._xy[ndx,:-1].astype(np.float32)
        x = x1
        y = np.array([self._xy[ndx][-1]]).astype(np.float32)
        return x,y

    def __len__(self):
        return self._len


if __name__=='__main__':
    test_dataset = SarimaDataset(end_index=1000)
    X, y = next(iter(test_dataset))
    print(np.shape(X))
    print(y)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    X, y = next(iter(train_loader))

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

