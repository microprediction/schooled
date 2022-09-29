from schooled.wherami import SKATER_DATA
import numpy as np
import pandas as pd

SEQ_LEN = 8
SCALE = 0.1

def load_skater_surrogates(seq_len=SEQ_LEN):
    dfs = list()
    for i in range(10):
        url = 'https://raw.githubusercontent.com/microprediction/schooled/main/data/sarima/sk_autoarima/train_' + str(
            i) + '.csv'
        try:
            df = pd.read_csv(url, header=None)
            print(url)
            dfs.append(df)
        except Exception:
            pass
    df = pd.concat(dfs, axis=0)
    df = df.dropna(how='any')
    last_col = df.columns[-1]
    y = (df[last_col].values) * SCALE
    del df[last_col]
    X = df.values[:, -seq_len-1:-1] * SCALE
    Y = np.atleast_2d(y).transpose()
    XY = np.concatenate([X,Y],axis=1)
    return XY


def load_skater_surrogate(file_no,seq_len):
    csv = SKATER_DATA + '/train_' + str(file_no) + '.csv'
    XY = np.loadtxt(csv, delimiter=',')
    XY = XY[:,-seq_len-1:]
    return XY

from torch.utils.data.dataset import Dataset
import numpy as np

BATCH_SIZE = 200


class SarimaDataset(Dataset):
    """
        Serves examples of lagged values and next predicted value
    """

    def __init__(self, seq_len=100, start_index=0, end_index=-1 ):
        assert seq_len==100
        self._xy = load_skater_surrogates()[start_index:end_index]
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



if __name__=='__main__':
    XY = load_skater_surrogates()
    print(np.shape(XY))
    print(np.max(np.max(XY)))
