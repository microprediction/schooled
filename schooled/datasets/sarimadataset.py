from schooled.wherami import SKATER_DATA
import numpy as np

SEQ_LEN = 20

def load_skater_surrogates(seq_len=SEQ_LEN):
    XYs = [ load_skater_surrogate(file_no=file_no,seq_len=seq_len) for file_no in range(1,3) ]
    XY = np.concatenate(XYs, axis=0)
    return XY


def load_skater_surrogate(file_no,seq_len):
    csv = SKATER_DATA + '/train_' + str(file_no) + '.csv'
    XY = np.loadtxt(csv, delimiter=',')
    XY = XY[:,-seq_len-1:]
    return XY

from torch.utils.data.dataset import Dataset
import numpy as np

BATCH_SIZE = 3


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
