import random
SKATER_RESIDUAL_URL = 'https://raw.githubusercontent.com/microprediction/precisedata/main/skaterresiduals/skater_residuals_0.csv'
n_data = 450
import pandas as pd
import numpy as np
from schooled.datasets.util import TimeseriesDataset


def random_skater_residual_dataframe(min_obs:int):
    """ Pull a dataframe of model residuals
    :param min_obs: Retry until we get at least min_obs
    :return:
    """
    got = False
    while not got:
        the_choice = random.choice(list(range(n_data)))
        the_url = SKATER_RESIDUAL_URL.replace('N', str(the_choice))
        try:
            df = pd.read_csv(the_url)
            del df['Unnamed: 0']
            got = len(df.index) > min_obs + 10
        except:
            got = False
    return df


def concatenated_skater_residuals(min_obs) -> [float]:
    """ Concatenate model residuals from different models """
    df = random_skater_residual_dataframe(min_obs=min_obs)
    n_elts = np.prod(df.shape)
    return df.values.reshape((n_elts,1),order="F").squeeze()




def show_some_iterations():
    train_dataset = TimeseriesDataset(X_lstm, y_lstm, seq_len=4)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3, shuffle=False)

    for i, d in enumerate(train_loader):
        print(i, d[0].shape, d[1].shape)


def plot_some_residuals():
    df = random_skater_residual_dataframe(min_obs=20)
    print(df[:2].transpose())
    v = concatenated_skater_residuals(min_obs=1000)
    import matplotlib.pyplot as plt
    for k in range(5000,15000,1000):
        plt.plot(v[k:k+20])
    plt.grid()
    plt.show()

