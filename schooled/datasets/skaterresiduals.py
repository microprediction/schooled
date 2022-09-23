import random
SKATER_RESIDUAL_URL = 'https://raw.githubusercontent.com/microprediction/precisedata/main/skaterresiduals/skater_residuals_0.csv'
n_data = 450
import pandas as pd
from functools import lru_cache
import math
import numpy as np


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



def plot_some_residuals():
    df = random_skater_residual_dataframe(min_obs=20)
    print(df[:2].transpose())
    v = concatenated_skater_residuals(min_obs=1000)
    import matplotlib.pyplot as plt
    for k in range(5000,15000,1000):
        plt.plot(v[k:k+20])
    plt.grid()
    plt.show()



# Serves data at precisedata repo in a torch dataset format

@lru_cache()
def memoized_residuals(seq_len=100):
    r = concatenated_skater_residuals(min_obs=500)
    n = int(math.floor(len(r)/(seq_len+1)))
    m = n*(seq_len+1)
    x = np.reshape(r[:m], newshape=(n,seq_len+1))
    return x

