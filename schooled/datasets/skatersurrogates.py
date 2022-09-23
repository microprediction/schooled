import os

import pandas

from schooled.wherami import SKATER_DATA, SKATER
import pathlib
import os
import numpy as np
from timemachines.skatertools.utilities.suppression import no_stdout_stderr

NUM_ROWS = 10000
SEQ_LEN = 100
START_FILE_NO=0


def simulate_arima(seq_len):
    import pandas as pd
    import statsmodels.api as sm

    # Load data
    y = sm.datasets.macrodata.load_pandas().data['cpi']
    for j in range(len(y)):
        y[j] += np.random.randn()

    y.index = pd.period_range('1959Q1', '2009Q3', freq='Q')

    # Create and fit the model
    import random
    p = random.choice(range(2,20))
    d = random.choice(range(0,3))
    q = random.choice(range(0,3))
    scale = random.choice([2,1,0.5,-0.5,-1,-2])
    with no_stdout_stderr():
        mod = sm.tsa.SARIMAX(y, order=(p, d, q), trend='c')
        res = mod.fit()
        # Simulate data starting at the end of the time series
        sim = res.simulate(seq_len, anchor='end', repetitions=1)

    x = sim.reset_index()['cpi'].values.squeeze()*scale

    x = (x - x[-1])/10.0

    return x


def make_data():
    pathlib.Path(SKATER_DATA).mkdir(parents=True, exist_ok=True)
    from timemachines.skaters.sk.skautoarima import sk_autoarima as f
    for file_no in range(START_FILE_NO,100):
        csv = SKATER_DATA + '/train_' + str(file_no) + '.csv'
        print('Making '+ csv)
        data = list()
        for row_no in range(NUM_ROWS):
            okay = False
            while not okay:
                try:
                    ys = simulate_arima(seq_len=SEQ_LEN)
                    assert np.max(ys)<1000
                    assert np.min(ys)>-1000
                    okay = True
                except:
                    print('Failed')

            s = {}
            for y in ys[:-1]:
                x, x_std, s = f(y=y, k=1, s=s, e=-1 )
            y_final = ys[-1]
            x, x_std, s = f(y=y_final, k=1, s=s, e=1000)

            example = np.concatenate([ys, x] )
            last_few = example[-4:]
            print(last_few)

            data.append(example)
            if row_no % 10 ==0:
                print(str(row_no)+' of '+str(NUM_ROWS))
                X = np.array(data)
                np.savetxt(fname=csv, X=X, delimiter=',')


if __name__=='__main__':
    make_data()