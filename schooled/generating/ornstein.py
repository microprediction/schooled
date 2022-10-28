import numpy as np
from timemachines.skatertools.utilities.suppression import no_stdout_stderr


def pull_towards_zero(xs, kappa=0.1):
    """ Take a sequence xs and apply mean reversion """
    dxs = np.diff(xs)
    xt = 0
    ou = [ xs[0] ]
    for dx in dxs:
        xt = xt + dx
        xt = xt - kappa*xt
        ou.append(xt)
    return np.array(ou)


def simulate_arima_like_path(seq_len, reverse=False):
    """  Just a way to generate a time-series that isn't completely trivial in structure
    :param seq_len:
    :param plot:
    :return:
    """
    import pandas as pd
    import statsmodels.api as sm

    # Load data
    y = sm.datasets.macrodata.load_pandas().data['cpi']
    for j in range(len(y)):
        y[j] += np.random.randn()

    y.index = pd.period_range('1959Q1', '2009Q3', freq='Q')

    # Create and fit a model once to generate data
    import random
    p = random.choice(range(2, 10))
    d = 1
    q = random.choice(range(0, p+1))
    scale = random.choice([2, 1, 0.5, -0.5, -1, -2])
    with no_stdout_stderr():
        mod = sm.tsa.SARIMAX(y, order=(p, d, q), trend='c')
        res = mod.fit()

    with no_stdout_stderr():
        sim = res.simulate(seq_len, anchor='end', repetitions=1)

    # Make sure the data doesn't go crazy in scale
    x = sim.reset_index()['cpi'].values.squeeze() * scale
    x = (x - x[-1]) / 10.0
    if reverse:
        ou = np.flip(pull_towards_zero(np.flip(x)))
    else:
        ou = pull_towards_zero(x)

    # Make ou a stupid series instead so that output of fitting is the same format
    y1 = pd.Series(data=ou, index=pd.period_range(start='1959Q1', freq='Q',periods=len(ou)))

    # Fit it again just to see
    with no_stdout_stderr():
        p_fit = int(p/2+0.5)
        d_fit = 1
        q_fit = int(q/2+0.5)
        mod1 = sm.tsa.SARIMAX(y1, order=(p_fit, d_fit, q_fit), trend='c')
        res1 = mod1.fit()
        prms = pd.DataFrame(res.params)
        prms[1] = res1.params
        prms.columns = ['simulated','fitted']
    print(prms)


    return ou


def show_example_ornstein():
    xs = np.cumsum(np.random.randn(500))
    y = pull_towards_zero(xs)
    import matplotlib.pyplot as plt
    t = list(range(len(y)))
    skip = 40
    plt.plot(t[skip:],xs[skip:],t[skip:],y[skip:])
    plt.grid()
    plt.legend(['raw','ou'])
    plt.show()


def show_example_arima_like(seq_len=500):
    y = simulate_arima_like_path(seq_len=seq_len)
    import matplotlib.pyplot as plt
    t = list(range(len(y)))
    skip = 40
    plt.plot(t[skip:],y[skip:])
    plt.grid()
    plt.legend(['raw','ou'])
    plt.show()


if __name__=='__main__':
    show_example_arima_like()



