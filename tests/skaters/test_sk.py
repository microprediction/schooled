import numpy as np


def test_using_sk():
    from timemachines.skaters.sk.skinclusion import using_sktime
    assert using_sktime,'pip install sktime'


def test_using_sk_autoarima():
    from timemachines.skaters.sk.skautoarima import sk_autoarima


def test_sk_autoarima():
    from timemachines.skaters.sk.skautoarima import sk_autoarima as f
    ys = np.random.randn(50,1)
    s = {}
    for y in ys:
        x, x_std, s = f(y=y, k=1, s=s)
        print(x)

