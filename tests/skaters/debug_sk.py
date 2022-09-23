import numpy as np

def using_sk():
    from timemachines.skaters.sk.skinclusion import using_sktime
    print(using_sktime)

def using_sk_autoarima():
    from timemachines.skaters.sk.skinclusion import using_sktime
    print(using_sktime)


def debug_sk_autoarima():
    from timemachines.skaters.sk.skautoarima import sk_autoarima as f
    ys = np.random.randn(50,1)
    s = {}
    for y in ys:
        x, x_std, s = f(y=y, k=1, s=s)
        print(x)


if __name__=='__main__':
    using_sk()
    using_sk_autoarima()
    debug_sk_autoarima()