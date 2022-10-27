from copy import deepcopy
from schooled.datasets.ornstein import simulate_arima_like_path


def skater_bump(ys, f, num_points=51, ndx=-1, k=1):
    """ Sensitivity to last value  """
    # Pass e<0 for most of the data
    assert ndx<0
    import numpy as np
    y_bumped = [ ys[ndx] + 0.01 * i for i in np.linspace(-100,100,num_points)]

    # Roll forward to -ndx
    s = {}
    for y in ys[:ndx]:
        x, x_std, s = f(y=y, k=k, s=s, e=-1)

    # Process bumped observations
    x_final_values = list()
    for y_bump in y_bumped:
        s_ = deepcopy(s)
        # Process ndx, which might be the last
        e = 1000 if ndx==-1 else -1
        x, x_std, s_ = f(y=y_bump, k=k, s=s_, e=e)
        # Might need to process from ndx+1 to -2
        for y in ys[ndx:-2]:
            x, x_std, s_ = f(y=y, k=k, s=s_, e=-1)
        # Process the last with e=1000
        if ndx<-1:
            x, x_std, s_ = f(y=ys[-1], k=k, s=s_, e=1000)
        x_final_values.append(x[k-1])

    return y_bumped, x_final_values


def skater_bump_plot(f, ndx, k):
    for _ in range(200):
        import numpy as np
        ys = simulate_arima_like_path(seq_len=100)
        y_final, x_final = skater_bump(ys=ys, f=f, ndx=ndx, k=k)
        slope = (x_final[-1]-x_final[0])/(y_final[-1]-y_final[0])
        discont_max = np.max(np.diff(np.array(x_final)))
        discont_median = np.median(np.abs(np.diff(np.array(x_final))))
        if discont_max>10*discont_median:
            import matplotlib.pyplot as plt
            plt.plot(y_final,x_final,'rx')
            plt.ylabel('Prediction '+str(k)+' steps ahead')
            plt.xlabel('Value taken by y['+str(ndx)+']')
            plt.grid()
            plt.title(f.__name__)
            plt.show()
            import time
            time.sleep(2)
            plt.close()


if __name__=='__main__':
    from timemachines.skaters.sk.skautoarima import sk_autoarima as f
    from timemachines.skaters.proph.prophskaterssingular import fbprophet_univariate as f
    #from timemachines.skaters.simple.thinking import thinking_fast_and_slow as f
    #from timemachines.skaters.simple.movingaverage import precision_ema_ensemble as f
    from timemachines.skaters.tsa.tsatheta import tsa_theta_additive as f
    from timemachines.skaters.bats.allbatsskaters import bats_arma_bc as f
    from timemachines.skaters.tsa.tsaensembles import tsa_precision_combined_ensemble as f

    skater_bump_plot(f=f, ndx=-5, k=1)

