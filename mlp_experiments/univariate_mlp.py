# -*- coding: utf-8 -*-
"""univariate_autosklearn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/microprediction/automl-notebooks/blob/main/univariate_autosklearn.ipynb
"""

from schooled.generating.step_2_collate import load_massaged
import sklearn.metrics
import numpy as np
import math



"""# Data to fit """
def surrogate_ratio(df):

    x_cols = [ c for c in df.columns if 'y_' in c ]
    assert 'x' not in x_cols
    y = df['p1'].values       # <--- Model we want surrogate for
    X = df[x_cols].values
    z = df['x'].values        # <--- The target (next value in the series)
    y1 = 1.00*y + 0.0*z        # <--- Train on combination of model and target, maybe

    X = X.astype(np.float32)
    n_test = int(len(df)/5)
    n_train = len(df)-2*n_test
    X_train, y_train, z_train = X[:n_train], y1[:n_train], z[:n_train]
    X_test, y_test, z_test = X[n_train:(n_train+n_test)], y1[n_train:(n_train+n_test)], z[n_train:(n_train+n_test)]
    X_val, y_val, z_val = X[(n_train+n_test):], y1[(n_train+n_test):], z[(n_train+n_test):]

    from sklearn.neural_network import MLPRegressor

    regr = MLPRegressor(random_state=1,
                        max_iter=50000,
                        activation='relu',
                        hidden_layer_sizes=(200, 150, 50)).fit(X_train, y_train)

    y_train_hat = regr.predict(X_train)
    y_test_hat = regr.predict(X_test)
    y_val_hat = regr.predict(X_val)

    print("Train surrogate R2 score:", sklearn.metrics.r2_score(y_train, y_train_hat))
    print("Test surrogate R2 score:", sklearn.metrics.r2_score(y_test, y_test_hat))
    print("Test surrogate R2 score:", sklearn.metrics.r2_score(y_val, y_val_hat))
    print("Train surrogate MSE score:", sklearn.metrics.mean_squared_error(y_train, y_train_hat))
    print("Test surrogate MSE score:", sklearn.metrics.mean_squared_error(y_test, y_test_hat))
    print("Val surrogate MSE score:", sklearn.metrics.mean_squared_error(y_val, y_val_hat))

    print("Val model error:", sklearn.metrics.mean_squared_error(y_val, z_val))
    print("Surrogate model error:", sklearn.metrics.mean_squared_error(y_val_hat, z_val))
    print('Val model error rel to last value:',sklearn.metrics.mean_squared_error(y_val, y_val_hat)/sklearn.metrics.mean_squared_error(y_val, np.zeros_like(y_val)))

    # How does surrogate error compare to original?
    print('Val surrogate prediction error relative to model:',sklearn.metrics.mean_squared_error(z_val, y_val_hat)/sklearn.metrics.mean_squared_error(z_val, y_val))
    print('Val surrogate abs error relative to model:',
          sklearn.metrics.mean_absolute_error(z_val, y_val_hat) / sklearn.metrics.mean_absolute_error(z_val, y_val))

    ratio = sklearn.metrics.mean_squared_error(z_val, y_val_hat)/sklearn.metrics.mean_squared_error(z_val, y_val)
    return ratio

if __name__=='__main__':
    import matplotlib.pyplot as plt

    df_big = load_massaged()
    df_big.rename(inplace=True, columns={'y_next': 'x'})
    n_big = len(df_big)

    n_totals = [ int(math.exp(g)) for g in np.linspace(start=math.log(1000),stop=math.log(n_big),num=10) ]

    import matplotlib.pyplot as plt
    ratios = list()
    for i,n_total in enumerate(n_totals):
        print({'n_total':n_total,'n_big':n_big})
        df = df_big[1:int(math.floor(n_total))]
        ratio = 1/(surrogate_ratio(df) - 1)
        ratios.append(ratio)
        print({'ratios':ratios})
        if i>=2:
            plt.plot(n_totals[:i+1],ratios)
            plt.show()
        import time
        time.sleep(5)


