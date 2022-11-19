# -*- coding: utf-8 -*-


from schooled.generating.step_2_collate import load_massaged
import sklearn.metrics
import numpy as np
import math


"""# Data to fit """



def classification_success(df):
    # Try to learn ensemble weights

    x_cols = [ c for c in df.columns if 'y_' in c ]
    assert 'x' not in x_cols

    err_cols = [ c for c in df.columns if '_err' in c ]
    model_cols = [ c[:-4] for c in err_cols ]
    n_models = len(err_cols)

    # Find the closest model
    best_model = np.argmin(df[err_cols].values,axis=1)

    y = best_model            # <--- What we want to train
    X = df[x_cols].values
    z = df['x'].values        # <--- The target (next value in the series)
    y1 = y                    # <--- What to classify


    X = X.astype(np.float32)
    n_test = int(len(df)/10)
    n_train = len(df)-2*n_test
    X_train, y_train, z_train = X[:n_train], y1[:n_train], z[:n_train]
    X_test, y_test, z_test = X[n_train:(n_train+n_test)], y1[n_train:(n_train+n_test)], z[n_train:(n_train+n_test)]
    X_val, y_val, z_val = X[(n_train+n_test):], y1[(n_train+n_test):], z[(n_train+n_test):]

    models_train = df[model_cols].values[:n_train]
    models_test = df[model_cols].values[n_train:(n_train+n_test)]
    models_val = df[model_cols].values[(n_train + n_test):]

    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(random_state=1, max_iter=5000,
                        hidden_layer_sizes=(200, 150, 150, 50)).fit(X_train, y_train)

    SHRINKAGE = 0.9
    y_val_proba_raw = clf.predict_proba(X_val)
    y_val_even_proba = np.ones_like(y_val_proba_raw) / n_models
    y_val_proba = y_val_proba_raw*(1-SHRINKAGE)+y_val_even_proba*SHRINKAGE

    y_val_hat = np.sum(y_val_proba * models_val, axis=1)
    y_val_raw_hat = np.sum(y_val_proba_raw * models_val, axis=1)

    y_val_even_hat = np.sum(y_val_even_proba * models_val, axis=1)

    val_mse_mixture = sklearn.metrics.mean_squared_error(y_val_hat, z_val)
    val_mse_average = sklearn.metrics.mean_squared_error(y_val_even_hat, z_val)
    val_mse_raw = sklearn.metrics.mean_squared_error(y_val_raw_hat, z_val)

    for i,model_name  in enumerate(model_cols):
        print("Val "+model_name+" error: ",sklearn.metrics.mean_squared_error(models_val[:,i],z_val)/val_mse_average)

    print("Val raw     model error:", val_mse_raw / val_mse_average)
    print("Val mixture model error:", val_mse_mixture/val_mse_average)
    print("Val average model error:", 1)

    ratio = val_mse_mixture/val_mse_average
    return ratio

if __name__=='__main__':
    import matplotlib.pyplot as plt

    df_big = load_massaged()
    df_big.rename(inplace=True, columns={'y_next': 'x'})
    n_big = len(df_big)

    n_totals = [ int(math.exp(g)) for g in np.linspace(start=math.log(1000),stop=math.log(n_big),num=5) ]

    import matplotlib.pyplot as plt
    ratios = list()
    for i,n_total in enumerate(n_totals):
        df = df_big[1:int(math.floor(n_total))]
        ratio = classification_success(df)
        ratios.append(ratio)
        print({'ratios':ratios})
        if i>=2:
            plt.plot(n_totals[:i+1],ratios)
            plt.show()
        import time
        time.sleep(5)


