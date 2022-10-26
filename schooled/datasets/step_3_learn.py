import pandas as pd
from schooled.whereami import OUTPUT_DIR, RUNNING_LOCALLY, ROOT_PATH
import time
import os
from glob import glob
from schooled.datasets.filenameconventions import collated_file_path


def load_collated():
    return pd.read_csv(collated_file_path(),header=None)


def save_statistics():
    # placeholder
    # Recall:  [ys, x01, x02, [y_next, x01[0]-y_next, x02[0]-y_next]]
    df = load_collated()
    new_cols = list(df.columns[:-6]) + ['y_last','y_hat_autoarima','y_hat_wiggle','y','y_err_autoarima','y_err_wiggle']
    df.rename(columns=dict(zip(df.columns,new_cols)),inplace=True)
    df['y_err_lv'] = df['y'] - df['y_last']
    err_cols = [ c for c in list(df.columns) if isinstance(c,str) and '_err_' in c]
    errors = df[err_cols].abs().describe()
    errors['wiggle_ratio'] = errors['y_err_wiggle'] / errors['y_err_autoarima']
    errors['lv_ratio'] = errors['y_err_lv'] / errors['y_err_autoarima']
    print(errors)
    errors.to_csv(os.path.join(OUTPUT_DIR,'errors.csv'))




if __name__=='__main__':
    save_statistics()
