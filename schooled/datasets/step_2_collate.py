import pandas as pd
from schooled.whereami import INPUT_CSVS, OUTPUT_DIR


def collate_arima_csv():
    from schooled.cnvrg.outputsync import sync_previous
    sync_previous(flow_name='sarima')
    dfs = [pd.read_csv(fn, header=None) for fn in INPUT_CSVS ]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(OUTPUT_DIR+'/collated.csv', header=False, index=False)


if __name__=='__main__':
    collate_arima_csv()
