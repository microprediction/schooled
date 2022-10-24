import pandas as pd
from schooled.datasets.filenameconventions import matching_generated_csvs, OUTPUT_DIR


def collate_arima_csv():
    dfs = [pd.read_csv(fn, header=None) for fn in matching_generated_csvs(keyword='train')]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(OUTPUT_DIR+'/collate.csv', header=False, index=False)


if __name__=='__main__':
    collate_arima_csv()