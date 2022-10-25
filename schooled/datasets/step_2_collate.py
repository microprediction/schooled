import pandas as pd
from schooled.datasets.filenameconventions import matching_generated_csvs, INPUT_DIR, OUTPUT_DIR


def collate_arima_csv():
    from schooled.cnvrg.outputsync import sync_previous
    sync_previous()
    all_csvs = matching_generated_csvs(keyword='')
    print({'all_csvs':all_csvs})

    matching_csvs = matching_generated_csvs(keyword='generated')
    print({'matching_csvs':matching_csvs})
    dfs = [pd.read_csv(fn, header=None) for fn in matching_csvs ]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(OUTPUT_DIR+'/collate.csv', header=False, index=False)


if __name__=='__main__':
    collate_arima_csv()
