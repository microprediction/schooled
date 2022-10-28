import pandas as pd
from schooled.whereami import OUTPUT_DIR, RUNNING_LOCALLY
import time
from glob import glob
from schooled.whereami import OUTPUT_DIR
from schooled.generating.generationfilenames import collated_file_path, massaged_file_path
from schooled.generating.step_1_generate import SEQ_LEN
from schooled.generating.step_1_generate import FS_SHORT_NAMES, FS_ERROR_NAMES, FS_COLS


if RUNNING_LOCALLY:
    def collate_arima_csv():
        # Do this manually
        pass
else:
    def collate_arima_csv():
        from schooled.cnvrg.outputsync import sync_previous
        sync_previous(flow_name='sarima')
        time.sleep(1)
        INPUT_CSVS = glob(OUTPUT_DIR+'/*/output/*.csv', recursive=False)
        print({'input_csv_files':INPUT_CSVS[:2],'num_files':len(INPUT_CSVS)})
        dfs = [pd.read_csv(fn, header=None) for fn in INPUT_CSVS ]
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(OUTPUT_DIR+'/collated.csv', header=False, index=False)


def load_collated():
    return pd.read_csv(collated_file_path(),header=None)


def massage_collated():
    df = load_collated()
    num_cols = len(df.columns)
    num_models = (num_cols-SEQ_LEN-1)/2
    assert int(num_models)==len(FS_SHORT_NAMES)
    df.rename(columns=dict(zip(df.columns,FS_COLS)),inplace=True)
    df.to_csv(massaged_file_path(), index=False)


def load_massaged():
    return pd.read_csv(massaged_file_path())


if __name__=='__main__':
    collate_arima_csv()
    massage_collated()
