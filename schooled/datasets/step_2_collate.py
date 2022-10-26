import pandas as pd
from schooled.whereami import OUTPUT_DIR, RUNNING_LOCALLY
import time
from glob import glob


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


if __name__=='__main__':
    collate_arima_csv()
