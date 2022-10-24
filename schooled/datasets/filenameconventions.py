from schooled.wherami import OUTPUT_DIR
from glob import glob


def output_csv(file_no):
    return OUTPUT_DIR + '/train_' + str(file_no) + '.csv'


def matching_output_csvs(keyword:str):
    return [name for name in glob(OUTPUT_DIR+'/*'+keyword+'*.csv')]



if __name__=='__main__':
    print(OUTPUT_DIR)
    print(matching_output_csvs('train'))