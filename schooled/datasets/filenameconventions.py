from schooled.wherami import OUTPUT_DIR
from glob import glob


def generated_csv(file_no):
    return OUTPUT_DIR + '/generated_' + str(file_no) + '.csv'


def matching_generated_csvs(keyword:str):
    return [name for name in glob(OUTPUT_DIR+'/*'+keyword+'*.csv')]



if __name__=='__main__':
    print(OUTPUT_DIR)
    print(matching_generated_csvs('train'))