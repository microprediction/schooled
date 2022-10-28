from schooled.whereami import OUTPUT_DIR, RUNNING_LOCALLY, ROOT_PATH
from glob import glob
import os


def generated_csv(file_no):
    return os.path.join(OUTPUT_DIR, 'generated_' + str(file_no) + '.csv')


def collated_file_path():
    return os.path.join(OUTPUT_DIR,'collated.csv')


def massaged_file_path():
    return os.path.join(OUTPUT_DIR,'massaged.csv')



if __name__=='__main__':
    print(OUTPUT_DIR)
    print(collated_file_path())
