from schooled.whereami import OUTPUT_DIR, RUNNING_LOCALLY, ROOT_PATH
from glob import glob
import os


def generated_csv(file_no):
    return OUTPUT_DIR + '/generated_' + str(file_no) + '.csv'


def collated_file_path():
    if RUNNING_LOCALLY:
        return os.path.join(ROOT_PATH, 'data', 'collated', 'sarima.csv')
    else:
        return os.path.join(OUTPUT_DIR, 'generate_collated', 'output', 'sarima.csv')


if __name__=='__main__':
    print(OUTPUT_DIR)
