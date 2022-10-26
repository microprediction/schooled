import os
import pathlib
import platform
from glob import glob

ROOT_PATH = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent)
SKATER = 'sf_autoarimawiggly'
CURRENT_EXPERIMENT = 'sarima'

RUNNING_LOCALLY = 'macOS' in platform.platform()

if RUNNING_LOCALLY:
    OUTPUT_DIR = ROOT_PATH + os.path.sep + 'data' + os.path.sep + CURRENT_EXPERIMENT + os.path.sep + SKATER
else:
    OUTPUT_DIR = '/cnvrg/output'
    try:
        INPUT_CSVS = glob(OUTPUT_DIR+'/*/output/*.csv', recursive=False)
        print({'input_csv_files':INPUT_CSVS[:2],'num_files':len(INPUT_CSVS)})
    except:
        INPUT_CSVS = []


if __name__=='__main__':
    print(platform.platform())
    print(RUNNING_LOCALLY)
    if RUNNING_LOCALLY:
       print('Skater data in ' + OUTPUT_DIR)



