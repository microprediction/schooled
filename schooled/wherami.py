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
    INPUT_DIR = OUTPUT_DIR
else:
    OUTPUT_DIR = '/cnvrg/output'
    try:
        input_dirs = glob('/input/*/', recursive=False)
        INPUT_DIR = input_dirs[0]+'output'
        print({'INPUT_DIR':INPUT_DIR,'input_dir_files':glob(INPUT_DIR+'/*/', recursive=False)})
    except IndexError:
        INPUT_DIR = ''


if __name__=='__main__':
    print(platform.platform())
    print(RUNNING_LOCALLY)
    if RUNNING_LOCALLY:
       print('Skater data in ' + OUTPUT_DIR)



