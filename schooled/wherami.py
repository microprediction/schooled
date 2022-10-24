import os
import pathlib
import platform

ROOT_PATH = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent)
SKATER = 'sf_autoarimawiggly'
CURRENT_EXPERIMENT = 'sarima'

RUNNING_LOCALLY = 'macOS' in platform.platform()

if RUNNING_LOCALLY:
    OUTPUT_DIR = ROOT_PATH + os.path.sep + 'data' + os.path.sep + CURRENT_EXPERIMENT + os.path.sep + SKATER
    INPUT_DIR = OUTPUT_DIR
else:
    OUTPUT_DIR = '/cnvrg/output'
    INPUT_DIR = '/cnvrg/input'


if __name__=='__main__':
    print(platform.platform())
    print(RUNNING_LOCALLY)
    if RUNNING_LOCALLY:
       print('Skater data in ' + OUTPUT_DIR)



