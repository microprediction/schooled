import os
import pathlib

ROOT_PATH = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent)
SKATER = 'sf_autoarimawiggly'
CURRENT_EXPERIMENT = 'sarima'
SKATER_DATA = ROOT_PATH + os.path.sep + 'data' + os.path.sep + CURRENT_EXPERIMENT + os.path.sep + SKATER


if __name__=='__main__':
    print('Skater data in ' + SKATER_DATA)


