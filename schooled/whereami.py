import os
import pathlib
import platform

ROOT_PATH = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent)
CURRENT_EXPERIMENT = 'bigsarima'
RUNNING_LOCALLY = 'macOS' in platform.platform()

if RUNNING_LOCALLY:
    OUTPUT_DIR = os.path.join(ROOT_PATH,'localoutput',CURRENT_EXPERIMENT,'output')
else:
    OUTPUT_DIR = '/cnvrg/output'


if __name__=='__main__':
    print(platform.platform())
    print({'RUNNING_LOCALLY':RUNNING_LOCALLY,'OUTPUT_DIR':OUTPUT_DIR})


