from schooled.whereami import OUTPUT_DIR
import pathlib
import numpy as np
from schooled.generating.ornstein import simulate_arima_like_path
from schooled.generating.generationfilenames import generated_csv

NUM_ROWS = 500
SEQ_LEN = 200

DEBUG_FLOW = False

from timemachines.skaters.tsa.tsaconstant import tsa_p1_d1_q0 as f1
from timemachines.skaters.tsa.tsaconstant import tsa_p2_d1_q0 as f2
from timemachines.skaters.tsa.tsaconstant import tsa_p3_d1_q0 as f3
from timemachines.skaters.simple.trivial import trivial_last_value as f4


FS = [f1, f2, f3, f4]
FS_SHORT_NAMES = ['p1', 'p2', 'p3', 'trivial']
FS_ERROR_NAMES = [fsn + '_err' for fsn in FS_SHORT_NAMES]
FS_COLS = [ 'y_'+str(i) for i in range(SEQ_LEN-1)] + ['y_last', 'y_next'] + FS_SHORT_NAMES + FS_ERROR_NAMES


def skater_single_prediction(ys, f):
    """ One step ahead prediction for any skater """
    # Pass e<0 for most of the data
    s = {}
    for y in ys[:-1]:
        x, x_std, s = f(y=y, k=1, s=s, e=-1)
    y_final = ys[-1]

    # Then tell it to think hard abouft the last one
    x, x_std, s = f(y=y_final, k=1, s=s, e=1000)

    return x


def generate_csv(start_file_no, end_file_no, fs, plot=False):
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for file_no in range(start_file_no,end_file_no):
        csv = generated_csv(file_no=file_no)
        print('Making '+ csv)
        data = list()
        row_no = 0

        if DEBUG_FLOW:
            X = np.random.randn(50,20)
            np.savetxt(fname=csv, X=X, delimiter=',')
        else:
            while row_no<NUM_ROWS:
                okay = False

                while not okay:
                    try:
                        ys_ = simulate_arima_like_path(seq_len=SEQ_LEN+11)[10:]
                        y_next = ys_[-1]
                        ys = ys_[:-1]
                        assert np.max(ys)<10
                        assert np.min(ys)>-10

                        y_hats_all_horizons = [ skater_single_prediction(ys=ys, f=f) for f in fs ]

                        okay = True
                        row_no+=1
                    except:
                        print('Arima-like path generation was too wild, or model failed')

                y_hats = [ yh[0] for yh in y_hats_all_horizons ]
                y_errs = [ y_next-yh for yh in y_hats ]
                example = np.concatenate([ys, [y_next],y_hats, y_errs ] )
                last_few = example[-6:]
                print(last_few)
                print(str(row_no) + ' of ' + str(NUM_ROWS))

                data.append(example)

                if row_no>1:
                    print(str(row_no)+' of '+str(NUM_ROWS))
                    X = np.array(data)
                    err_abs_mean = np.mean(np.abs(X[:, -len(fs):]),axis=0)
                    err_rms_mean = np.sqrt(np.mean(X[:, -len(fs):]**2,axis=0))
                    from pprint import pprint
                    pprint({'names':['auto','wiggly','tsa_2','tsa_1'],
                            'err_abs': err_abs_mean, 'err_rms_mean': err_rms_mean})

                    np.savetxt(fname=csv, X=X, delimiter=',')
            X = np.array(data)
            np.savetxt(fname=csv, X=X, delimiter=',')


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='sarima data')
    parser.add_argument('--index', help='number of epochs to run', default='1000')
    args = parser.parse_args()
    start_file_no = int(args.index)*100
    end_file_no = start_file_no+2

    generate_csv(start_file_no=start_file_no, end_file_no=end_file_no, fs=FS)

    
    
