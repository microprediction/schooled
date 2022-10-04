from schooled.wherami import SKATER_DATA, SKATER
import pathlib
import numpy as np
from schooled.datasets.ornstein import simulate_arima_like_path

NUM_ROWS = 10
SEQ_LEN = 20


def skater_single_prediction(ys, f):
    """ One step ahead prediction for any skater """
    # Pass e<0 for most of the data
    s = {}
    for y in ys[:-1]:
        x, x_std, s = f(y=y, k=1, s=s, e=-1)
    y_final = ys[-1]

    # Then tell it to think hard about the last one
    x, x_std, s = f(y=y_final, k=1, s=s, e=1000)

    return x


def make_data(start_file_no, end_file_no,plot=False):
    pathlib.Path(SKATER_DATA).mkdir(parents=True, exist_ok=True)
    from timemachines.skaters.sk.skautoarima import sk_autoarima as f
    for file_no in range(start_file_no,end_file_no):
        csv = SKATER_DATA + '/train_' + str(file_no) + '.csv'
        csv = '/output/train_' + str(file_no) + '.csv'
        print('Making '+ csv)
        data = list()
        row_no = 0
        while row_no<NUM_ROWS:
            okay = False
            while not okay:
                try:
                    ys = simulate_arima_like_path(seq_len=SEQ_LEN)
                    assert np.max(ys)<10
                    assert np.min(ys)>-10
                    okay = True
                    row_no+=1
                except:
                    print('Failed')

            # Run skater twice
            x0 = skater_single_prediction(ys, f)
            x1 = skater_single_prediction(ys, f)
            if abs(x1[0]-x0[0])>0.00001:
                raise Exception('Skater is non-deterministic ')

            example = np.concatenate([ys, x0] )
            last_few = example[-6:]
            print(last_few)

            data.append(example)
            if row_no % 100 ==0:
                print(str(row_no)+' of '+str(NUM_ROWS))
                X = np.array(data)
                np.savetxt(fname=csv, X=X, delimiter=',')


if __name__=='__main__':
        import argparse
        parser = argparse.ArgumentParser(description='sarima data')
        parser.add_argument('--index', help='number of epochs to run', default='1000')
        args = parser.parse_args()
        start_file_no = int(args.index)*100
        end_file_no = start_file_no+100
        make_data(start_file_no=start_file_no, end_file_no=end_file_no)
    
    
    
