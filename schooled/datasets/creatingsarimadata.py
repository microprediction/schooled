from schooled.wherami import OUTPUT_DIR, SKATER
import pathlib
import numpy as np
from schooled.datasets.ornstein import simulate_arima_like_path

NUM_ROWS = 10
SEQ_LEN = 100


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


def make_data(start_file_no, end_file_no,plot=False):
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    from timemachines.skaters.sk.skautoarima import sk_autoarima as f1
    from timemachines.skaters.sk.skautoarimawiggly import sk_autoarima_wiggly_huber_d05_m3 as f2

    for file_no in range(start_file_no,end_file_no):
<<<<<<< HEAD
        csv = OUTPUT_DIR + '/train_' + str(file_no) + '.csv'
=======
        #csv = SKATER_DATA + '/train_' + str(file_no) + '.csv'
        csv = '/cnvrg/output/train_' + str(file_no) + '.csv'
>>>>>>> 1cd75fa88d59be05084df229f32943af55398d9c
        print('Making '+ csv)
        data = list()
        row_no = 0
        while row_no<NUM_ROWS:
            okay = False
            while not okay:
                try:
                    ys_ = simulate_arima_like_path(seq_len=SEQ_LEN+11)[10:]
                    y_next = ys_[-1]
                    ys = ys_[:-1]
                    assert np.max(ys)<10
                    assert np.min(ys)>-10
                    # Run skater twice, maybe
                    x01 = skater_single_prediction(ys=ys, f=f1)
                    x02 = skater_single_prediction(ys=ys, f=f2)
                    if row_no % 20 == 0:
                        x1 = skater_single_prediction(ys=ys, f=f1)
                        if abs(x1[0] - x01[0]) > 0.00001:
                            raise Exception('Skater is non-deterministic ')

                    okay = True
                    row_no+=1
                except:
                    print('Arima-like path generation was too wild, or model failed')


            example = np.concatenate([ys, x01, x02, [y_next, x01[0]-y_next, x02[0]-y_next]] )
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
        end_file_no = start_file_no+2
        make_data(start_file_no=start_file_no, end_file_no=end_file_no)
    
    
    
