from schooled.wherami import OUTPUT_DIR
import numpy as np
file_no = 1
csv = OUTPUT_DIR + '/train_' + str(file_no) + '.csv'


if __name__=='__main__':
    X = np.loadtxt(csv,delimiter=',')
    import matplotlib.pyplot as plt
    for x in X:
        plt.plot(x)
