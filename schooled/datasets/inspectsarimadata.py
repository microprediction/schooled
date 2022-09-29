from schooled.wherami import SKATER_DATA
import numpy as np
file_no = 1
csv = SKATER_DATA + '/train_' + str(file_no) + '.csv'


if __name__=='__main__':
    X = np.loadtxt(csv,delimiter=',')
    import matplotlib.pyplot as plt
    for x in X:
        plt.plot(x)
