from schooled.wherami import OUTPUT_DIR
from glob import glob


def generated_csv(file_no):
    return OUTPUT_DIR + '/generated_' + str(file_no) + '.csv'



if __name__=='__main__':
    print(OUTPUT_DIR)
