from FLAME import fcnn_train
from FLAME import load_and_show_history
from FLAME import load_and_show_history1

import os


if __name__ == '__main__':
    epoch = 2000
    targets = ['abs', 'emi', 'plqy', 'e']
    # for data_base in ['deep4chem', 'FluoDB']:
    for data_base in [ 'FluoDB']:
        for target in targets:
            train_data = f'data/{data_base}/{target}_train.csv'
            test_data = f'data/{data_base}/{target}_test.csv'
            valid_data = f'data/{data_base}/{target}_valid.csv'
            os.makedirs('model/fcnn', exist_ok=True)
            model_path = f'model/fcnn/{data_base}_{target}.h5'
            pkl_file_path = f'model/fcnn/{data_base}_{target}.h5.pkl'
            #fcnn_train(model_path, train_data, valid_data, test_data, epoch)
            load_and_show_history1(pkl_file_path)
