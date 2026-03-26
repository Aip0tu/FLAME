from FLAME import fcnn_predict
import os

if __name__ == '__main__':
    targets = ['abs', 'emi', 'plqy', 'e']
    # for data_base in ['deep4chem', 'FluoDB']:
    #     for model in ['deep4chem', 'FluoDB']:
    for data_base in [ 'FluoDB']:
        for model in [ 'FluoDB']:
            for target in targets:
                model_path = f'model/fcnn/{model}_{target}.h5'
                # input_file = f'data/{data_base}/{target}_test.csv'
                # output_file = f'pred/{data_base}/fcnn_{model}_{target}.csv'
                input_file = f'data/{data_base}/pre.csv'
                output_file = f'pred/test/fcnn_{target}_2.csv'
                if not os.path.exists(f'pred/{data_base}/'):
                    os.makedirs(f'pred/{data_base}/')
                fcnn_predict(model_path, output_file, input_file=input_file)
