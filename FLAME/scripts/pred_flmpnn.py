from FLAME import flsf_predict
import os
tag='_scaffold'
if __name__ == '__main__':
    targets = ['abs', 'emi', 'plqy', 'e']
    # for data_base in ['deep4chem', 'FluoDB']:
    #     for model in ['deep4chem', 'FluoDB']:
    for data_base in ['FluoDB']:
        for model in [ 'FluoDB']:
            for target in targets:
                model_path = f'model/flsf/{model}_{target}/fold_0/model_0'
                #input_file = f'data/{data_base}/{target}_test.csv'
                #output_file = f'pred/{data_base}/flsf{tag}_{model}_{target}.csv'
                if not os.path.exists(f'pred/test/'):
                    os.makedirs(f'pred/test/')
                input_file = f'data/{data_base}/pre.csv'
                output_file = f'pred/test/flsf_{target}_DMSO.csv'

                flsf_predict(model_path, output_file, input_file=input_file)


