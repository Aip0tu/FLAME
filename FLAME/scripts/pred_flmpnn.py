import os

import pandas as pd

from FLAME import flsf_predict


tag = '_scaffold'


if __name__ == '__main__':
    targets = ['abs', 'emi', 'plqy', 'e']

    # for data_base in ['deep4chem', 'FluoDB']:
    #     for model in ['deep4chem', 'FluoDB']:
    for data_base in ['FluoDB']:
        for model in ['FluoDB']:
            os.makedirs('pred/test', exist_ok=True)

            input_file = f'data/{data_base}/pre.csv'
            output_file = 'pred/test/flsf_all_DMSO_0421.csv'

            df = pd.read_csv(input_file)
            smiles = df[['smiles', 'solvent']].values.tolist()

            for target in targets:
                model_path = f'model/flsf/{model}_{target}/fold_0/model_0'
                df[f'{target}_pred'] = flsf_predict(
                    model_path=model_path,
                    smiles=smiles
                )

            df.to_csv(output_file, index=False)
            print(f'Saved combined predictions to {output_file}')
