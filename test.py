import torch
torch_version = torch.__version__
gpu = torch.cuda.is_available()
print(f'Troch version: {torch_version}, GPU status: {gpu}')

import FLAME
print(f'FLAME verion: {FLAME.__version__}')

if __name__ == '__main__':

    print('Check Done!')
    from FLAME import run_search

    SMILES = 'COc1cc(OC)c2ccc(=O)oc2c1'
    similarity_limit = 0.5
    fingerprinter_type = 'morgan'

    result_df = run_search(SMILES, similarity_limit, fingerprinter_type)
    result_df
    # make schnet data
    import pandas as pd
    import numpy as np
    from FLAME.utils import get_schnet_data

    data_path = 'data/FluoDB/abs_test.csv'
    df = pd.read_csv(data_path)
    df = df[df['smiles'].str.len() < 40]
    df = df[df['smiles'].str.find('+') == -1]
    df = df[df['smiles'].str.find('-') == -1]
    df = df.drop_duplicates(subset=['smiles'])

    get_schnet_data(df['smiles'].values, np.zeros(len(df['smiles'])), df.index.values, save_path=data_path + '.db',
                    conf_num=1)

    import os
    from FLAME import flsf_train as training

    # from FLAME import uvvisml_train as training
    # from FLAME import abtmpnn_train as training
    # from FLAME import fcnn_train as training
    # from FLAME import gbrt_train as training
    # from FLAME import schnet_train as training

    epoch = 1
    train_data = 'data/FluoDB/abs_train.csv'
    valid_data = 'data/FluoDB/abs_valid.csv'
    test_data = 'data/FluoDB/abs_test.csv.db'
    model_save_path = 'model/test'

    if os.path.exists(model_save_path):
        print(model_save_path, ' exists!')

    training(model_save_path, train_data, valid_data, test_data, epoch)

    # Predict single
    import pandas as pd
    import numpy as np
    from FLAME import flsf_predict as predict

    # from FLAME import uvvisml_predict as predict
    # from FLAME import abtmpnn_predict as predict
    # from FLAME import fcnn_predict as predict
    # from FLAME import gbrt_predict as predict
    # from FLAME import schnet_predict as predict

    solvent = ['O', 'CS(C)=O', 'C(Cl)Cl', 'CCO']
    smiles = ['O=C1OC2=CC=C(C=C2C3=C1N=CO3)N', 'O=C1OC2=CC(N)=CC=C2C3=C1N=CO3']

    df = pd.DataFrame({
        'smiles': sorted((smiles * len(solvent))),
        'solvent': solvent * len(smiles)
    })
    target = 'abs'
    model_path = f'../model/flsf/FluoDB_{target}'
    output_file = 'test.csv'

    df[f'{target}_pred'] = predict(model_path, output_file, smiles=df[['smiles', 'solvent']].values.tolist())
    if target == 'e':
        df[f'{target}_pred'] = np.log10(df[f'{target}_pred'])

    # Predict File
    import pandas as pd
    from FLAME import flsf_predict as predict

    # from FLAME import uvvisml_predict as predict
    # from FLAME import abtmpnn_predict as predict
    # from FLAME import fcnn_predict as predict
    # from FLAME import gbrt_predict as predict
    # from FLAME import schnet_predict as predict

    save = False

    target = 'abs'
    model_path = f'model/flsfluoDB_{target}'
    input_file = f'data/FluoDB/{target}_test.csv'
    output_file = 'pred/test.csv'

    df = pd.read_csv(input_file)
    df['pred'] = predict(model_path, output_file, smiles=df[['smiles', 'solvent']].values.tolist())

    import numpy as np
    import pandas as pd
    from scipy import stats

    res = []
    idx = []

    solvent = -1
    # solvent = 4

    models = ['GBRT', 'FCNN', 'uvvisml', 'schnet', 'abtmpnn', 'flsf_maccs', 'flsf_morgan', 'flsf']
    targets = ['abs', 'emi', 'plqy', 'e']

    for target in targets:
        base = pd.read_csv(f'data/FluoDB/{target}_test.csv').iloc[:, -1]

        for m in models:
            idx.append((target, m))
            pred_df = pd.read_csv(f'pred/{m}/{m}_{target}.csv')

            if 'pred' not in pred_df.columns:
                pred_df['pred'] = pred_df[target]

            pred_df[target] = base

            if target == 'e':
                pred_df['pred'] = np.log10(pred_df['pred'])

            pred_df[target] = np.log10(pred_df[target])
            pred_df = pred_df.dropna()

            if solvent == 1:  # single solvent
                pred_df = pred_df[pred_df['solvent'] == 'ClCCl']
            elif solvent > 1:
                pred_df['snum'] = pred_df.groupby('smiles')['smiles'].transform('count')
                pred_df = pred_df[pred_df['snum'] > solvent]

            pred_df['err'] = abs(pred_df['pred'] - pred_df[target])
            mae = pred_df['err'].mean()
            mse = (pred_df['err'] ** 2).mean()
            rmse = mse ** 0.5
            slope, intercept, r_value, p_value, std_err = stats.linregress(pred_df['pred'], pred_df[target])
            r2 = r_value ** 2

            res.append(
                [
                    round(mae, 3),
                    round(mse, 3),
                    round(rmse, 3),
                    round(r2, 3),
                    len(pred_df),
                    pred_df['smiles'].nunique(),
                ]
            )

    res = np.array(res)

    metrics = pd.DataFrame(
        res,
        index=pd.MultiIndex.from_tuples(idx),
        columns=['MAE', 'MSE', 'RMSE', 'R2', 'n_data', 'n_mol'],
    )

    print(metrics)