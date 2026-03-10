# Predict single
import pandas as pd
import numpy as np
from FLAME import flsf_predict as predict
# from FLAME import uvvisml_predict as predict
# from FLAME import abtmpnn_predict as predict
# from FLAME import fcnn_predict as predict
# from FLAME import gbrt_predict as predict
# from FLAME import schnet_predict as predict

solvent = ['O', 'CS(C)=O','C(Cl)Cl', 'CCO']
smiles = ['O=C1OC2=CC=C(C=C2C3=C1N=CO3)N','O=C1OC2=CC(N)=CC=C2C3=C1N=CO3']

df = pd.DataFrame({
    'smiles': sorted((smiles*len(solvent))),
    'solvent': solvent*len(smiles)
})
target = 'abs'
model_path = f'./model/test/fold_0/model_0'
output_file = 'test.csv'

df[f'{target}_pred'] = predict(model_path, output_file, smiles=df[['smiles','solvent']].values.tolist())
if target == 'e':
    df[f'{target}_pred'] = np.log10(df[f'{target}_pred'])


