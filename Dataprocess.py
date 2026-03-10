import pandas as pd
import numpy as np
from FLAME.dataprocess.utils import load_data

df_origin = load_data('data/Database/')
from FLAME.dataprocess.utils import get_solmap, get_solvent_df

solvents = set(df_origin.solvent.values)
print('Total %d solvents' %(len(solvents)))

solmap, mix_solvent, unknow_solvent = get_solmap(solvents)
df, df_mix = get_solvent_df(df_origin, solmap)
# analysis the duplicated data
delta_a = []
delta_e = []
delta_p = []
delta_ep = []
print('start Merge Duplicate data')
for smi, sdf in df.groupby(['smiles', 'solvent']):
    if len(sdf) > 1:
        delta_a.append(max(sdf['absorption/nm'])-min(sdf['absorption/nm']))
        delta_e.append(max(sdf['emission/nm'])-min(sdf['emission/nm']))
        delta_p.append(max(sdf['plqy'])-min(sdf['plqy']))
        delta_ep.append(max(sdf['e/m-1cm-1'])-min(sdf['e/m-1cm-1']))
from FLAME.dataprocess.utils import merge_item

#delta = 5
ndf = []
print('start Merge Duplicated data')
for smi, sdf in df.groupby(['smiles', 'solvent']):
    if len(sdf) > 1:
        ssdf = merge_item(sdf)
    else:
        ssdf = sdf.copy()
    if not len(ssdf):
        print('drop item')
        continue
    if len(ndf):
        ndf = pd.concat([ndf, ssdf])
    else:
        ndf = ssdf

print(len(df),' before')
print(len(ndf),' after')
ndf = ndf.reset_index(drop=True)
import pandas as pd
from rdkit import Chem
from FLAME.dataprocess.flsf.scaffold import scaffold


dt = [(k,Chem.MolFromSmiles(m)) for k,v in scaffold.items() for m in v]
scaff_dict = dict([(k,v) for v,k in enumerate(scaffold.keys())])
patterns = pd.DataFrame({
    'idx':[scaff_dict[x] for x in list(zip(*dt))[0]],
    'mol':list(zip(*dt))[1]
})
from tqdm import tqdm

ndf['tag'] = -1
for i in tqdm(range(len(ndf))):
    if ndf.loc[i, 'tag'] != -1:
        continue
    mol = Chem.MolFromSmiles(ndf.loc[i].smiles)
    for _, patt in patterns.iterrows():
        if mol.HasSubstructMatch(patt.mol):
            ndf.loc[i, 'tag'] = patt.idx
            break
# export scaffold
import os
os.makedirs('data/scaffold', exist_ok=True)
for k,v in scaffold.items():
    writer = Chem.SDWriter(f'data/scaffold/{k}.sdf')
    for i, smi in enumerate(v):
        mol = Chem.MolFromSmiles(smi)
        writer.write(mol)
    writer.close()

scaff_dict_r = dict([(str(v),k) for k,v in scaff_dict.items()])
scaff_dict_r['-1'] = 'None'
for k,v in dict(ndf.groupby(['tag']).size()).items():
    print(k, scaff_dict_r[str(k)], v)
ndf['tag_name'] = [scaff_dict_r[str(t)] for t in ndf.tag]

from rdkit.Chem import Draw

# Sample molecules with tag=-1, handle case where there might be fewer than 20
tag_minus_one = ndf[ndf.tag==-1]
if len(tag_minus_one) > 0:
    n_sample = min(20, len(tag_minus_one))
    smis = dict(tag_minus_one.sample(n=n_sample).smiles)
    mols = [Chem.MolFromSmiles(m) for m in smis.values()]
    Draw.MolsToGridImage(mols, molsPerRow=5, legends=np.array(list(smis.keys())).astype('str').tolist())
else:
    print("No molecules with tag=-1 found for visualization")

# Dataset Split
import pandas as pd
df = pd.read_csv('data/FluoDB-Lite.csv')
df = df[df['smiles'].str.find('.')==-1]
df = df.sample(frac=1.)
df.iloc[:(len(df)*7)//10, -1] = 'train'
df.iloc[(len(df)*7)//10:(len(df)*8)//10, -1] = 'valid'
df.iloc[(len(df)*8)//10:, -1] = 'test'
df = df.sample(frac=1.).reset_index(drop=True)

# process FluoDB dataset
target_dict = {
            'abs': 'absorption/nm',
            'emi':'emission/nm',
            'plqy':'plqy',
            'e':'e/m-1cm-1'
        }
target = 'abs'
df_target = df[df[target_dict[target]]>0]
df_target.rename(columns={target_dict[target]:target}, inplace = True)
# Ensure directory exists
os.makedirs('data/FluoDB', exist_ok=True)
df_target[df_target['split']=='train'].loc[:,['smiles', 'solvent', target]].to_csv(f'data/FluoDB/{target}_train.csv', index=False)
df_target[df_target['split']=='test'].loc[:,['smiles', 'solvent', target]].to_csv(f'data/FluoDB/{target}_test.csv', index=False)
df_target[df_target['split']=='valid'].loc[:,['smiles', 'solvent', target]].to_csv(f'data/FluoDB/{target}_valid.csv', index=False)

# process schnet dataset
import sys
import os
from ase.db import connect
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from ase import Atoms
from tqdm import tqdm
#from FLAME.schnetpack.data import AtomsData
from FLAME.schnetpack.environment import SimpleEnvironmentProvider

RDLogger.DisableLog('rdApp.*')

def valid_atoms(
    atoms,
    environment_provider=SimpleEnvironmentProvider(),
    collect_triples=False,
    centering_function=None,
):
    inputs = {}
    atoms.numbers.astype(np.dtype('int'))
    atoms.positions.astype(np.dtype('float32'))

    # get atom environment
    nbh_idx, offsets = environment_provider.get_environment(atoms)

    # Get neighbors and neighbor mask
    nbh_idx.astype(np.dtype('int'))
    # Get cells
    np.array(atoms.cell.array, dtype=np.dtype('float32'))
    offsets.astype(np.dtype('float32'))
    return True

def numpyfy_dict(data):
    for k, v in data.items():
        if type(v) in [int, float]:
            v = np.array([v])
        if v.shape == ():
            v = v[np.newaxis]
        data[k] = v
    return data

def get_center_of_mass(atoms):
    masses = atoms.get_masses()
#    print(atoms)
    return np.dot(masses, atoms.arrays["positions"]) / masses.sum()

def smiles2coord(smi, conf_num=3):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
#     smiles = Chem.MolToSmiles(mol)
    AllChem.Compute2DCoords(mol)
    coords = []
    for i in range(conf_num):
        AllChem.EmbedMolecule(mol)
        data = Chem.MolToXYZBlock(mol)
        coord = np.array([x[2:].strip().split() for x in data.strip().split('\n')[2:]]).astype(float)
        coords.append(coord)
    species = data.split()[1::4]
    return coords, species

def get_schnet_data(smiles, target, indices, save_path='', conf_num=1):
    if len(save_path) == 0:
        save_path = 'data/schnet/data.db'
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    available_properties = ["energy", "indices"]
    atoms_list = []
    property_list = []
    energies = target
    for idx,smi in tqdm(enumerate(smiles)):
        positions, species = smiles2coord(smi, conf_num=conf_num)
        species = ''.join(species)
        if len(species) == 0:
            print(smi)
            continue
        for i in range(conf_num):
            atm = Atoms(species, positions[i])
#             valid_atoms(atm)
#             nbh_idx, offsets = environment.get_environment(atm)
#             ct = get_center_of_mass(atm)
            try:
                energy = energies[idx]
                properties = {"energy": energy, "indices":indices[idx]}
                atoms_list.append(atm)
                property_list.append(properties)
            except:
                print(smi)
                break

    key_value_pairs_list = [dict() for _ in range(len(atoms_list))]

    with connect(save_path) as conn:
        for at, prop, kv_pair in tqdm(zip(atoms_list, property_list, key_value_pairs_list)):
            data = {}
            # add available properties to database
            for pname in available_properties:
                try:
                    data[pname] = prop[pname]
                except:
                    raise Exception("Required property missing:" + pname)
            # transform to np.ndarray
            data = numpyfy_dict(data)
            conn.write(at, data=data, key_value_pairs=kv_pair)

db = 'FluoDB'
target = 'abs'
split = 'test'
schnet_df = pd.read_csv(f'data/{db}/{target}_{split}.csv')

# Note: The CSV file has column name 'abs' (not 'absorption/nm') after processing
schnet_df = schnet_df[schnet_df[target]>0]
schnet_df = schnet_df[schnet_df['smiles'].str.find('+')==-1]
schnet_df = schnet_df[schnet_df['smiles'].str.find('-')==-1]
schnet_df = schnet_df[schnet_df['smiles'].str.len() < 40] # schnet cannot process molecules too large
schnet_df = schnet_df.drop_duplicates(subset=['smiles'])

get_schnet_data(schnet_df['smiles'].values, 1240./schnet_df[target].values, schnet_df.index.values, save_path=f'data/schnet/{db}/{target}_{split}.db', conf_num=5)

# process Deep4Chem dataset

deep4chem_df = df[df['source'].str.find('Deep4Chem')>-1]
# Use the same target as defined above (target = 'abs')
# Get the original column name from target_dict
original_col_name = target_dict[target]
deep4chem_df_target = deep4chem_df[deep4chem_df[original_col_name]>0]
# Rename the column to match the target name
deep4chem_df_target.rename(columns={original_col_name: target}, inplace=True)
# Ensure directory exists
os.makedirs('data/deep4chem', exist_ok=True)
deep4chem_df_target[deep4chem_df_target['split']=='train'].loc[:,['smiles', 'solvent', target]].to_csv(f'data/deep4chem/{target}_train.csv', index=False)
deep4chem_df_target[deep4chem_df_target['split']=='test'].loc[:,['smiles', 'solvent', target]].to_csv(f'data/deep4chem/{target}_test.csv', index=False)
deep4chem_df_target[deep4chem_df_target['split']=='valid'].loc[:,['smiles', 'solvent', target]].to_csv(f'data/deep4chem/{target}_valid.csv', index=False)

# process GBRT dataset
import pandas as pd

from FLAME.dataprocess.gbrt import get_GBRT_data

database = 'FluoDB'
target = 'abs'
split = 'test'
train_df = pd.read_csv(f'data/{database}/{target}_{split}.csv')
train_df = get_GBRT_data(train_df, feature_save=False)