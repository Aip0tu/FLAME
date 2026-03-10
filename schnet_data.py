import sys
import os
from ase.db import connect
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from ase import Atoms
from tqdm import tqdm
import numpy as np
# from FLAME.schnetpack.data import AtomsData
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
    for idx, smi in tqdm(enumerate(smiles)):
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
                properties = {"energy": energy, "indices": indices[idx]}
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


# ============================================================================
# 处理所有目标属性
# ============================================================================

db = 'FluoDB'
targets = ['abs', 'emi', 'plqy', 'e']
splits = ['train', 'valid', 'test']

# 需要将波长转换为能量的目标
wavelength_targets = ['abs', 'emi']

for target in targets:
    print(f"\n{'=' * 50}")
    print(f"Processing target: {target}")
    print('=' * 50)

    for split in splits:
        print(f"\n--- Processing {split} set for {target} ---")

        # 读取CSV文件
        csv_path = f'data/{db}/{target}_{split}.csv'
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping...")
            continue

        schnet_df = pd.read_csv(csv_path)

        # 数据过滤
        schnet_df = schnet_df[schnet_df[target] > 0]
        schnet_df = schnet_df[schnet_df['smiles'].str.find('+') == -1]
        schnet_df = schnet_df[schnet_df['smiles'].str.find('-') == -1]
        schnet_df = schnet_df[schnet_df['smiles'].str.len() < 40]  # schnet cannot process molecules too large
        schnet_df = schnet_df.drop_duplicates(subset=['smiles'])

        print(f"  Samples after filtering: {len(schnet_df)}")

        if len(schnet_df) == 0:
            print(f"  No data for {target}_{split}")
            continue

        # 计算目标值
        if target in wavelength_targets:
            # 波长转换为能量 (eV)
            target_values = 1240. / schnet_df[target].values
            print(f"  Converting wavelength to energy (1240/{target})")
        else:
            # plqy 和 e 直接使用原始值
            target_values = schnet_df[target].values
            print(f"  Using raw values for {target}")

        # 转换为SchNet数据库
        save_path = f'data/schnet/{db}/{target}_{split}.db'

        get_schnet_data(
            smiles=schnet_df['smiles'].values,
            target=target_values,
            indices=schnet_df.index.values,
            save_path=save_path,
            conf_num=5
        )

        print(f"  Saved to: {save_path}")

print("\n" + "=" * 50)
print("All targets processed successfully!")
print("=" * 50)