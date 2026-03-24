import copy
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import torch

from FLAME.flsf.args import PredictArgs, TrainArgs
from FLAME.dataprocess.flsf.data import MoleculeDataset, get_data_from_smiles
from FLAME.dataprocess.flsf.features import set_extra_atom_fdim, set_extra_bond_fdim
from FLAME.flsf.utils import load_args, load_checkpoint, load_scalers, update_prediction_args


def _build_predict_args(model_path: str, no_cuda: bool = False) -> PredictArgs:
    pred_args = [
        '--test_path', '',
        '--checkpoint_dir', model_path,
        '--preds_path', '',
        '--number_of_molecules', '2'
    ]
    if no_cuda:
        pred_args.append('--no_cuda')

    return PredictArgs().parse_args(pred_args)


def _clone_batch_graphs(batch_graphs):
    return [copy.deepcopy(batch_graph) for batch_graph in batch_graphs]


def _mask_atom_features(batch_graph, mol_index: int, atom_index: int) -> None:
    a_start, a_size = batch_graph.a_scope[mol_index]
    if atom_index < 0 or atom_index >= a_size:
        raise IndexError(f'Atom index {atom_index} is out of range for molecule with {a_size} atoms.')

    global_atom_index = a_start + atom_index
    batch_graph.f_atoms[global_atom_index].zero_()

    b_start, b_size = batch_graph.b_scope[mol_index]
    b_end = b_start + b_size
    for bond_index in range(b_start, b_end):
        if int(batch_graph.b2a[bond_index]) == global_atom_index:
            batch_graph.f_bonds[bond_index, :batch_graph.atom_fdim] = 0


def _prepare_masked_batch(smiles_pair: Sequence[str],
                          atom_indices: Sequence[int],
                          features_generator: Optional[List[str]]) -> MoleculeDataset:
    repeated_smiles = [list(smiles_pair) for _ in range(len(atom_indices) + 1)]
    dataset = get_data_from_smiles(
        smiles=repeated_smiles,
        skip_invalid_smiles=False,
        features_generator=features_generator
    )

    if len(dataset) == 0 or any(mol is None for mol in dataset[0].mol):
        raise ValueError('Invalid fluorophore or solvent SMILES.')

    return dataset


def _predict_masked_batch(model,
                          scaler,
                          dataset: MoleculeDataset,
                          masked_atom_indices: Sequence[int]) -> np.ndarray:
    batch_graphs = _clone_batch_graphs(dataset.batch_graph())
    for masked_row, atom_index in enumerate(masked_atom_indices, start=1):
        _mask_atom_features(batch_graphs[0], masked_row, atom_index)

    features_batch = dataset.features()
    mol_adj_batch = dataset.adj_features()
    mol_dist_batch = dataset.dist_features()
    mol_clb_batch = dataset.clb_features()

    with torch.no_grad():
        batch_preds = model(
            batch_graphs,
            features_batch,
            mol_adj_batch,
            mol_dist_batch,
            mol_clb_batch
        )

    batch_preds = np.asarray(batch_preds.data.cpu().numpy(), dtype=float)
    if scaler is not None:
        batch_preds = scaler.inverse_transform(batch_preds)
        batch_preds = np.asarray(batch_preds, dtype=float)

    return batch_preds


def _get_atom_metadata(mol: Chem.Mol, atom_index: int) -> dict:
    atom = mol.GetAtomWithIdx(atom_index)
    return {
        'atom_index': atom_index,
        'atom_symbol': atom.GetSymbol(),
        'atomic_num': atom.GetAtomicNum(),
        'is_aromatic': bool(atom.GetIsAromatic()),
        'formal_charge': atom.GetFormalCharge(),
        'degree': atom.GetDegree()
    }


def _score_to_color(score: float, max_abs_score: float) -> tuple:
    if max_abs_score <= 0:
        return (0.85, 0.85, 0.85)

    intensity = min(abs(score) / max_abs_score, 1.0)
    fade = 0.75 * intensity
    if score >= 0:
        return (1.0, 1.0 - fade, 1.0 - fade)

    return (1.0 - fade, 1.0 - fade, 1.0)


def draw_flsf_atom_attribution(smiles: str,
                               attribution: pd.DataFrame,
                               output_path: str,
                               score_column: str = 'contribution',
                               legend: str = '') -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid fluorophore SMILES: {smiles}')

    scores = attribution.set_index('atom_index')[score_column].to_dict()
    max_abs_score = max((abs(float(v)) for v in scores.values()), default=0.0)

    draw_mol = Chem.Mol(mol)
    highlight_atoms = []
    highlight_colors = {}
    highlight_radii = {}

    for atom in draw_mol.GetAtoms():
        atom_index = atom.GetIdx()
        score = float(scores.get(atom_index, 0.0))
        atom.SetProp('atomNote', f'{atom_index}:{score:.2f}')
        highlight_atoms.append(atom_index)
        highlight_colors[atom_index] = _score_to_color(score, max_abs_score)
        highlight_radii[atom_index] = 0.32 + 0.22 * (abs(score) / max_abs_score if max_abs_score else 0.0)

    drawer = rdMolDraw2D.MolDraw2DCairo(900, 700)
    options = drawer.drawOptions()
    options.addAtomIndices = False
    options.legendFontSize = 24
    drawer.DrawMolecule(
        draw_mol,
        legend=legend,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=highlight_radii
    )
    drawer.FinishDrawing()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(drawer.GetDrawingText())

    return output_path


def flsf_atom_explain(model_path: str,
                      fluorophore_smiles: str,
                      solvent_smiles: str,
                      atom_indices: Optional[Iterable[int]] = None,
                      output_csv: str = '',
                      output_png: str = '',
                      no_cuda: bool = False,
                      include_generic_aliases: bool = True) -> pd.DataFrame:
    """
    Performs atom-level masking analysis for FLSF by zeroing each selected atom's graph features
    while preserving the molecular topology, then comparing predictions before and after masking.

    Positive contribution values mean the presence of that atom pushes the prediction upward.
    For wavelength tasks such as lambda_em, this corresponds to a redshift contribution.
    """
    if not os.path.exists(model_path):
        raise ValueError(f'Model: {model_path} not exist')

    smiles_pair = [fluorophore_smiles, solvent_smiles]
    fluorophore = Chem.MolFromSmiles(fluorophore_smiles)
    if fluorophore is None:
        raise ValueError(f'Invalid fluorophore SMILES: {fluorophore_smiles}')

    all_atom_indices = list(range(fluorophore.GetNumAtoms()))
    if atom_indices is None:
        masked_atom_indices = all_atom_indices
    else:
        masked_atom_indices = list(atom_indices)
        invalid = sorted(set(masked_atom_indices) - set(all_atom_indices))
        if invalid:
            raise ValueError(f'Atom indices out of range: {invalid}')

    args = _build_predict_args(model_path=model_path, no_cuda=no_cuda)
    train_args = load_args(args.checkpoint_paths[0])
    update_prediction_args(predict_args=args, train_args=train_args)
    args: TrainArgs

    if args.atom_descriptors == 'feature':
        set_extra_atom_fdim(train_args.atom_features_size)
    if args.bond_features_path is not None:
        set_extra_bond_fdim(train_args.bond_features_size)

    dataset = _prepare_masked_batch(
        smiles_pair=smiles_pair,
        atom_indices=masked_atom_indices,
        features_generator=args.features_generator
    )

    num_tasks = train_args.num_tasks
    sum_preds = np.zeros((len(masked_atom_indices) + 1, num_tasks))

    for checkpoint_path in args.checkpoint_paths:
        model = load_checkpoint(checkpoint_path, device=args.device)
        scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler = load_scalers(checkpoint_path)

        if args.features_scaling or train_args.atom_descriptor_scaling or train_args.bond_feature_scaling:
            dataset.reset_features_and_targets()
            if args.features_scaling:
                dataset.normalize_features(features_scaler)
            if train_args.atom_descriptor_scaling and args.atom_descriptors is not None:
                dataset.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
            if train_args.bond_feature_scaling and args.bond_features_size > 0:
                dataset.normalize_features(bond_feature_scaler, scale_bond_features=True)

        model_preds = _predict_masked_batch(
            model=model,
            scaler=scaler,
            dataset=dataset,
            masked_atom_indices=masked_atom_indices
        )
        sum_preds += np.asarray(model_preds, dtype=float)

    avg_preds = sum_preds / len(args.checkpoint_paths)
    baseline = avg_preds[0]
    masked_preds = avg_preds[1:]

    records = []
    task_names = train_args.task_names
    for row_index, atom_index in enumerate(masked_atom_indices):
        row = _get_atom_metadata(fluorophore, atom_index)
        importance = 0.0

        for task_index, task_name in enumerate(task_names):
            masked_pred = float(masked_preds[row_index, task_index])
            baseline_pred = float(baseline[task_index])
            contribution = baseline_pred - masked_pred
            row[f'baseline_{task_name}'] = baseline_pred
            row[f'masked_{task_name}'] = masked_pred
            row[f'contribution_{task_name}'] = contribution
            importance = max(importance, abs(contribution))

        if include_generic_aliases and len(task_names) == 1:
            task_name = task_names[0]
            row['baseline_prediction'] = row[f'baseline_{task_name}']
            row['masked_prediction'] = row[f'masked_{task_name}']
            row['contribution'] = row[f'contribution_{task_name}']

        row['abs_contribution'] = importance
        records.append(row)

    result = pd.DataFrame(records)
    if len(result) > 0:
        result['importance_rank'] = result['abs_contribution'].rank(method='dense', ascending=False).astype(int)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        result.to_csv(output_csv, index=False)

    if output_png:
        legend = f'Baseline: {baseline[0]:.3f}' if len(task_names) == 1 else ''
        draw_flsf_atom_attribution(
            smiles=fluorophore_smiles,
            attribution=result,
            output_path=output_png,
            score_column='contribution' if len(task_names) == 1 else f'contribution_{task_names[0]}',
            legend=legend
        )

    return result


def flsf_atom_explain_all_targets(model_root: str,
                                  fluorophore_smiles: str,
                                  solvent_smiles: str,
                                  model_prefix: str = 'FluoDB',
                                  targets: Sequence[str] = ('abs', 'emi', 'plqy', 'e'),
                                  atom_indices: Optional[Iterable[int]] = None,
                                  output_csv: str = '',
                                  no_cuda: bool = False) -> pd.DataFrame:
    """
    Runs atom-level masking explainability for multiple targets and merges the results into a single table.
    Each row corresponds to one atom of the fluorophore, with target-specific prediction and contribution columns.
    """
    merged: Optional[pd.DataFrame] = None
    target_results: Dict[str, pd.DataFrame] = {}

    metadata_columns = [
        'atom_index',
        'atom_symbol',
        'atomic_num',
        'is_aromatic',
        'formal_charge',
        'degree'
    ]

    for target in targets:
        model_path = os.path.join(model_root, f'{model_prefix}_{target}')
        target_df = flsf_atom_explain(
            model_path=model_path,
            fluorophore_smiles=fluorophore_smiles,
            solvent_smiles=solvent_smiles,
            atom_indices=atom_indices,
            no_cuda=no_cuda,
            include_generic_aliases=False
        ).copy()

        target_results[target] = target_df
        keep_columns = metadata_columns + [
            f'baseline_{target}',
            f'masked_{target}',
            f'contribution_{target}',
            'abs_contribution',
            'importance_rank'
        ]
        target_df = target_df.loc[:, keep_columns].rename(columns={
            'abs_contribution': f'abs_contribution_{target}',
            'importance_rank': f'importance_rank_{target}'
        })

        if merged is None:
            merged = target_df
        else:
            merged = merged.merge(target_df, on=metadata_columns, how='inner')

    if merged is None:
        raise ValueError('No targets were provided.')

    merged.insert(0, 'solvent_smiles', solvent_smiles)
    merged.insert(0, 'fluorophore_smiles', fluorophore_smiles)

    overall_abs_columns = [f'abs_contribution_{target}' for target in targets]
    merged['max_abs_contribution'] = merged[overall_abs_columns].max(axis=1)
    merged['overall_importance_rank'] = merged['max_abs_contribution'].rank(
        method='dense',
        ascending=False
    ).astype(int)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        merged.to_csv(output_csv, index=False)

    return merged
